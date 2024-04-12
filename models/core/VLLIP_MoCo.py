import torch
import torch.nn as nn
from cluster.Group import Cluster_GPU
import clip

class VLLIP(nn.Module):
    def __init__(self, 
        dim=2048, K=65536,
        m=0.999, T=0.07,
        multi_positive = False,
        positive_selection = 'cluster',
        cluster_num = 10,
        soft_gamma=0.5):
        super(VLLIP, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.dim = dim
        self.multi_positive = multi_positive
        self.forward_fn = self.forward_VLLIP
        self.cluster_num = cluster_num
        self.soft_gamma = soft_gamma
        assert self.cluster_num > 0
        
        # positive selection strategy
        if 'cluster' in positive_selection:
            self.selection_fn = self.get_q_and_k_index_cluster
            self.cluster_obj = Cluster_GPU(self.cluster_num)
        else:
            raise NotImplementedError
        
        clip_model_q, _ = clip.load('ViT-B/32', device=f'cuda:{torch.cuda.current_device()}' ,jit=False)
        clip_model_k, _ = clip.load('ViT-B/32', device=f'cuda:{torch.cuda.current_device()}' ,jit=False)
        self.encoder_q = VLLIP_encoder(clip_model_q.visual)
        self.encoder_k = VLLIP_encoder(clip_model_k.visual)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  
            param_k.requires_grad = False 

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        #keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle
    

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def get_q_and_k_index_cluster(self, embeddings, return_group=False) -> tuple:

        B = embeddings.size(0)
        target_index = list(range(0, B))
        q_index = target_index

        choice_cluster, choice_points = self.cluster_obj(embeddings)
        k_index = []
        for c in choice_cluster:
            k_index.append(int(choice_points[c]))
        if return_group:
            return (q_index, k_index, choice_cluster, choice_points)
        else:
            return (q_index, k_index)


    def forward(self, img_q, img_k):
        """
        Input:
            query , key (images)
        Output:
            logits, targets
        """
        return self.forward_fn(img_q, img_k)


    def forward_VLLIP(self, img_q, img_k):
        # compute query features
        embeddings = self.encoder_q(img_q)
        embeddings = nn.functional.normalize(embeddings, dim=1)
        
        # get q and k index
        index_q, index_k = self.selection_fn(embeddings)
        
        # features of q
        q = embeddings[index_q]

        # compute key features
        with torch.no_grad():  
            # update the key encoder
            self._momentum_update_key_encoder()  

            # shuffle for making use of BN
            #img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.encoder_k(img_k)  
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            #k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        k_ori = k
        k = k[index_k]

        # compute logits
        # positive logits: Nx1
        if self.multi_positive:
            # SCRL Soft-SC 
            k = (k + k_ori) * self.soft_gamma

        
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class VLLIP_encoder(nn.Module):
    def __init__(self,
                 base_encoder,
                 batch_size = 64
                 ):
        super(VLLIP_encoder, self).__init__()
        self.base_encoder = base_encoder
        self.final_layer = nn.Conv1d(batch_size*3, batch_size, kernel_size=1, stride=1, padding=0).cuda()
        self.mlp = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        # ViT-B/32
        self.last_layer = 11
        self.num_heads = 12
         
    def forward(self, images):
        images = images.reshape(images.size()[0]*3, 3, 224, 224)
        x = self.base_encoder(images)
        x = x[:, 0, :]
        x = self.final_layer(x)
        x = self.mlp(x)
        return x