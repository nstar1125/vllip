import torch
import torch.nn as nn
from cluster.Group import Cluster_GPU
import models.backbones.visual.resnet as resnet
import clip

class VLLIP(nn.Module):
    def __init__(self, dino_path, sam_path, type,
                 dim=2048, K=65536,
                 m=0.999, T=0.07,
                 multi_positive = False,
                 positive_selection = 'cluster',
                 cluster_num = 10,
                 soft_gamma=0.5):
        super(VLLIP,self).__init__()
        self.dino = dino_path
        self.sam = sam_path
        self.type = type

        #CLIP
        clip_model, _ = clip.load('RN50') #ViT-B/32
        clip_visual = clip_model.cuda()
        #clip_visual.conv1 = nn.Conv2d(in_channels=9, out_channels=768, kernel_size=32, stride=32, bias=False)
        clip_visual.visual.conv1 = nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)

        #MoCo
        self.K = K
        self.m = m
        self.T = T
        self.dim = dim
        self.multi_positive = multi_positive
        self.cluster_num = cluster_num
        self.soft_gamma = soft_gamma
        assert self.cluster_num > 0
        
        # positive selection strategy
        if 'cluster' in positive_selection:
            self.selection_fn = self.get_q_and_k_index_cluster
            self.cluster_obj = Cluster_GPU(self.cluster_num)
        else:
            raise NotImplementedError

        #self.q_encoder = resnet.encoder_resnet50(weight_path = './pretrain/resnet50-19c8e357.pth')
        #self.k_encoder = resnet.encoder_resnet50(weight_path = './pretrain/resnet50-19c8e357.pth')
        self.q_encoder =clip_visual
        self.k_encoder =clip_visual

        for param_q, param_k in zip(self.q_encoder.parameters(), self.k_encoder.parameters()):
            param_k.data.copy_(param_q.data)  
            param_k.requires_grad = False 

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, img_q, img_k):
        # compute query features
        embeddings = self.q_encoder.encode_image(img_q, attn=True)
        #embeddings = self.q_encoder(img_q)
        #embeddings = embeddings[:, 0, :]
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
            img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.k_encoder.encode_image(img_k, attn=True)
            #k = self.k_encoder(img_k)  
            #k = k[:, 0, :]
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        k_ori = k
        k = k[index_k]

        # compute logits
        # positive logits: Nx1
        if self.multi_positive:
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


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.q_encoder.parameters(), self.k_encoder.parameters()):
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
    def __init__(self,dino, sam, clip, type='train'):
        super().__init__()
        self.dino = dino
        self.sam = sam
        self.clip = clip
        self.type = type
         
    def forward(self, shot):
        if self.type == 'train':
            visual_features  = self.clip(shot)
            #visual_features  = visual_features[:, 0, :]

        """
        images = shot.reshape(shot.size()[0]*3, 3, 224, 224)
        if self.type=='train':
            masks = torch.ones(shot.size()[0], 3, 224, 224)
            visual_features = self.clip(shot.cuda(), masks.cuda(), masking_type='token_masking', masking_block=9)
        visual_features = visual_features.cpu()
        """
        torch.cuda.empty_cache()
        return visual_features
