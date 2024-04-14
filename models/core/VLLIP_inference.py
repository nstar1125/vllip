import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as TF

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment anything
from MobileSAM.build_sam import sam_model_registry
from MobileSAM.utils.transforms import ResizeLongestSide

class VLLIP_SAM_test_encoder(nn.Module):
    def __init__(self, encoder,
                 dino_config_path,
                 dino_pretrain_path,
                 sam_pretrain_path
                 ):
        super(VLLIP_SAM_test_encoder, self).__init__()
        self.base_encoder = encoder.base_encoder
        self.mlp = encoder.mlp
        #ViT-B/32
        self.last_layer = 11
        self.num_heads = 12
        #GroundedDINO
        self.dino = self.load_dino(dino_config_path, dino_pretrain_path, device=f'cuda:{torch.cuda.current_device()}')
        #MobileSAM
        self.sam = sam_model_registry["vit_t"](checkpoint=sam_pretrain_path).cuda()
    
    def load_dino(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model
    
    def forward(self, images):
        frames = images.reshape(images.size()[0]*3, 3, 224, 224)
        
        #Grounded DINO
        boxes_filt, _, _ = self.get_grounding_output(
                self.dino, frames, "human", 0.45, 0.25)
        #Mobile SAM
        frames = (frames * 255).clamp(0, 255).to(torch.uint8)
        resize_transform = ResizeLongestSide(self.sam.image_encoder.img_size) #1024

        batched_input = []
        for i in range(frames.size()[0]):
            prepared_image = self.prepare_image(frames[i], resize_transform)
            transformed_boxes = self.transform_boxes(boxes_filt[i],frames[i],resize_transform)
            batched_input.append({
                'image': prepared_image,
                'boxes': transformed_boxes,
                'original_size': frames[i].shape[1:] 
            })
        with torch.no_grad():    
            start = 0
            slice = int(len(batched_input)/3)
            outputs = []
            for i in range(3):
                end = start+slice
                output = self.sam(batched_input[start:end], multimask_output=False)
                outputs.extend(output)
                start+=slice

        mask_list = []
        for i, output in enumerate(outputs):
            if output["masks"] != None:
                output["masks"] = output["masks"].cpu()
                output["masks"] = output["masks"].any(dim=0)
                output["masks"] = (~output["masks"]).int()
                mask_list.append(output["masks"])
            else:
                mask_list.append(torch.ones([1, 224, 224]))
            
        torch.cuda.empty_cache()
        mask_list = torch.stack(mask_list)

        #CLIP
        mask_list = mask_list.reshape(images.size()[0],3,224,224)
        pred_masks = torch.mean(mask_list, dim=1, keepdim=True)
        embedding = self.VLLIP_encoder(images.cuda(), pred_masks.cuda(), masking_type='token_masking', masking_block=9)
        return embedding

    @torch.no_grad()
    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, with_logits=True):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        captions = [caption]*image.size()[0]
        images = image.cuda()
        model = model.cuda()
        with torch.no_grad():
            outputs = model(images, captions=captions)

        prediction_logits = outputs["pred_logits"].cpu().sigmoid()  # (bs, nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()  # (bs, nq, 4)
        
        logits_res = [] #length: bs
        boxs_res = [] #length: bs
        phrases_list = [] #length: bs
        tokenizer = model.tokenizer
        for ub_logits, ub_boxes, ub_captions in zip(prediction_logits, prediction_boxes, captions):
            mask = ub_logits.max(dim=1)[0] > box_threshold
            logits = ub_logits[mask]  # logits.shape = (n, 256)
            boxes = ub_boxes[mask]  # boxes.shape = (n, 4)
            logits_res.append(logits.max(dim=1)[0])
            boxs_res.append(boxes)

            tokenized = tokenizer(ub_captions)
            phrases = [
                get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
                for logit
                in logits
            ]
            phrases_list.append(phrases)
        return boxs_res, phrases_list, logits_res,
    
    @torch.no_grad()
    def transform_boxes(self, boxes_filt, image, transform):
        H, W = image.size()[1], image.size()[2] #224, 224
        for i in range(boxes_filt.size(0)): #XYWH -> X1Y1X2Y2 형식 변경
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        boxes_filt = boxes_filt.cpu()
        transformed_boxes = transform.apply_boxes_torch(boxes_filt, image.shape[1:]).cuda()
        return transformed_boxes
    
    @torch.no_grad()
    def prepare_image(self, image, transform):
        image = transform.apply_image(image) #3, 224, 224 -> 1024, 1024, 3
        image = torch.as_tensor(image) #to cuda
        image = image.cuda()
        return image.permute(2, 0, 1).contiguous() #1024, 1024, 3 -> 3, 1024, 1024
    
    @property
    def device(self):
        return self.base_encoder.conv1.weight.device

    @property
    def dtype(self):
        return self.base_encoder.conv1.weight.dtype

    @torch.no_grad()
    def VLLIP_encoder(self, images, pred_masks,  masking_block=None, masking_type='token_masking'):
        if masking_block is None:
            masking_block = self.last_layer

        vit = self.base_encoder
        x = images.type(self.dtype)
        
        x = vit.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = x.permute(0, 2, 1)
        x = torch.cat([vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                     dtype=x.dtype, device=x.device), x], dim=1)

        x = x + vit.positional_embedding.to(x.dtype)
        x = vit.ln_pre(x)
        
        x = x.permute(1, 0, 2)

        L, N, D = x[1:, :, :].size(0), x.size(1), x.size(2)
        size = int(np.sqrt(L))
        assert size * size == L 

        pred_masks = TF.resize(pred_masks.type(torch.float32), (size, size)) # [*, 1, 7, 7]
        
        if masking_type == 'token_masking':
            for block_idx, resblock in enumerate(vit.transformer.resblocks):
                if block_idx >= masking_block:
                    cls = x[:1,:,:]
                    x = x[1:,:,:]
                    x = x.permute(1,2,0)
                    x = x.view(N, D, size, size).contiguous()
                    x = torch.mul(x, pred_masks.expand(-1, x.size()[1], -1, -1))
                    N = x.size(0)
                    x = x.view(N, D, L).contiguous()
                    x = x.permute(2,0,1)
                    x = torch.cat([cls.expand(-1,N,-1), x], dim=0)
                    x = resblock(x)
                    if block_idx == self.last_layer:
                        x = x.permute(1, 0, 2)
                        x = self.base_encoder.ln_post(x[:, 0, :])
                        if self.base_encoder.proj is not None:
                            x = x @ self.base_encoder.proj

                else:
                    x = resblock(x)
        
        x = self.mlp(x)
        return x

class VLLIP_test_encoder(nn.Module):
    def __init__(self, encoder):
        super(VLLIP_test_encoder, self).__init__()
        self.base_encoder = encoder.base_encoder
        self.mlp = encoder.mlp
    def forward(self, images):
        x = self.base_encoder(images)
        x = x[:, 0, :]
        x = self.mlp(x)
        return x
