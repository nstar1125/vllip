import torch
import torch.nn as nn
from cluster.Group import Cluster_GPU

class VLLIP(nn.Module):
    def __init__(self, dino, sam, clip):
        super(VLLIP,self).__init__()
        self.q_encoder = VLLIP_encoder(dino, sam, clip)
        self.k_encoder = VLLIP_encoder(dino, sam, clip)

    def forward(self, img_q, img_k):
        embeddings = self.q_encoder(img_q)
        print("Hello there!")
        #to be filled ...
        
        logits, labels = 0, 0
        return logits, labels
    
class VLLIP_encoder(nn.Module):
    def __init__(self,dino, sam, clip):
        super().__init__()
        self.dino = dino
        self.sam = sam
        self.clip = clip
         
    def forward(self, img):

        #to be filled ...
        #img -> feature

        return 0
