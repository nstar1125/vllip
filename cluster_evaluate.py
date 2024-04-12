import pickle
import os
import torch
import argparse
import time
import json
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader
import clip
from models.core.VLLIP_MoCo import VLLIP_encoder
from models.backbones.visual.resnet import encoder_resnet50

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os
import shutil

class Cluster_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_path, shot_info_path, transform,
        frame_per_shot = 3):
        self.img_path = img_path
        with open(shot_info_path, 'rb') as f:
            self.shot_info = json.load(f)
        self.img_path = img_path
        self.frame_per_shot = frame_per_shot
        self.transform = transform
        self.idx_imdb_map = {}
        data_length = 0
        for info in self.shot_info['test']:
            imdb = info['name']
            for shot in info['label']:
                self.idx_imdb_map[data_length] = (imdb, shot[0], shot[1])
                data_length += 1

    def __len__(self):
        return len(self.idx_imdb_map.keys())

    def _process(self, idx):
        imdb, _id, label = self.idx_imdb_map[idx] #(imdb, shot_num, shot_label)
        img_path_0 =  f'{self.img_path}/{imdb}/shot_{_id}_img_0.jpg'
        img_path_1 =  f'{self.img_path}/{imdb}/shot_{_id}_img_1.jpg'
        img_path_2 =  f'{self.img_path}/{imdb}/shot_{_id}_img_2.jpg'
        img_0 = cv2.cvtColor(cv2.imread(img_path_0), cv2.COLOR_BGR2RGB)
        img_1 = cv2.cvtColor(cv2.imread(img_path_1), cv2.COLOR_BGR2RGB)
        img_2 = cv2.cvtColor(cv2.imread(img_path_2), cv2.COLOR_BGR2RGB)
        data_0 = self.transform(img_0)
        data_1 = self.transform(img_1)
        data_2 = self.transform(img_2)
        data = torch.cat([data_0, data_1, data_2], axis=0)
        label = int(label)
        # According to LGSS[1]
        # [1] https://arxiv.org/abs/2004.02678
        if label == -1:
            label = 1
        return data, label, (imdb, _id)  #img0+img1+img2, shot_label, (imdb, shot_nim)


    def __getitem__(self, idx):
        return self._process(idx)

def get_loader(cfg):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

    _transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])
    dataset = Cluster_Dataset(
        img_path = cfg.shot_img_path,
        shot_info_path = cfg.shot_info_path,
        transform = _transform,
        frame_per_shot = cfg.frame_per_shot,
    )
    loader = DataLoader(
        dataset, batch_size=cfg.bs,  drop_last=False,
        shuffle=False, num_workers=cfg.worker_num, pin_memory=True
    )
    return loader

def get_encoder(cfg, model_name='vllip', weight_path=''):
    encoder = None
    model_name = model_name.lower()
    if model_name == 'vllip':
        clip_model, _ = clip.load('ViT-B/32', device=f'cuda:{torch.cuda.current_device()}' ,jit=False)
        encoder = VLLIP_encoder(clip_model.visual, cfg.bs)
        model_weight = torch.load(weight_path,map_location=torch.device('cpu'))['state_dict']
        pretrained_dict = {}
        for k, v in model_weight.items():
            if k.startswith('module.encoder_k'):
                continue
            if k == 'module.queue' or k == 'module.queue_ptr':
                continue
            if k.startswith('module.encoder_q.base_encoder'):
                k = k[17:]
            pretrained_dict[k] = v
        encoder.load_state_dict(pretrained_dict, strict = False)
        print(f'loaded from {weight_path}')
    elif model_name == 'scrl':
        encoder = encoder_resnet50(weight_path='',input_channel=9)
        model_weight = torch.load(weight_path,map_location=torch.device('cpu'))['state_dict']
        pretrained_dict = {}
        for k, v in model_weight.items():
            # moco loading 
            if k.startswith('module.encoder_k'):
                continue
            if k == 'module.queue' or k == 'module.queue_ptr':
                continue
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                k = k[17:]
            pretrained_dict[k] = v
        encoder.load_state_dict(pretrained_dict, strict = False)
        print(f'loaded from {weight_path}')
    else:
        NotImplementedError
    return encoder


@torch.no_grad()
def get_embeddings(cfg, model, loader, shot_num):
    # dict
    # key: index, value: [(embeddings, label), ...]
    embeddings = {} 
    model.eval()
    
    print(f'total length of dataset: {len(loader.dataset)}')
    print(f'total length of loader: {len(loader)}')
    
    for batch_idx, (data, target, index) in enumerate(loader): #img0+img1+img2/ label=0,1/ index(imdb, shot_num)
        dummy_len = cfg.bs - data.size()[0]
        if dummy_len != 0:
            data = torch.cat([data,torch.zeros(dummy_len,9,224,224)]) #BS 사이즈 마지막 맞추기
        data = data.cuda(non_blocking=True) # ([bs, shot_num, 9, 224, 224])
        data = data.view(-1, 9, 224, 224)
        target = target.view(-1).cuda()
        output = model(data)
        output = output[:(cfg.bs-dummy_len), :] #BS 사이즈 복귀
        for i, key in enumerate(index[0]): #idx, imdb
            if key not in embeddings:
                embeddings[key] = []
            t_emb = output[i*shot_num:(i+1)*shot_num].cpu().numpy()
            t_label =  target[i].cpu().numpy()
            embeddings[key].append((t_emb.copy() ,t_label.copy()))
    return embeddings

def extract_features(cfg):
    encoder = get_encoder(
        cfg,
        model_name=cfg.model_name,
        weight_path=cfg.model_path,
        ).cuda()
    
    loader = get_loader(cfg)
    embeddings = get_embeddings(
        cfg,
        encoder, 
        loader, 
        cfg.shot_num
    )
    return embeddings

def evaluate(cfg, e):
    with open(cfg.shot_info_path, 'rb') as f:
        video_info = json.load(f)
    gt_labels = []
    for info in video_info["test"]:
        imdb = info['name']
        for shot in info['label']:
            gt_labels.append(int(shot[1]))
    gt_labels = np.array(gt_labels)

    embeddings = [np.squeeze(np.array(row[0])) for row in e[imdb]]
    embeddings = np.array(embeddings)

    optimal_k = 5
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    kmeans.fit(embeddings)
    cluster_labels = kmeans.labels_

    ari = adjusted_rand_score(gt_labels, cluster_labels)
    print(f"Adjusted Rand Index (ARI): {ari:.4}")


def to_log(cfg, content, echo=True):
    with open(cfg.log_file, 'a') as f:
        f.writelines(content+'\n')
    if echo: print(content)


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('--shot_info_path', type=str)
    parser.add_argument('--shot_img_path', type=str)
    parser.add_argument('--model_name', type=str, default='vllip')
    parser.add_argument('--frame_per_shot', type=int, default=3)
    parser.add_argument('--shot_num', type=int, default=1)
    parser.add_argument('--worker_num', type=int, default=16)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--save_dir', type=str, default='./inference_output/')
    parser.add_argument('--gpu-id', type=str, default='0')
    cfg = parser.parse_args()

    # select GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id

    return cfg

def main():
    cfg = get_config() # arguments
    embeddings = extract_features(cfg)
    evaluate(cfg, embeddings)
    

if __name__ == '__main__':
    main()
    