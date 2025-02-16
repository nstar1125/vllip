import models.backbones.visual.resnet as resnet
from models.core.VLLIP_MoCo import VLLIP
from models.core.SCRL_MoCo import SCRL
from data.movienet_data import get_train_loader
import torch, os
from utils import to_log

def get_model(cfg):
    model = None
    
    if cfg['model']['SSL'] == 'SCRL':
        encoder = resnet.encoder_resnet50
        assert encoder is not None
        #resnet.encoder_resnet50(weight_path = './pretrain/resnet50-19c8e357.pth')

        model = SCRL(
            base_encoder                = encoder,
            dim                         = cfg['MoCo']['dim'], #512
            K                           = cfg['MoCo']['k'],  #65536
            m                           = cfg['MoCo']['m'], 
            T                           = cfg['MoCo']['t'], 
            mlp                         = cfg['MoCo']['mlp'], 
            encoder_pretrained_path     = cfg['model']['SCRL']['backbone_pretrain'],
            multi_positive              = cfg['MoCo']['multi_positive'],
            positive_selection          = cfg['model']['SCRL']['Positive_Selection'],
            cluster_num                 = cfg['model']['SCRL']['cluster_num'],
            soft_gamma                  = cfg['model']['SCRL']['soft_gamma'],
        ) #SCRL 모델 불러오기
        to_log(cfg, 'model init: SCRL', True)
        if cfg['model']['SCRL']['SyncBatchNorm']:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        to_log(cfg, 'SyncBatchNorm: on' if cfg['model']['SCRL']['SyncBatchNorm'] else 'SyncBatchNorm: off', True)

    elif cfg['model']['SSL'] == 'VLLIP':
        model = VLLIP(
                dim                         = cfg['MoCo']['dim'], 
                K                           = cfg['MoCo']['k'], 
                m                           = cfg['MoCo']['m'], 
                T                           = cfg['MoCo']['t'], 
                multi_positive              = cfg['MoCo']['multi_positive'],
                positive_selection          = cfg['model']['VLLIP']['Positive_Selection'],
                cluster_num                 = cfg['model']['VLLIP']['cluster_num'],
                soft_gamma                  = cfg['model']['VLLIP']['soft_gamma'],
        ) # VLLIP 모델 불러오기
        for name, param in model.named_parameters():
            if 'encoder_q' in name:
                param.requires_grad = True
        
        to_log(cfg, 'model init: VLLIP', True)
        if cfg['model']['VLLIP']['SyncBatchNorm']:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        to_log(cfg, 'SyncBatchNorm: on' if cfg['model']['VLLIP']['SyncBatchNorm'] else 'SyncBatchNorm: off', True)
    else:
        raise NotImplementedError
    return model

def get_loader(cfg):
    train_loader, train_sampler = get_train_loader(cfg)
    return train_loader, train_sampler


def get_criterion(cfg):
    criterion = None
    if cfg['model']['SSL'] == 'VLLIP':
        criterion = torch.nn.CrossEntropyLoss()
    elif cfg['model']['SSL'] == 'SCRL':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
    to_log(cfg, 'criterion init: ' + str(criterion), True)
    return criterion

def get_optimizer(cfg, model):
    optimizer = None
    if cfg['optim']['optimizer'] == 'sgd':
        if cfg['model']['SSL'] == 'VLLIP':
            optim_params = model.parameters()
        elif cfg['model']['SSL'] == 'SCRL':
            optim_params = model.parameters()
        else:
            raise NotImplementedError
        
        optimizer = torch.optim.SGD(optim_params, cfg['optim']['init_lr'],
                                    momentum=cfg['optim']['momentum'],
                                    weight_decay=cfg['optim']['wd'])
    else:
        raise NotImplementedError
    return optimizer

def get_training_stuff(cfg, gpu, ngpus_per_node):
    cfg['optim']['bs'] = int(cfg['optim']['bs'] / ngpus_per_node)
    to_log(cfg, 'shot per GPU: ' + str(cfg['optim']['bs']), True)

    if cfg['data']['clipshuffle']:
        len_per_data = cfg['data']['clipshuffle_len']
    else:
        len_per_data = 1
    assert cfg['optim']['bs'] % len_per_data == 0
    cfg['optim']['bs'] = int(cfg['optim']['bs'] / len_per_data )
    cfg['data']['workers'] = int(( cfg['data']['workers'] + ngpus_per_node - 1) / ngpus_per_node)
    to_log(cfg, 'batch size per GPU: ' + str(cfg['optim']['bs']), True)
    to_log(cfg, 'worker per GPU: ' +  str(cfg['data']['workers']) , True)

    train_loader, train_sampler = get_train_loader(cfg) #Movienet DataLoader, Sampler
    model = get_model(cfg) #vllip
    model.cuda(gpu)
    total_params = 0
    train_params = 0
    print("--------------------------------------------------")
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            train_params += param.numel()
        print(f"Layer: {name}\t\tParam: {param.numel()}\t\tTrainable: {param.requires_grad}")
    print("--------------------------------------------------")
    print(f"Total Parameters #:\t\t{total_params}")
    print(f"Trainable Parameters #:\t\t{train_params}")
    print("--------------------------------------------------")
    model = torch.nn.parallel.DistributedDataParallel(model, 
        device_ids=[gpu], 
        output_device=gpu, 
        find_unused_parameters=True)
    criterion = get_criterion(cfg).cuda(gpu)
    optimizer = get_optimizer(cfg, model) #SGD
    cfg['optim']['start_epoch'] = 0
    if cfg['model']['SSL'] == 'SCRL':
        resume = cfg['model']['SCRL']['resume'] 
    elif cfg['model']['SSL'] == 'VLLIP':
        resume = cfg['model']['SCRL']['resume'] 
    if resume is not None and len(resume) > 1: #불러오기
        if os.path.isfile(resume):
            to_log(cfg, "=> loading checkpoint '{}'".format(resume), True)
            if gpu is None:
                checkpoint = torch.load(resume)
            else:
                loc = f'cuda:{gpu}'
                checkpoint = torch.load(resume, map_location=loc)
            start_epoch = checkpoint['epoch']
            cfg['optim']['start_epoch'] = start_epoch
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            to_log(cfg, "=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']), True)
        else:
            to_log(cfg, "=> no checkpoint found at '{}'".format(resume), True)
            raise FileNotFoundError
         

    assert model is not None \
        and train_loader is not None \
        and criterion is not None \
        and optimizer is not None
    
    return (model, train_loader, train_sampler, criterion, optimizer)

