import time
import torch
import torch.nn.parallel
import torch.optim
from utils import AverageMeter, ProgressMeter, to_log, accuracy

from tqdm import tqdm #추가

def train_vllip(gpu, train_loader, model, criterion, optimizer, epoch, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        )
    gradient_clip_val = cfg['optim']['gradient_norm'] # -1

    model.train()
    view_size = (-1, 3 * cfg['data']['frame_size'], 224, 224) #채널 3개 x 프레임 3개
    pivot = time.time()
    for i, data in tqdm(enumerate(train_loader),total=len(train_loader),ncols=50, position=0, leave=True):
        if gpu is not None:
            data_q = data[0].cuda(gpu, non_blocking=True)
            data_k = data[1].cuda(gpu, non_blocking=True)
        data_time.update(time.time() - pivot)
        data_q = data_q.view(view_size) #Q resize
        data_k = data_k.view(view_size) #K resize
        output, target = model(data_q, data_k) 
        loss = criterion(output, target) #Q와 K간 거리 비교

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), target.size(0))
        top1.update(acc1[0], target.size(0))
        top5.update(acc5[0], target.size(0)) #계산

        optimizer.zero_grad()
        loss.backward() #역전파

        if gradient_clip_val > 0: #-1 
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

        optimizer.step() #step descent

        batch_time.update(time.time() - pivot)
        pivot = time.time()
    
    _out = progress.display(i) # epoch 당 한번 출력
    to_log(cfg, _out, True)