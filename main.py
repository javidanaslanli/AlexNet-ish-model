import os
import time
import argparse
import functools
from dataclasses import dataclass


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import ImageFolder

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

@dataclass
class Config:

    #First Conv Layer

    inp1: int = 3
    out1: int = 32
    kernel1:int = 5
    stride1: int = 1
    padding1: int = 0

    #Local Response Norm

    size: int = 7
    alpha: float = 1e-4
    beta: float = 0.75
    k: int = 4

    #MaxPool

    poolkernel: int = 2
    poolstride: int = 2
        
     #Second Conv Layer

    inp2: int = 32
    out2: int = 64
    kernel2: int = 3
    stride2: int = 1
    padding2: int = 1
    groups2: int = 2

    #Third Conv Layer

    inp3: int = 64
    out3: int = 128
    kernel3: int = 5
    padding3: int = 2

    #Fourth Conv Layer

    inp4: int = 128
    out4: int = 64
    kernel4: int = 3
    padding4: int = 1
    stride4: int = 2
    groups4: int = 2

    #Fifth Conv Layer

    inp5: int = 64
    out5: int = 32
    kernel5: int = 3
    padding5: int = 1
    stride5: int = 2
    groups5: int = 2

    #Linear layers
    l1_in: int = 32 * 7 * 7
    ll_dim: int = 256

class AlexNetishModel(nn.Module):
    def __init__(self, num_classes, args: Config):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=args.inp1, out_channels=args.out1, kernel_size=args.kernel1, stride=args.stride1, padding=args.padding1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=args.size, alpha=args.alpha, beta=args.beta, k=args.k),
            nn.MaxPool2d(kernel_size=args.poolkernel, stride=args.poolstride),
            nn.Conv2d(in_channels=args.inp2, out_channels=args.out2, kernel_size=args.kernel2, stride=args.stride2, padding=args.padding2, groups=args.groups2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=args.size, alpha=args.alpha, beta=args.beta, k=args.k),
            nn.MaxPool2d(kernel_size=args.poolkernel, stride=args.poolstride),
            nn.Conv2d(in_channels=args.inp3, out_channels=args.out3, kernel_size=args.kernel3, padding=args.padding3),
            nn.ReLU(),
            nn.Conv2d(in_channels=args.inp4, out_channels=args.out4, kernel_size=args.kernel4, padding=args.padding4, stride = args.stride4, groups=args.groups4),
            nn.ReLU(),
            nn.Conv2d(in_channels=args.inp5, out_channels=args.out5, stride = args.stride5, kernel_size=args.kernel5, padding=args.padding5, groups=args.groups5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=args.poolkernel, stride=args.poolstride)
)
        self.ffn = nn.Sequential(
            nn.Linear(in_features=args.l1_in, out_features=args.ll_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=args.ll_dim, out_features=num_classes, bias=True)

 )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean = 0.001, std = 0.02)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        output = self.ffn(x)

        return output
    
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_and_validate(args, model, rank, world_size, train_loader, val_loader, optimizer, epoch, sampler):
    start_epoch_time = time.perf_counter_ns()
    model.train()
    loss_fn = nn.CrossEntropyLoss(reduction = 'sum') 
    ddp_train_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)

    for batch_idx, (data, target) in enumerate(train_loader):
        start_time = time.perf_counter_ns()
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        ddp_train_loss[0] += loss.item()
        ddp_train_loss[1] += len(data)
        end_time = time.perf_counter_ns()
        time_taken = (end_time - start_time) / 1e6

        if rank == 0:
            if (batch_idx+1) % 25 == 0:
                print(f'Train Epoch: {epoch} Mini-batch: {batch_idx+1} \tLoss: {loss.item() / len(data):.6f} \tTime: {time_taken:.2f} ms')

    dist.all_reduce(ddp_train_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f'Train Epoch: {epoch} \tLoss: {ddp_train_loss[0] / ddp_train_loss[1]:.6f}')

    # Validation
    model.eval()
    ddp_val_loss = torch.zeros(2).to(rank)
    ddp_val_correct = torch.zeros(1).to(rank)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            loss = loss_fn(output, target)
            ddp_val_loss[0] += loss.item()
            ddp_val_loss[1] += len(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            ddp_val_correct[0] += correct

            if rank == 0:
                if (batch_idx+1) % 15 == 0:
                    print(f'Val Epoch: {epoch} Mini-batch: {batch_idx+1} \tLoss: {loss.item() / len(data):.6f} \tAcc: {100.*correct/len(data):.2f}%')

    dist.all_reduce(ddp_val_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(ddp_val_correct, op=dist.ReduceOp.SUM)

    if rank == 0:
        val_loss = ddp_val_loss[0] / ddp_val_loss[1]
        val_acc = 100. * ddp_val_correct[0] / ddp_val_loss[1]
        print(f'Val Epoch: {epoch} \tLoss: {val_loss:.6f} \tAcc: {val_acc:.2f}%')

    end_epoch_time = time.perf_counter_ns()
    epoch_time_taken = (end_epoch_time - start_epoch_time) / 1e9
    if rank == 0:
        print(f'Epoch {epoch} time: {epoch_time_taken:.2f} seconds')

def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    
    train_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomAutocontrast(p=0.4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
    valid_preprocess = transforms.Compose([
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'valid')
    
    trdata = ImageFolder(traindir, transform = train_preprocess)
    vldata = ImageFolder(valdir, transform = valid_preprocess)
    
    sampler1 = DistributedSampler(trdata, rank=rank, num_replicas = world_size, shuffle=True)
    sampler2 = DistributedSampler(vldata, rank=rank, num_replicas = world_size)
    
    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    val_kwargs = {'batch_size': args.batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2,
                  'pin_memory': True,
                  'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    val_kwargs.update(cuda_kwargs)
    
    train_loader = torch.utils.data.DataLoader(trdata,**train_kwargs)
    val_loader = torch.utils.data.DataLoader(vldata, **val_kwargs)
    
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    
    torch.cuda.set_device(rank)
    
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    
    model = AlexNetishModel(num_classes = args.num_classes, args = Config).to(rank)
    model = FSDP(model)
    
    optimizer = optim.AdamW(model.parameters(), lr = args.lr , weight_decay = args.weight_decay)
    
    scheduler = StepLR(optimizer, step_size = 8, gamma = args.gamma)
    init_start_event.record()
    
    for epoch in range(1, args.epochs + 1):
        train_and_validate(args, model, rank, world_size, train_loader, val_loader, optimizer, epoch, sampler1)
        scheduler.step()

    init_end_event.record()
    
    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        
    if args.save_model:
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "alexnet.pt") 
            
    cleanup() 
        
       

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Alexnet Pytorch')
    parser.add_argument('--data', metavar='DIR',help='path to dataset')
    parser.add_argument('--num_classes', type=int, default=525, metavar='N',
                       help='number of classes in the dataset')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=90, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001,metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--momentum', default=0.7, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)      