#!/usr/bin/env python3

import os, time
import argparse

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms
from torchvision.models import AlexNet
from torchvision.models import vgg16

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader


def cifar_trainset(local_rank, dl_path='/tmp/cifar10-data'):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Ensure only one rank downloads.
    # Note: if the download path is not on a shared filesytem, remove the semaphore
    # and switch to args.local_rank
    dist.barrier()
    if local_rank != 0:
        dist.barrier()
    trainset = torchvision.datasets.CIFAR10(root=dl_path,
                                            train=True,
                                            download=True,
                                            transform=transform)
    if local_rank == 0:
        dist.barrier()
    return trainset


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=100,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def train_base(args):
    torch.manual_seed(args.seed)

    # VGG also works :-)
    net = vgg16(num_classes=10)
    # net = AlexNet(num_classes=10)

    trainset = cifar_trainset(args.local_rank)

    engine, _, dataloader, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)

    dataloader = RepeatingLoader(dataloader)
    data_iter = iter(dataloader)

    rank = dist.get_rank()
    gas = engine.gradient_accumulation_steps()

    criterion = torch.nn.CrossEntropyLoss()

    # Inference Code
    print("Inference")
    tik = time.time()
    total_steps = args.steps * engine.gradient_accumulation_steps()
    print("Total Steps:", total_steps)
    print("train batch size", engine.train_batch_size())
    for i in range(total_steps):
        batch = next(data_iter)
        inputs = batch[0].to(engine.device)
        labels = batch[1].to(engine.device)

        outputs = engine(inputs)
    tok = time.time()
    print(f"{tok - tik}, {(total_steps * engine.train_micro_batch_size_per_gpu()) / (tok - tik)}")

    # # Training Code
    # print("Training")
    # tik = time.time()
    # total_steps = args.steps * engine.gradient_accumulation_steps()
    # print("Total Steps:", total_steps)
    # print("train batch size", engine.train_batch_size())
    # step = 0
    # for micro_step in range(total_steps):
    #     batch = next(data_iter)
    #     inputs = batch[0].to(engine.device)
    #     labels = batch[1].to(engine.device)

    #     outputs = engine(inputs)
    #     loss = criterion(outputs, labels)
    #     engine.backward(loss)
    #     engine.step()

    #     if micro_step % engine.gradient_accumulation_steps() == 0:
    #         step += 1
    #         if rank == 0 and (step % 10 == 0):
    #             print(f'step: {step:3d} / {args.steps:3d} loss: {loss}')

    # tok = time.time()
    # print(f"{tok - tik}, {(total_steps * engine.train_micro_batch_size_per_gpu()) / (tok - tik)}")



def join_layers(vision_model):
    layers = [
        *vision_model.features,
        vision_model.avgpool,
        lambda x: torch.flatten(x, 1),
        *vision_model.classifier,
    ]
    return layers


def train_pipe(args, part='parameters'):
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)

    #
    # Build the model
    #

    # both AlexNet and VGG work :-)
    net = vgg16(num_classes=10)
    # net = AlexNet(num_classes=10)
    net = PipelineModule(layers=join_layers(net),
                         loss_fn=torch.nn.CrossEntropyLoss(),
                         num_stages=args.pipeline_parallel_size,
                         partition_method=part,
                         activation_checkpoint_interval=0)

    trainset = cifar_trainset(args.local_rank)
    print("Data Sample:", trainset[0], trainset[0][0].shape)

    engine, _, dataloader, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)

    # # Training Code
    # print("Training")
    # tik = time.time()
    # for step in range(args.steps):
    #     loss = engine.train_batch()

    # tok = time.time()
    # print(f"{tok - tik}, {(args.steps * engine.train_batch_size()) / (tok - tik)}")
    
    # Inference Code
    print("Inference")
    tik = time.time()
    data_iter = iter(RepeatingLoader(dataloader))
    for i in range(args.steps):
        loss = engine.eval_batch(data_iter, compute_loss=False)
    tok = time.time()
    print(f"{tok - tik}, {(args.steps * engine.train_batch_size()) / (tok - tik)}")

if __name__ == '__main__':
    args = get_args()

    deepspeed.init_distributed(dist_backend=args.backend)
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)

    if args.pipeline_parallel_size == 0:
        train_base(args)
    else:
        train_pipe(args)