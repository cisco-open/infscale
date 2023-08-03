import os
import threading
import time
import sys
from functools import wraps

import torch
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torchvision.models.resnet import Bottleneck, resnet50, ResNet50_Weights

sys.path.append(".")
from inference_pipeline import CNNShardBase, RR_CNNPipeline


#########################################################
#                   Run RPC Processes                   #
#########################################################

num_batches = 100
num_classes = 1000
batch_size = 120
image_w = 128
image_h = 128


def run_master(split_size, num_workers, partitions, shards, pre_trained = False):

    file = open("./resnet50_even.csv", "a")
    original_stdout = sys.stdout
    sys.stdout = file

    if pre_trained == True:
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        net = resnet50()

    workers = ["worker{}".format(i + 1) for i in range(num_workers)]
    layers = [
        net.conv1,
        net.bn1,
        net.relu,
        net.maxpool,
        *net.layer1,
        *net.layer2,
        *net.layer3,
        *net.layer4,
        net.avgpool,
        lambda x: torch.flatten(x, 1),
        net.fc
    ]
    devices = ["cuda:{}".format(i) for i in range(4)]

    model = RR_CNNPipeline(split_size, workers, layers, partitions, shards, devices)

    # generating inputs
    inputs = torch.randn(batch_size, 3, image_w, image_h, dtype=next(net.parameters()).dtype)
    
    print("{}".format(shards),end=", ")
    tik = time.time()
    for i in range(num_batches):
        outputs = model(inputs)

    tok = time.time()
    print(f"{split_size}, {tok - tik}, {(num_batches * batch_size) / (tok - tik)}")

    sys.stdout = original_stdout


def run_worker(rank, world_size, split_size, partitions, shards, pre_trained = False):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=300)

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_master(split_size, num_workers=world_size - 1, partitions=partitions, shards=shards, pre_trained=pre_trained)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__=="__main__":
    file = open("./resnet50_even.log", "w")
    original_stdout = sys.stdout
    sys.stdout = file

    partitions = [6]
    placements = [[1, 2], [1, 1, 2, 2], [1, 1, 2], [1, 2, 2]]
    for shards in placements:
        world_size = len(shards) + 1
        print("Placement:", shards)
        for split_size in [1, 2, 4, 8]:
            tik = time.time()
            mp.spawn(run_worker, args=(world_size, split_size, partitions, shards), nprocs=world_size, join=True)
            tok = time.time()
            print(f"size of micro-batches = {split_size}, end-to-end execution time = {tok - tik} s")

    sys.stdout = original_stdout