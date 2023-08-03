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

from torchvision.models import vgg16, VGG16_Weights

sys.path.append(".")
from inference_pipeline import RR_CNNPipeline, CNNShardBase

#########################################################
#                   Run RPC Processes                   #
#########################################################

num_batches = 100
batch_size = 64
image_w = 224
image_h = 224


def run_master(split_size, num_workers, partitions, shards, pre_trained = False):

    file = open("./vgg16_mild_uneven.csv", "a")
    original_stdout = sys.stdout
    sys.stdout = file

    if pre_trained:
        net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    else:
        net = vgg16()

    workers = ["worker{}".format(i + 1) for i in range(num_workers)]
    layers = [
        *net.features,
        net.avgpool,
        lambda x: torch.flatten(x, 1),
        *net.classifier
    ]
    cuda_list = ["cuda:{}".format(i) for i in range(1, 4)]
    model = RR_CNNPipeline(split_size, workers, layers = layers, partitions=partitions, devices=cuda_list + cuda_list + cuda_list, shards=shards)

    # generating inputs
    inputs = torch.randn(batch_size, 3, image_w, image_h)
    
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
    file = open("./vgg16_mild_uneven.log", "w")
    original_stdout = sys.stdout
    sys.stdout = file
    partitions = [3, 6]
    combo = [[1, 2, 3], [1, 1, 2, 3], [1, 1, 2, 2, 3], [1, 1, 2, 2, 3, 3]]
    for shards in combo:
        print("Placement:", shards)
        world_size = len(shards) + 1
        for split_size in [1, 2, 4, 8]:
            tik = time.time()
            mp.spawn(run_worker, args=(world_size, split_size, partitions, shards, True), nprocs=world_size, join=True)
            tok = time.time()
            print(f"size of micro-batches = {split_size}, end-to-end execution time = {tok - tik} s")

    sys.stdout = original_stdout