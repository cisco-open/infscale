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


#########################################################
#           Define Model Parallel ResNet50              #
#########################################################

# In order to split the ResNet50 and place it on different pipeline stages, we
# implement it in two model shards. The ResNetShardBase class defines common
# attributes and methods shared by two shards. ResNetShard1 and ResNetShard2
# contain two partitions of the model layers respectively.


num_classes = 1000


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNetShardBase(nn.Module):
    def __init__(self, device, layers, *args, **kwargs):
        super(ResNetShardBase, self).__init__()

        self.lock = threading.Lock()
        self.layers = [m.to(device) for m in layers]
        self.device = device
    
    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self.lock:
            for m in self.layers:
                x = m(x)
        return x.cpu()
 
class RRDistResNet(nn.Module):
    """
    Assemble multiple ResNet parts as an nn.Module and define pipelining logic
    May have several replicas for one ResNet part
    Use round-robin to schedule workload across replicas
    """
    def __init__(self, split_size, workers, layers, partitions, shards, devices, *args, **kwargs):
        super(RRDistResNet, self).__init__()

        self.split_size = split_size
        layer_partitions = []
        partitions = [0] + partitions + [len(partitions)]
        for i in range(len(partitions) - 1):
            layer_partitions.append(layers[partitions[i]:partitions[i+1]])

        assert len(workers) >= len(shards)
        assert len(devices) >= len(shards)
        self.shards_ref = [[] for i in range(len(layer_partitions))]

        # place shards according to configuration
        for i in range(len(shards)):
            shard_id = shards[i] - 1
            shard_layers = layer_partitions[shard_id]
            rref = rpc.remote(
                workers[i],
                ResNetShardBase,
                args = (devices[i], shard_layers, ) + args,
                kwargs = kwargs
            )
            self.shards_ref[shard_id].append(rref)

    def forward(self, xs):
        # Split the input batch xs into micro-batches, and collect async RPC
        # futures into a list
        out_futures = []
        p = [0] * len(self.shards_ref)
        for x in iter(xs.split(self.split_size, dim=0)):
            x_rref = RRef(x)
            for i in range(len(self.shards_ref) - 1):
                x_rref = self.shards_ref[i][p[i]].remote().forward(x_rref)
                p[i] = (p[i] + 1) % len(self.shards_ref[i])

            i = -1
            z_fut = self.shards_ref[i][p[i]].rpc_async().forward(x_rref)
            p[i] = (p[i] + 1) % len(self.shards_ref[i])
            out_futures.append(z_fut)

        # collect and cat all output tensors into one tensor.
        return torch.cat(torch.futures.wait_all(out_futures))


#########################################################
#                   Run RPC Processes                   #
#########################################################

num_batches = 100
batch_size = 120
image_w = 128
image_h = 128


def run_master(split_size, num_workers, partitions, shards, pre_trained = False):

    file = open("./resnet50_even.csv", "a")
    original_stdout = sys.stdout
    sys.stdout = file

    if pre_trained == True:
        net = resnet50(ResNet50_Weights.IMAGENET1K_V2)
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

    model = RRDistResNet(split_size, workers, layers, partitions, shards, devices)

    one_hot_indices = torch.LongTensor(batch_size) \
                           .random_(0, num_classes) \
                           .view(batch_size, 1)

    # generating inputs
    inputs = torch.randn(batch_size, 3, image_w, image_h, dtype=next(net.parameters()).dtype)
    labels = torch.zeros(batch_size, num_classes) \
                    .scatter_(1, one_hot_indices, 1)
    
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