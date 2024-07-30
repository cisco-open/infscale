import os, json, yaml
import argparse
import time
import sys
from functools import wraps

import torch
import torch.nn as nn
import torch.distributed
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torchvision.models.resnet import Bottleneck, resnet50, ResNet50_Weights

sys.path.append(".")
from inference_pipeline import CNNShardBase, RR_CNNPipeline, list2csvcell


#########################################################
#                   Run RPC Processes                   #
#########################################################

num_batches = 100
num_classes = 1000
batch_size = 64
image_w = 224
image_h = 224


def flat_func(x):
    return torch.flatten(x, 1)


def run_leader(
    split_size,
    num_workers,
    partitions,
    shards,
    devices,
    pre_trained=False,
    logging=False,
):

    file = open("./resnet50.csv", "a")
    original_stdout = sys.stdout
    sys.stdout = file

    if pre_trained == True:
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        net = resnet50()
    net.eval()

    workers = ["worker{}".format(i + 1) for i in range(num_workers)]

    layers = [
        net.conv1,
        net.bn1,
        net.relu,
        net.maxpool,
        net.layer1,
        net.layer2,
        net.layer3,
        net.layer4,
        net.avgpool,
        flat_func,
        net.fc,
    ]
    device_list = []
    while len(device_list) < len(shards):
        device_list += devices
    # generating inputs
    inputs = torch.randn(
        batch_size, 3, image_w, image_h, dtype=next(net.parameters()).dtype
    )

    if len(shards) == 0:
        # no partitioning
        model = net.to(devices[0])
    else:
        model = RR_CNNPipeline(
            split_size,
            workers,
            layers,
            partitions,
            shards,
            device_list,
            logging=logging,
        )

    print("{}".format(list2csvcell(shards)), end=", ")
    tik = time.time()
    for i in range(num_batches):
        if len(shards) == 0:
            batch = inputs.to(devices[0])
        else:
            batch = inputs

        outputs = model(batch)

    tok = time.time()
    print(f"{split_size}, {tok - tik}, {(num_batches * batch_size) / (tok - tik)}")

    sys.stdout = original_stdout


def run_worker(
    rank,
    world_size,
    split_size,
    partitions,
    shards,
    devices,
    pre_trained=False,
    logging=False,
):
    # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=300)

    if rank == 0:
        rpc.init_rpc(
            "leader", rank=rank, world_size=world_size, rpc_backend_options=options
        )
        run_leader(
            split_size,
            num_workers=world_size - 1,
            partitions=partitions,
            shards=shards,
            devices=devices,
            pre_trained=pre_trained,
            logging=logging,
        )
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options,
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    logfile = open(f"{rank}-proc.log", "w")

    print("Rank: {}, World size: {}".format(rank, world_size), file=logfile, flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("partitions", type=str)
    parser.add_argument("shards", type=str)
    parser.add_argument("devices", type=str)
    parser.add_argument("--repeat_times", type=int, default=5)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--pre_trained", type=int, default=0)
    parser.add_argument("--logging", type=int, default=0)
    args = parser.parse_args()

    print(args, file=logfile, flush=True)

    partitions = json.loads(args.partitions)
    devices = json.loads(args.devices)
    shards = json.loads(args.shards)
    split_size = args.micro_batch_size
    pre_trained = args.pre_trained != 0
    logging = args.logging != 0
    tik = time.time()
    run_worker(
        rank, world_size, split_size, partitions, shards, devices, pre_trained, logging
    )
    tok = time.time()
