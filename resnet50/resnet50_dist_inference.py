import os, json, yaml
import threading
import time
import sys
from functools import wraps

import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch.distributed.rpc import RRef
from torchvision.models.resnet import resnet50, ResNet50_Weights

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

def run_master(split_size, num_workers, partitions, shards, devices, pre_trained = False, logging = False):

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
        net.fc
    ]
    device_list = []
    while len(device_list) < len(shards):
        device_list += devices
    # generating inputs
    inputs = torch.randn(batch_size, 3, image_w, image_h, dtype=next(net.parameters()).dtype)

    if len(shards) == 0:
        # no partitioning
        model = net.to(devices[0])
    else:
        model = RR_CNNPipeline(split_size, workers, layers, partitions, shards, device_list, logging=logging)
    
    print("{}".format(list2csvcell(shards)),end=", ")
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


def run_worker(rank, world_size, split_size, partitions, shards, devices, pre_trained = False, logging = False):
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
        run_master(split_size, num_workers=world_size - 1, partitions=partitions, shards=shards, devices=devices, pre_trained=pre_trained, logging=logging)
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
    with open("resnet50_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
        file = open("./resnet50.log", "w")
        open("./resnet50.csv", "w")
        original_stdout = sys.stdout
        sys.stdout = file

        partitions = config["partitions"]
        devices = config["devices"]
        repeat_times = config["repeat_times"]
        placements = config["placements"]
        split_size = config["micro_batch_size"]
        pre_trained = config["pre_trained"] == "True" if "pre_trained" in config else False
        logging = config["logging"] == "True" if "logging" in config else False
        for shards in placements:
            world_size = len(shards) + 1
            print("Placement:", shards)
            for i in range(repeat_times):
                tik = time.time()
                mp.spawn(run_worker, args=(world_size, split_size, partitions, shards, devices, pre_trained, logging), nprocs=world_size, join=True)
                tok = time.time()
                print(f"size of micro-batches = {split_size}, end-to-end execution time = {tok - tik} s")

        sys.stdout = original_stdout