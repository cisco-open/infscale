import os
import sys
import time

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from infscale import get_logger
from infscale.config import Config
from infscale.pipeline.cnn import CNNPipeline
from torchvision.models.resnet import ResNet50_Weights, resnet50

#########################################################
#                   Run RPC Processes                   #
#########################################################

num_batches = 100
num_classes = 1000
batch_size = 64
image_w = 224
image_h = 224

BACKEND_GLOO = "gloo"


logger = get_logger(__name__)


def flat_func(x):
    return torch.flatten(x, 1)


def run_leader(config: Config):
    """Run leader."""
    logger.debug("starting leader")
    if config.pre_trained:
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        net = resnet50()

    # with this call, calculation on graidents is disabled
    # since we don't need graidents for inference
    net.eval()

    num_workers = config.partitions.get_num_shards()
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
    while len(device_list) < num_workers:
        device_list += config.devices
    # generating inputs
    inputs = torch.randn(
        batch_size, 3, image_w, image_h, dtype=next(net.parameters()).dtype
    )

    model = None
    if num_workers == 1:
        # no partitioning
        model = net.to(config.devices[0])
    else:
        model = CNNPipeline(
            config.mini_batch_size,
            workers,
            layers,
            config.partitions,
            device_list,
            backend=BACKEND_GLOO,
        )

    logger.debug(f"created cnn pipleine: {model}")
    logger.info(f"{config.partitions.get_name()}")
    tik = time.time()

    logger.debug("starting inference")
    for i in range(num_batches):
        logger.debug(f"before batch id: {i}")
        if num_workers == 1:
            batch = inputs.to(config.devices[0])
        else:
            batch = inputs

        _ = model(batch)
        logger.debug(f"after batch id: {i}")

    tok = time.time()
    logger.info(
        f"{config.mini_batch_size}, {tok - tik}, {(num_batches * batch_size) / (tok - tik)}"
    )


def run_worker(rank, config: Config):
    """Run worker."""
    os.environ["LEADER_ADDR"] = "localhost"
    os.environ["LEADER_PORT"] = "29500"

    world_size = config.partitions.get_num_shards() + 1

    # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=300)

    torch.distributed.init_process_group(
        backend=BACKEND_GLOO, rank=rank, world_size=world_size
    )

    if rank == 0:
        rpc.init_rpc(
            "leader", rank=rank, world_size=world_size, rpc_backend_options=options
        )
        run_leader(config)
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
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    print(config)

    world_size = config.partitions.get_num_shards() + 1

    repeat_times = config.repetition
    mini_batch_size = config.mini_batch_size

    for i in range(repeat_times):
        tik = time.time()
        mp.spawn(run_worker, args=(config,), nprocs=world_size, join=True)
        tok = time.time()
        print(f"mini-batch size: {mini_batch_size}, e2e time: {tok - tik} s")
