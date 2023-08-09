import threading

import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch.distributed.rpc import RRef

class CNNShardBase(nn.Module):
    def __init__(self, device, layers, *args, **kwargs):
        super(CNNShardBase, self).__init__()

        self.lock = threading.Lock()
        self.layers = [m.to(device) for m in layers]
        self.device = device
    
    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self.lock:
            for m in self.layers:
                x = m(x)
        return x.cpu()
    
class RR_CNNPipeline(nn.Module):
    """
    Assemble multiple ResNet parts as an nn.Module and define pipelining logic
    May have several replicas for one ResNet part
    Use round-robin to schedule workload across replicas
    """
    def __init__(self, split_size, workers, layers, partitions, shards, devices, *args, **kwargs):
        super(RR_CNNPipeline, self).__init__()

        self.split_size = split_size
        layer_partitions = []
        partitions = [0] + partitions + [len(layers)]
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
                CNNShardBase,
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