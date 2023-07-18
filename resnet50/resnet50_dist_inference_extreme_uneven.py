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

from torchvision.models.resnet import Bottleneck


#########################################################
#           Define Model Parallel ResNet50              #
#########################################################

# In order to split the ResNet50 and place it on two different workers, we
# implement it in two model shards. The ResNetBase class defines common
# attributes and methods shared by two shards. ResNetShard1 and ResNetShard2
# contain two partitions of the model layers respectively.


num_classes = 1000


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNetBase(nn.Module):
    def __init__(self, block, inplanes, num_classes=1000,
                 groups=1, width_per_group=64, norm_layer=None, *args, **kwargs):
        super(ResNetBase, self).__init__()

        self._lock = threading.Lock()
        self._block = block
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = inplanes
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

    def _make_layer(self, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * self._block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * self._block.expansion, stride),
                norm_layer(planes * self._block.expansion),
            )

        layers = []
        layers.append(self._block(self.inplanes, planes, stride, downsample, self.groups,
                                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * self._block.expansion
        for _ in range(1, blocks):
            layers.append(self._block(self.inplanes, planes, groups=self.groups,
                                      base_width=self.base_width, dilation=self.dilation,
                                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]


class ResNetShard1(ResNetBase):
    """
    The first part of ResNet.
    """
    def __init__(self, device, *args, **kwargs):
        super(ResNetShard1, self).__init__(
            Bottleneck, 64, num_classes=num_classes, *args, **kwargs)

        self.device = device
        if "pre_trained" in kwargs and kwargs["pre_trained"]:
            self.seq = nn.Sequential(kwargs["conv1"], kwargs['bn1'], kwargs["relu"], kwargs["maxpool"]).to(self.device)
        else:
            self.seq = nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                self._norm_layer(self.inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ).to(self.device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.seq(x)
        return out.cpu()


class ResNetShard2(ResNetBase):
    """
    The second part of ResNet.
    """
    def __init__(self, device, *args, **kwargs):
        super(ResNetShard2, self).__init__(
            Bottleneck, 64, num_classes=num_classes, *args, **kwargs)

        self.device = device
        if "pre_trained" in kwargs and kwargs["pre_trained"]:
            self.seq = nn.Sequential(kwargs["seq0"], kwargs["seq1"], kwargs["seq2"], kwargs["seq3"], kwargs["avgpool"]).to(self.device)
            self.fc = kwargs["fc"].to(self.device)
        else:
            self.seq = nn.Sequential(
                self._make_layer(64, 3),
                self._make_layer(128, 4, stride=2),
                self._make_layer(256, 6, stride=2),
                self._make_layer(512, 3, stride=2),
                nn.AdaptiveAvgPool2d((1, 1)),
            ).to(self.device)
            self.fc =  nn.Linear(512 * self._block.expansion, num_classes).to(self.device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.fc(torch.flatten(self.seq(x), 1))
        return out.cpu()


class RRDistResNet50(nn.Module):
    """
    Assemble two ResNet parts as an nn.Module and define pipelining logic
    May have several replicas for one ResNet part
    Use round-robin to schedule workload across replicas
    """
    def __init__(self, split_size, workers, devices, *args, **kwargs):
        super(RRDistResNet50, self).__init__()

        self.split_size = split_size

        if "pre_trained" in kwargs and kwargs["pre_trained"] == True:
            resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
            kwargs['conv1'] = resnet50._modules['conv1']
            kwargs['bn1'] = resnet50._modules['bn1']
            kwargs['relu'] = resnet50._modules['relu']
            kwargs['maxpool'] = resnet50._modules['maxpool']
            kwargs['avgpool'] = resnet50._modules['avgpool']
            kwargs['fc'] = resnet50._modules['fc']
            for i in range(4):
                kwargs['seq{}'.format(i)] = resnet50._modules['layers']._modules[str(i)]

        assert len(workers) > 0
        self.p1_rrefs = [] # list of part 1 references
        self.p2_rrefs = [] # list of part 2 references

        if 'shards' in kwargs:
            # place shards according to configuration
            for i in range(len(workers)):
                shard = ResNetShard1 if kwargs['shards'][i] == 1 else ResNetShard2
                rref = rpc.remote(
                    workers[i],
                    shard,
                    args = (devices[i],) + args,
                    kwargs = kwargs
                )
                if kwargs['shards'][i] == 1:
                    self.p1_rrefs.append(rref)
                else:
                    self.p2_rrefs.append(rref)
        else:
            # place shards uniformly
            for i in range(len(workers)):
                if i % 2 == 0:
                    rref = rpc.remote(
                        workers[i],
                        ResNetShard1,
                        args = (devices[i],) + args,
                        kwargs = kwargs
                    )
                    self.p1_rrefs.append(rref)
                else:
                    rref = rpc.remote(
                        workers[i],
                        ResNetShard2,
                        args = (devices[i],) + args,
                        kwargs = kwargs
                    )
                    self.p2_rrefs.append(rref)

    def forward(self, xs):
        # Split the input batch xs into micro-batches, and collect async RPC
        # futures into a list
        out_futures = []
        p1 = 0
        p2 = 0
        for x in iter(xs.split(self.split_size, dim=0)):
            x_rref = RRef(x)
            y_rref = self.p1_rrefs[p1].remote().forward(x_rref)
            p1 = (p1 + 1) % len(self.p1_rrefs)
            z_fut = self.p2_rrefs[p2].rpc_async().forward(y_rref)
            p2 = (p2 + 1) % len(self.p2_rrefs)
            out_futures.append(z_fut)

        # collect and cat all output tensors into one tensor.
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        return remote_params


#########################################################
#                   Run RPC Processes                   #
#########################################################

num_batches = 100
batch_size = 120
image_w = 128
image_h = 128


def run_master(split_size, num_workers, shards):

    model = RRDistResNet50(split_size, ["worker{}".format(i + 1) for i in range(num_workers)], ["cuda:{}".format(i) for i in range(4)], args=(), shards=shards)

    one_hot_indices = torch.LongTensor(batch_size) \
                           .random_(0, num_classes) \
                           .view(batch_size, 1)

    # generating random inputs
    inputs = torch.randn(batch_size, 3, image_w, image_h)
    labels = torch.zeros(batch_size, num_classes) \
                    .scatter_(1, one_hot_indices, 1)
    
    for i in range(num_batches):
        outputs = model(inputs)


def run_worker(rank, world_size, num_split, shards):
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
        run_master(num_split, num_workers=world_size - 1, shards=shards)
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
    file = open("./resnet50_extreme_uneven.log", "w")
    original_stdout = sys.stdout
    sys.stdout = file
    combo = [[1, 2], [1, 1, 2, 2], [1, 1, 2], [1, 2, 2]]
    for shards in combo:
        world_size = len(shards) + 1
        print("Placement:", shards)
        for num_split in [1, 2, 4, 8]:
            tik = time.time()
            mp.spawn(run_worker, args=(world_size, num_split, shards), nprocs=world_size, join=True)
            tok = time.time()
            print(f"size of micro-batches = {num_split}, execution time = {tok - tik} s, throughput = {(num_batches * batch_size) / (tok - tik)} samples/sec")

    sys.stdout = original_stdout