import threading
import time
import traceback
from queue import Queue
from typing import List

import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from infscale import get_logger
from infscale.pipeline.util import recv_tensor, send_tensor


class CNNShardBase(nn.Module):
    """CNNShardBase class."""

    def __init__(self, device, layers, *args, **kwargs):
        """Initialize the class."""
        super().__init__()

        self.layers = [m.to(device) if isinstance(m, nn.Module) else m for m in layers]
        self.device = device
        self.shard_index = (
            kwargs["sid"] if "sid" in kwargs else None
        )  # index of the shard in the global set of shards
        self.partition_index = (
            kwargs["pid"] if "pid" in kwargs else None
        )  # index of the partition of layers beared by this shard

        logger_name = f"shard-sid{self.shard_index}-pid{self.partition_index}"
        self.logger = get_logger(logger_name)

        self.logger.info(f"Layers: {self.layers}")

        self.accumulated_processing_time = 0
        self.accumulated_local_trans_time = 0
        self.accumulated_locking_time = 0
        self.forward_times = 0

        self.receive_ranks = dict()
        self.send_ranks = dict()
        self.send_lock = threading.Lock()
        self.control_channel_tag = (
            kwargs["control_channel"] if "control_channel" in kwargs else 0
        )
        self.data_channel_tag = (
            kwargs["data_channel"] if "data_channel" in kwargs else 1
        )
        self.receive_queue = Queue()
        self.stopFlag = False

    def forward(self, x):
        """Run the forwarding logic of all contained layers for one input tensor."""
        t1 = time.time()
        for m in self.layers:
            x = m(x)
        t2 = time.time()
        self.forward_times += 1

        self.logger.debug(
            f"Forward Pass: {self.forward_times}, Data Processing Time: {t2 - t1}s"
        )
        self.accumulated_processing_time += t2 - t1

        return x

    def run(self):
        """The main function of the shard"""
        self.logger.info("Start running")

        ptr = 0
        while not self.stopFlag:
            try:
                index, x = self.receive_queue.get(timeout=3)
            except:
                continue

            x = x.to(self.device)
            x = self.forward(x)

            # the scheduling logic to determine which downstream stage replica to send the output tensor
            # after picked a downstream stage repilca simply put the output to the queue for that destination process
            # current scheduling algorithm: round-robin
            with self.send_lock:
                num_dst_ranks = len(self.send_ranks.keys())
                ptr = ptr % num_dst_ranks
                dst = list(self.send_ranks.keys())[ptr]
                _, sq = self.send_ranks[dst]
                sq.put((index, x))

            ptr += 1

    def stop(self):
        self.stopFlag = True

    def add_send_ranks(self, ranks):
        """Add a number of shards bearing the downstream stage as destination processes to send output tensors."""
        for rank in ranks:
            if rank not in self.send_ranks:
                thread = threading.Thread(target=self._send_to, args=(rank,))
                self.send_ranks[rank] = (thread, Queue())
                thread.start()
                self.logger.debug(f"Add send rank: {rank}")

    def add_receive_ranks(self, ranks):
        """Add a number of shards bearing the upstream stage as source processes to receive input tensors"""
        for rank in ranks:
            if rank not in self.receive_ranks:
                thread = threading.Thread(target=self._recv_from, args=(rank,))
                self.receive_ranks[rank] = thread
                thread.start()
                self.logger.debug(f"Add receive rank: {rank}")

    def _send_to(self, dst):
        """The main function for a sending thread"""
        _, sq = self.send_ranks[dst]
        errorCount = 0
        resend = False

        self.logger.debug(f"Start sending to process {dst}")

        while True:
            if not resend:
                index, t = Queue.get(sq)
            try:
                # data movement should depend on communication backend
                t = t.cpu()

                send_tensor(index, t, dst, self.data_channel_tag)
                self.logger.debug(f"Sended a tensor to process {dst}")

            except:
                self.logger.warn(f"Error occurs when sending to {dst}")

                errorCount += 1
                resend = True
            else:
                errorCount = 0
                resend = False

            if errorCount >= 3:
                break

        self.logger.debug(
            f"Encounter 3 consecutive failures when sending to process {dst}"
        )
        self.logger.debug(f"Shut down the thread sending data to process {dst}")

        self.send_ranks.pop(dst)
        del sq

    def _recv_from(self, src):
        """Receive data from src.

        This is the main function for a receiving thread.
        """
        self.logger.debug(f"Start receiving from process {src}")

        while True:
            try:
                index, t = recv_tensor(src, self.data_channel_tag)
            except:
                self.logger.warn(f"error occurs when receiving from {src}")
                self.logger.warn(traceback.format_exc())

                continue

            self.receive_queue.put((index, t))
            self.logger.debug(f"Received one tensor from {src}")

    def __del__(self):
        self.logger.debug(
            f"Avg Data Processing Time: {self.accumulated_processing_time / self.forward_times} s"
        )


class CNNPipelineCollector:
    """CNNPipelineCollector class."""

    def __init__(self, log_en, *args, **kwargs) -> None:
        """Initialize class CNNPipelineCollector."""
        self.logger = get_logger("pipeline-collector")

        self.receive_ranks = dict()
        self.receive_queue = Queue()
        self.control_channel_tag = (
            kwargs["control_channel"] if "control_channel" in kwargs else 0
        )
        self.data_channel_tag = (
            kwargs["data_channel"] if "data_channel" in kwargs else 1
        )

    def add_receive_ranks(self, ranks):
        """Add a number of shards bearing the upstream stage as source processes to receive input tensors"""
        for rank in ranks:
            if rank not in self.receive_ranks:
                thread = threading.Thread(target=self._recv_from, args=(rank,))
                self.receive_ranks[rank] = thread
                thread.start()
                self.logger.debug(f"Add receive rank: {rank}")

    def _recv_from(self, src):
        """The main function for a receiving thread"""
        self.logger.debug(f"Start receiving from process {src}")

        while True:
            try:
                index, t = recv_tensor(src, self.data_channel_tag)
            except:
                self.logger.warn(f"error occurs when receiving from {src}")
                self.logger.warn(traceback.format_exc())

                continue

            self.receive_queue.put((index, t))

            self.logger.debug(f"Received one tensor from {src}")

    def get_res(self, num) -> List[torch.Tensor]:
        res = []
        for i in range(num):
            index, t = self.receive_queue.get()
            res.append((index, t))

        ret = res
        return [t[1] for t in ret]


class CNNPipeline(nn.Module):
    """
    Assemble multiple ResNet parts as an nn.Module and define pipelining logic
    May have several replicas for one ResNet part
    Use round-robin to schedule workload across replicas
    """

    def __init__(
        self,
        split_size,
        workers,
        layers,
        partitions,
        shards,
        devices,
        backend,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.control_channel_tag = 0
        self.data_channel_tag = 1
        self.comm_backend = backend
        self.split_size = split_size
        self.buffer_device = devices[0]
        self.collector = CNNPipelineCollector(
            log_en=kwargs["logging"] if "logging" in kwargs else False,
            args=(),
            kwargs=kwargs,
        )

        layer_partitions = []
        partitions = [0] + partitions + [len(layers)]
        for i in range(len(partitions) - 1):
            layer_partitions.append(layers[partitions[i] : partitions[i + 1]])

        assert len(workers) >= len(shards)
        assert len(devices) >= len(shards)
        self.shards_refs = [[] for i in range(len(layer_partitions))]
        self.shards_ranks = [[0]] + [[] for i in range(len(layer_partitions))] + [[0]]

        # place shards according to configuration
        for i in range(len(shards)):
            rank = i + 1
            partition_id = shards[i] - 1
            shard_layers = layer_partitions[partition_id]
            kwargs["pid"] = partition_id + 1
            kwargs["sid"] = i
            rref = rpc.remote(
                workers[i],
                CNNShardBase,
                args=(
                    devices[i],
                    shard_layers,
                )
                + args,
                kwargs=kwargs,
            )
            self.shards_refs[partition_id].append(rref)
            self.shards_ranks[partition_id + 1].append(rank)

        # connect shards according to model dependencies
        for i in range(1, len(layer_partitions) + 1):
            for rref in self.shards_refs[i - 1]:
                # add ranks of the previous stage as receive ranks
                rref.rpc_sync().add_receive_ranks(self.shards_ranks[i - 1])
                # add ranks of the next stage as receive ranks
                rref.rpc_sync().add_send_ranks(self.shards_ranks[i + 1])
                rref.rpc_async().run()

        # assign the ranks of shards in the last stage to the result collector
        self.collector.add_receive_ranks(self.shards_ranks[-2])

    def forward(self, xs):
        # Split the input batch xs into micro-batches, and distribute them across the shards of the first stage
        p = 0
        mini_batches = xs.split(self.split_size, dim=0)
        for index, x in enumerate(mini_batches):
            t = x
            if self.comm_backend == "nccl":
                t = t.to(self.buffer_device)
            p = p % len(self.shards_ranks[1])

            send_tensor(index, t, self.shards_ranks[1][p], self.data_channel_tag)
            p += 1

        res = self.collector.get_res(len(mini_batches))
        # collect and cat all output tensors into one tensor.
        return torch.cat(res, dim=0)

    def __del__(self):
        for i in range(len(self.shards_refs)):
            for rref in self.shards_refs[i]:
                rref.rpc_async().stop()
