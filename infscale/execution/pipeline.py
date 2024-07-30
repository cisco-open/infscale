"""Pipeline class."""

import asyncio

import torch
import torch.distributed as dist
from infscale import get_logger
from infscale.config import ServeConfig
from infscale.execution.router import Router
from infscale.execution.stage import Stage
from infscale.execution.world import WorldInfo
from infscale.module.dataset import HuggingFaceDataset
from infscale.module.modelir import ModelIR

logger = get_logger()


class Pipeline:
    """Pipeline class."""

    def __init__(
        self,
        spec: ServeConfig,
        modelir: ModelIR,
        dataset: HuggingFaceDataset,
    ):
        """Initialize pipeline instance."""
        self.stage: Stage = None
        self.spec = spec
        self.world_manager = dist.WorldManager()

        self.device = torch.device(self.spec.device)

        self.world_info_list: list[WorldInfo] = list()

        if "s" in spec.stage.id:  # it's server
            logger.info("I am server and leader")
            self.dataset = dataset
            self._run = self._run_server
            self._predict_fn = modelir.predict_fn
        else:
            logger.info("I am a worker")
            self._run = self._run_worker
            self._initialize_worker(modelir)

    async def _initialize_multiworld(self):
        my_id = self.spec.stage.id

        for world_idx, (k, v) in enumerate(self.spec.flow_graph.items()):
            for wrk_info in v:
                if my_id == k:
                    my_rank = 0
                    other_rank = 1
                elif my_id == wrk_info.peer:
                    my_rank = 1
                    other_rank = 0
                else:
                    continue

                world_name = f"w{world_idx}"
                logger.info(f"initializing world {world_name} with my rank {my_rank}")
                logger.info(f"leader addr={wrk_info.addr}, port={wrk_info.port}")
                await self.world_manager.initialize_world(
                    world_name,
                    my_rank,
                    2,
                    backend=self.spec.backend,
                    addr=wrk_info.addr,
                    port=wrk_info.port,
                )
                data = {"name": world_name, "me": my_rank, "other": other_rank}
                world_info = WorldInfo(**data)
                self.world_info_list.append(world_info)
                logger.debug(f"done initializing {world_name}")

    def _initialize_worker(self, modelir: ModelIR):
        output_parser = modelir.output_parser if self.spec.stage.is_last else None
        layers = modelir.layers[self.spec.stage.start : self.spec.stage.end + 1]

        self.stage = Stage(
            self.spec.stage.id,
            layers,
            device=self.device,
            output_parser=output_parser,
            modelir=modelir,
        )

    async def _server_send(self, router: Router):
        logger.info("start to send requests")
        seqno = 0
        while True:
            batch = self.dataset.next_batch(self.device)
            if batch is None:
                break

            logger.debug(f"sending batch {seqno}")
            # send batch to the first stage
            await router.tx_q.put((batch, seqno))
            seqno += 1

        logger.info("_server_send task done")

    async def _server_recv(self, router: Router, max_seqno: int = -1):
        """
        Receive inference results from the last stage.

        max_seqno: if it's -1, run forever;
                   come out of loop if seqno is max_seqno
        """
        logger.info("start to receive responses")
        seqno = -1
        while max_seqno == -1 or max_seqno != seqno:
            logger.debug("waiting for response")
            outputs, seqno = await router.rx_q.get()
            results = self._predict_fn(outputs)
            logger.info(f"response for {seqno}: {results}")

        logger.info("_server_recv task done")

    async def _run_server(self):
        # create router
        router = Router(
            self.world_manager, self.world_info_list, self.spec, self.device
        )
        router.prepare()

        # TODO: we read data directly from a dataset right now.
        #       in the future, we need to take dataset from stream as well.
        self.dataset.set_micro_batch_size(self.spec.micro_batch_size)
        max_seqno = self.dataset.num_of_batches() - 1

        # send and recv asynchronously
        send_task = asyncio.create_task(self._server_send(router))
        recv_task = asyncio.create_task(self._server_recv(router, max_seqno))
        logger.debug("created _send_request and _recv_resp tasks")

        await asyncio.gather(*[send_task, recv_task])
        logger.info("inference serving is done")

    async def _run_worker(self):
        logger.debug("start to run worker")
        router = Router(
            self.world_manager, self.world_info_list, self.spec, self.device
        )
        router.prepare()
        while True:
            inputs, seqno = await router.rx_q.get()
            logger.debug(f"received input {seqno} from rx_q")

            with torch.inference_mode():
                outputs = self.stage(inputs)
                # if isinstance(inputs, tuple):
                #     logger.debug(f"len(inputs) = {len(inputs)}")
                #     outputs = self.stage(*inputs)
                # else:
                #     outputs = self.stage(inputs)
            logger.debug("got output from stage and put output into tx_q")
            await router.tx_q.put((outputs, seqno))
            logger.debug("put output into tx_q")

    async def run(self):
        """Run pipeline."""
        # initialize multiworld
        await self._initialize_multiworld()
        # run pipeline
        await self._run()
