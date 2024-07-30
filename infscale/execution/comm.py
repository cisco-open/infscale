"""Communication module."""

import torch
from infscale import get_logger
from torch.distributed.world_communicator import WorldCommunicator

# from https://github.com/SymbioticLab/Oobleck/blob/develop/oobleck/execution/utils.py#L4-L18
ID_TO_DTYPE = [
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.float16,
    torch.bfloat16,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]

DTYPE_TO_ID = {dtype: id_ for id_, dtype in enumerate(ID_TO_DTYPE)}


logger = get_logger()


class TensorSender:
    """Tensor sender class.

    This class maintains state related to sending tensors and sends tensors.
    """

    def __init__(
        self,
        communicator: WorldCommunicator,
        world_name: str,
        rank: int,
        device: torch.device,
    ):
        """Initialize tensor sender instance."""
        self.communicator = communicator
        self.world_name = world_name
        self.rank = rank  # destination's rank
        self.device = device

        self.sent_tensor_meta = False

    async def _send(self, tensor: torch.Tensor) -> None:
        await self.communicator.send(tensor, self.rank, self.world_name)

    async def send(self, tensors: tuple[torch.Tensor], seqno: int) -> None:
        """Send tensors to destination rank.

        seqno: the seqno of a tensor; will be used to keep track of tensors
        traversing a pipeline.
        """

        async def _send_tensor_meta(tensors: tuple[torch.Tensor]) -> None:
            """
            Send meta data for tensor.

            sending order of the meta data:
            t_dim -> t_dtype -> t_shape
            """
            count = torch.LongTensor(data=[len(tensors)]).to(self.device)
            await self._send(count)

            for tensor in tensors:
                dim = len(tensor.size())
                t_dim = torch.LongTensor(data=[dim]).to(self.device)

                dtype = DTYPE_TO_ID[tensor.dtype]
                t_dtype = torch.LongTensor(data=[dtype]).to(self.device)

                shape = tensor.size()
                t_shape = torch.LongTensor(data=shape).to(self.device)

                # TODO: Make send asynchronous
                await self._send(t_dim)
                await self._send(t_dtype)
                await self._send(t_shape)

        logger.debug("calling send")
        if not self.sent_tensor_meta:
            logger.debug("sending tensor meta data")
            # we only send meta data once
            await _send_tensor_meta(tensors)
            self.sent_tensor_meta = True
            logger.debug("done tensor meta data tx")

        logger.debug("sending tensors")
        for tensor in tensors:
            await self._send(tensor)
        logger.debug("sent tensors")

        seqno = torch.tensor([seqno], dtype=torch.int).to(self.device)
        await self._send(seqno)
        logger.debug(f"sent seqno {seqno}")


class TensorReceiver:
    """TensorReceiver class."""

    def __init__(
        self,
        communicator: WorldCommunicator,
        world_name: str,
        rank: int,
        device: torch.device,
    ):
        """Initialize communication instance."""
        self.communicator = communicator
        self.world_name = world_name
        self.rank = rank  # source's rank
        self.device = device

        self.buffer: torch.Tensor = None

    async def _recv(self, tensor: torch.Tensor):
        await self.communicator.recv(tensor, self.rank, self.world_name)

    async def recv(self) -> tuple[tuple[torch.Tensor], int]:
        """Receive tensors from source rank.

        seqno: the seqno of a tensor; will be used to keep track of tensors
        traversing a pipeline
        """

        async def _create_receive_buffer() -> tuple[torch.Tensor]:
            """Receive menta data for tensor and return allocated buffer.

            receiving order of the meta data:
            t_dim -> t_dtype -> t_shape
            """

            count = torch.LongTensor(data=[0]).to(self.device)
            await self._recv(count)
            num_tensors = count.item()
            tensors: list[torch.Tensor] = []

            for _ in range(num_tensors):
                t_dim = torch.LongTensor(data=[0]).to(self.device)
                await self._recv(t_dim)
                t_dim = t_dim.item()

                t_dtype = torch.LongTensor(data=[0]).to(self.device)
                await self._recv(t_dtype)
                t_dtype = ID_TO_DTYPE[t_dtype.item()]

                t_shape = torch.LongTensor([1] * t_dim).to(self.device)
                await self._recv(t_shape)
                t_shape = t_shape.tolist()

                tensor = torch.zeros(
                    t_shape,
                    device=self.device,
                    dtype=t_dtype,
                    requires_grad=False,
                )
                tensors.append(tensor)

            return tuple(tensors)

        logger.debug("calling recv")
        if self.buffer is None:
            logger.debug("creating a recv buffer")
            # allocate buffer once and reuse it
            self.buffer = await _create_receive_buffer()
            logger.debug("done recv buffer creation")

        recvd: list[torch.Tensor | None] = [None] * len(self.buffer)
        for idx, tensor in enumerate(self.buffer):
            logger.debug(f"receiving tensor {idx}")
            assert torch.is_tensor(tensor)
            await self._recv(tensor)
            recvd[idx] = tensor.clone().detach()

        seqno = torch.LongTensor(data=[0]).to(self.device)
        await self._recv(seqno)
        seqno = seqno.item()
        logger.debug(f"received tensors of seqno {seqno}")

        return tuple(recvd), seqno
