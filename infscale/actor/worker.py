"""Worker class."""

from multiprocessing.connection import Connection
from typing import Any

from infscale import get_logger
from infscale.config import parse_serve_config
from infscale.execution.pipeline import Pipeline
from infscale.module.dataset import HuggingFaceDataset
from infscale.module.modelir import ModelIR
from infscale.module.zoo import Zoo

logger = get_logger()


class Worker:
    """Worker class."""

    def __init__(self, local_rank: int, conn: Connection, spec: dict[str, Any]):
        """Initialize an instance."""
        self.local_rank = local_rank
        self.conn = conn
        self.spec = parse_serve_config(spec)
        logger.info(f"{self.spec}")

        self.dataset: HuggingFaceDataset = None
        self.ir: ModelIR = None

        self._initialize()

    async def run(self) -> None:
        """Run the worker."""
        logger.info(f"worker {self.local_rank}")

        pipeline = Pipeline(self.spec, self.ir, self.dataset)
        await pipeline.run()

    def _initialize(self) -> None:
        # load model meta info from zoo
        mmd = Zoo.get_model_metadata(self.spec.model)
        (path, name, split) = (
            self.spec.dataset.path,
            self.spec.dataset.name,
            self.spec.dataset.split,
        )

        # load dataset
        self.dataset = HuggingFaceDataset(mmd, path, name, split)

        # load model intermediate representation
        self.ir = ModelIR(mmd, self.dataset.sample)
