"""Config parser."""

from typing import Optional

import yaml
from pydantic import BaseModel, Field

RAW_KEY_PARTITIONS = "partitions"
RAW_KEY_MICRO_BATCH_SIZE = "micro_batch_size"
RAW_KEY_PRE_TRAINED = "pre_trained"
RAW_KEY_DEVICES = "devices"
RAW_KEY_REPETITION = "repetition"


DEFAULT_MICRO_BATCH_SIZE = 8


class Partitions(BaseModel):
    """Partitions class."""

    index_shards_map: dict

    def get_all(self):
        """Return all pairs of parition index and its shards."""
        index_shards_pairs = []
        for index in sorted(self.index_shards_map.keys()):
            shards = self.index_shards_map[index]
            index_shards_pairs.append((index, shards))

        return index_shards_pairs


class Config(BaseModel):
    """Config class."""

    def __init__(self, config_path: str):
        """Initialize class instance."""
        raw_config = read_config(config_path)
        transformed_config = transform_config(raw_config)

        super().__init__(**transformed_config)

    partitions: Partitions
    micro_batch_size: Optional[int] = Field(default=DEFAULT_MICRO_BATCH_SIZE)
    pre_trained: Optional[bool] = Field(defaut=False)
    devices: Optional[list[str]] = Field(default=[])
    repetition: Optional[int] = Field(default=1)


def read_config(filename: str) -> dict:
    """Read YAML format config."""
    with open(filename) as f:
        return yaml.safe_load(f)


def transform_config(raw_config: dict) -> dict:
    """Transform config."""
    micro_batch_size = raw_config.get(
        RAW_KEY_MICRO_BATCH_SIZE, DEFAULT_MICRO_BATCH_SIZE
    )
    pre_trained = raw_config.get(RAW_KEY_PRE_TRAINED, False)

    devices = raw_config.get(RAW_KEY_DEVICES, [])
    repetition = raw_config.get(RAW_KEY_REPETITION, 1)

    index_shards_map = transform_partitions(raw_config[RAW_KEY_PARTITIONS])

    config_data = {
        RAW_KEY_MICRO_BATCH_SIZE: micro_batch_size,
        RAW_KEY_PRE_TRAINED: pre_trained,
        RAW_KEY_DEVICES: devices,
        RAW_KEY_REPETITION: repetition,
        RAW_KEY_PARTITIONS: index_shards_map,
    }

    return config_data


def transform_partitions(raw_partitions_config: dict):
    """Transform partitions into kv pairs."""
    index_zero_found = False

    index_shards_map = {}
    for raw_index_shards in raw_partitions_config:
        index = raw_index_shards["index"]
        shards = raw_index_shards["shards"]

        if index == 0:
            index_zero_found = True

        if index in index_shards_map:
            raise ValueError(f"Duplicate index {index} specified")

        # TODO: 0 can be used to indicatre dynamic scaling
        if shards <= 0:
            print("WARNING: The number of shards can't be less than 0")
            print("         The value is set to 1")
            shards = 1
        index_shards_map[index] = shards

    if not index_zero_found:
        raise ValueError("config the first partition (index 0) not found")

    return Partitions(index_shards_map=index_shards_map)
