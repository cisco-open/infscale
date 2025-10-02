from infscale.configs.job import JobConfig
import pytest

from tests.configs.conftest import remove_pipeline_test_cases


@pytest.mark.parametrize("config,worker_ids,expected", remove_pipeline_test_cases)
def test_remove_pipeline(config: JobConfig, worker_ids: set[str], expected: JobConfig):
    new_cfg = JobConfig.remove_pipeline(config, worker_ids)
    assert new_cfg.flow_graph == expected.flow_graph
    assert len(new_cfg.workers) == len(expected.workers)
