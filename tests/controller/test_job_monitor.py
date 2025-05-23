from infscale.controller.job_monitor import JobMonitor
import pytest

from tests.controller.conftest import dfs_test_cases


@pytest.mark.parametrize("config,worker_status,expected", dfs_test_cases)
def test_has_loop(config, worker_status, expected):
    job_mgr = JobMonitor(worker_status)
    job_mgr.init_config(config)
    result = job_mgr.is_job_failed()
    assert result == expected
