from infscale.controller.job_checker import JobChecker
import pytest

from tests.controller.conftest import dfs_test_cases


@pytest.mark.parametrize("config,worker_status,expected", dfs_test_cases)
def test_has_loop(config, worker_status, expected):
    job_checker = JobChecker(worker_status)
    job_checker.setup(config)
    result = job_checker.is_job_failed()
    assert result == expected
