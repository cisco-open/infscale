import pytest

from infscale.actor.config_diff import get_config_diff_ids
from tests.module.conftest import job_config_diffs


@pytest.mark.parametrize(
    "old_config,new_config,expected_terminate_ids,expected_start_ids,expected_updated_ids",
    job_config_diffs,
)
def test_get_job_diff(
    old_config,
    new_config,
    expected_terminate_ids,
    expected_start_ids,
    expected_updated_ids,
):
    terminate_ids, start_ids, updated_ids = get_config_diff_ids(old_config, new_config)

    assert set(terminate_ids) == set(expected_terminate_ids)
    assert set(start_ids) == set(expected_start_ids)
    assert set(updated_ids) == set(expected_updated_ids)
