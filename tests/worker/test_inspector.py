import pytest

from infscale.worker.pipeline_inspector import PipelineInspector
from tests.worker.conftest import inspector_test_cases


@pytest.mark.parametrize("config,failed_wid,expected", inspector_test_cases)
def test_has_loop(config, failed_wid, expected):
    inspector = PipelineInspector()
    inspector.configure(config)
    result = inspector.get_suspended_worlds(failed_wid)
    assert result == expected
