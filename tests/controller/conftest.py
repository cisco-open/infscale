from infscale.common.job_msg import WorkerStatus
from infscale.configs.job import JobConfig, WorkerData

dfs_test_cases = [
    # one complete loop: s-0 → 0-0 → 1-0 → s-0
    (
        JobConfig(
            job_id="job1",
            workers=[
                WorkerData(
                    **{"id": "s-0", "stage": {}, "device": "cpu", "is_server": True}
                ),
                WorkerData(**{"id": "0-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "1-0", "stage": {}, "device": "cpu"}),
            ],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [{"name": "w0", "peers": ["0-0"]}],
                "0-0": [{"name": "w1", "peers": ["1-0"]}],
                "1-0": [{"name": "w2", "peers": ["s-0"]}],
            },
        ),
        {
            "s-0": WorkerStatus.RUNNING,
            "0-0": WorkerStatus.RUNNING,
            "1-0": WorkerStatus.RUNNING,
        },
        False,  # job failed
    ),
    # multiple complete loops: s-0 → 0-0 → 1-0 → s-0, s-0 → 0-1 → 1-1 → s-0
    (
        JobConfig(
            job_id="job1",
            workers=[
                WorkerData(
                    **{"id": "s-0", "stage": {}, "device": "cpu", "is_server": True}
                ),
                WorkerData(**{"id": "0-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "1-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "0-1", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "1-1", "stage": {}, "device": "cpu"}),
            ],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [{"name": "w0", "peers": ["0-0", "0-1"]}],
                "0-0": [{"name": "w1", "peers": ["1-0"]}],
                "1-0": [{"name": "w2", "peers": ["s-0"]}],
                "0-1": [{"name": "w1", "peers": ["1-1"]}],
                "1-1": [{"name": "w2", "peers": ["s-0"]}],
            },
        ),
        {
            "s-0": WorkerStatus.RUNNING,
            "0-0": WorkerStatus.RUNNING,
            "1-0": WorkerStatus.RUNNING,
            "0-1": WorkerStatus.RUNNING,
            "1-1": WorkerStatus.RUNNING,
        },
        False,  # job failed
    ),
    # multiple loops, one broken: s-0 → 0-0 → 1-0 → s-0,  s-0 → 0-1(X) → 1-1 → s-0
    (
        JobConfig(
            job_id="job1",
            workers=[
                WorkerData(
                    **{"id": "s-0", "stage": {}, "device": "cpu", "is_server": True}
                ),
                WorkerData(**{"id": "0-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "1-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "0-1", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "1-1", "stage": {}, "device": "cpu"}),
            ],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [{"name": "w0", "peers": ["0-0", "0-1"]}],
                "0-0": [{"name": "w1", "peers": ["1-0"]}],
                "1-0": [{"name": "w2", "peers": ["s-0"]}],
                "0-1": [{"name": "w1", "peers": ["1-1"]}],
                "1-1": [{"name": "w2", "peers": ["s-0"]}],
            },
        ),
        {
            "s-0": WorkerStatus.RUNNING,
            "0-0": WorkerStatus.RUNNING,
            "1-0": WorkerStatus.RUNNING,
            "0-1": WorkerStatus.FAILED,
            "1-1": WorkerStatus.RUNNING,
        },
        False,  # job failed
    ),
    # multiple loops, all broken: s-0 → 0-0(X) → 1-0 → s-0,  s-0 → 0-1(X) → 1-1 → s-0
    (
        JobConfig(
            job_id="job1",
            workers=[
                WorkerData(
                    **{"id": "s-0", "stage": {}, "device": "cpu", "is_server": True}
                ),
                WorkerData(**{"id": "0-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "1-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "0-1", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "1-1", "stage": {}, "device": "cpu"}),
            ],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [{"name": "w0", "peers": ["0-0", "0-1"]}],
                "0-0": [{"name": "w1", "peers": ["1-0"]}],
                "1-0": [{"name": "w2", "peers": ["s-0"]}],
                "0-1": [{"name": "w1", "peers": ["1-1"]}],
                "1-1": [{"name": "w2", "peers": ["s-0"]}],
            },
        ),
        {
            "s-0": WorkerStatus.RUNNING,
            "0-0": WorkerStatus.FAILED,
            "1-0": WorkerStatus.RUNNING,
            "0-1": WorkerStatus.FAILED,
            "1-1": WorkerStatus.RUNNING,
        },
        True,  # job failed
    ),
    # no loop: s-0 → 0-0 → 1-0 (no back to s-0)
    (
        JobConfig(
            job_id="job1",
            workers=[
                WorkerData(
                    **{"id": "s-0", "stage": {}, "device": "cpu", "is_server": True}
                ),
                WorkerData(**{"id": "0-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "1-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "2-0", "stage": {}, "device": "cpu"}),
            ],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [{"name": "w0", "peers": ["0-0"]}],
                "0-0": [{"name": "w1", "peers": ["1-0"]}],
                "1-0": [{"name": "w2", "peers": []}],
            },
        ),
        {
            "s-0": WorkerStatus.RUNNING,
            "0-0": WorkerStatus.RUNNING,
            "1-0": WorkerStatus.RUNNING,
        },
        True,  # job failed
    ),
    # broken loop due to FAILED node: s-0 → 0-0 → 1-0(X) → s-0
    (
        JobConfig(
            job_id="job1",
            workers=[
                WorkerData(
                    **{"id": "s-0", "stage": {}, "device": "cpu", "is_server": True}
                ),
                WorkerData(**{"id": "0-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "1-0", "stage": {}, "device": "cpu"}),
            ],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [{"name": "w0", "peers": ["0-0"]}],
                "0-0": [{"name": "w1", "peers": ["1-0"]}],
                "1-0": [{"name": "w2", "peers": ["s-0"]}],
            },
        ),
        {
            "s-0": WorkerStatus.RUNNING,
            "0-0": WorkerStatus.RUNNING,
            "1-0": WorkerStatus.FAILED,
        },
        True,
    ),
    # serving server failed s-0(X) → 0-0 → s-0
    (
        JobConfig(
            job_id="job1",
            workers=[
                WorkerData(
                    **{"id": "s-0", "stage": {}, "device": "cpu", "is_server": True}
                ),
                WorkerData(**{"id": "0-0", "stage": {}, "device": "cpu"}),
            ],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [{"name": "w0", "peers": ["0-0"]}],
                "0-0": [{"name": "w1", "peers": ["s-0"]}],
            },
        ),
        {
            "s-0": WorkerStatus.FAILED,
            "0-0": WorkerStatus.RUNNING,
        },
        True,
    ),
]
