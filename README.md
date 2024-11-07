# InfScale

## Prerequisites

Python 3.10+ is needed. We recommend to use pyenv to set up an environment.

```bash
pyenv install 3.10.12
pyenv global 3.10.12
```

Note that Python 3.10+ needs openssl1.1.1 and make sure openssl1.1.1+ is
installed in your system.

## Dependencies and Version

* [PyTorch](https://pypi.org/project/torch/2.4.0/) version: `2.4.0`
* [PyMultiWorld](https://pypi.org/project/multiworld/) version: >= `0.2.1`

## Installation

Run the following under the top folder (`infscale`):

```bash
pip install .
```

This will install dependencies as well as InfScale package.

## Running development code

This is useful during local development. As a prerequisite, dependencies should
be resolved.
Thus, it is necessary to install InfScale once
(see [Installation](#installation)).
Once dependencies are resolved, under `infscale` (top folder), run the following
command:

```bash
python -m infscale
```

This command will print out the following (for example):

```text
Usage: python -m infscale [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  start       Start agent / controller / job.
  stop        Stop job.
  update      Update job.
```

## Quickstart

For minimal execution of InfScale, one controller and one agent are needed.
Run controller first and then agent, each on a separate terminal.

```bash
python -m infscale start controller
```

```bash
python -m infscale start agent id123
```

After the agent and controller started successfully, a job can be started by running the following command:

```bash
python -m infscale start job job1 examples/resnet152/single/job_config_rep.yaml
```
Note that:
2. `job1` is the unique job ID - it can be any value and it has to be unique, it will be used to change the state of a job (stop, update)
3.  `examples/resnet152/single/job_config_rep.yaml` is a job specification file.

To see some log messages, add `INFSCALE_LOG_LEVEL=DEBUG` before each of the above command.

## To Run Tests

```bash
python -m pytest
```

## License

[Apache License 2.0](LICENSE).
