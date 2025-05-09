# Integration tests

## Prerequisites

To run integration tests, ```pystache``` and ```ansible``` are needed.

## Test config

```yaml
controller:
  host: host-1
  ip: 1.2.3.4
agents: # list of hosts used to deploy agents
  - host: host-1
  - host: host-2
```

**Note: controller host is mandatory, also at least one agent host should be provided.**

Controller's IP address will be used by the agents.
Therefore, the config must contain an IP that is reachable from
the hosts of agents.

## Create test file

Each test file should have this structure:

```yaml
controller:
  policy: static
  config: examples/configs/controller.yaml
work_dir: ~/projects/infscale
env_activate_command: conda activate pytorch-dev
log_level: INFO
steps:
  - processes:
    - cmd: start job
      args: examples/resnet152/static/linear.yaml
    - type: other
      cmd: sleep 3

  - processes:
    - cmd: status job
      args: job1
      condition:
        - complete
        - failed

```

### Top-level attributes

- ```controller```: this can have two attributes, policy and config:

  - ```policy``` is the deployment policy, where default is ```random```.
  - ```config``` is the path to the controller's config file.

- ```work_dir```: working directory on the remote machine

- ```env_activate_command```: command to activate the python environment
on the remote machine.

- ```log_level```: change ```infscale``` log level, default is ```WARNING```

- ```steps```: a list of steps that will be executed. Think of these steps as
a way to instruct the system on how to perform certain tasks.

### Attributes for steps

- ```processes```: a list of processes that will be executed for each step
(e.g.: start job, then, status job, etc)

- ```cmd```: command that will be executed for each process,
there are two types of commands ```infscale_cmd (default)``` or ```other```.

  - ```infscale_cmd``` will execute ```infscale``` specific commands
(start job, update job, stop job). This command needs some ```args```.
For example, when starting a job, ```args``` should be the path to the job config
YAML file or when checking the status of a job, ```args``` will be the ```job_id```.

  - ```other``` command will execute other built-in or third-party commands
available in the system. The difference between ```infscale_cmd``` and
```other``` is that ```other``` expects the program name to be specified.
For example, ```sleep``` is a separate program in ```sleep 3```;
it's not a command argument or something similar.

So, by following the YAML structure from above,
a multitude of test scenarios can be created.

In order to start a test, run the following command:

```bash
python run_tests.py --config config.yaml--test tests/static_single_host_linear.yaml
```

New tests can be created using the structure from above,
or run the ones that are present in /tests folder.
