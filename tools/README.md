# Job Visualizer

This utility visualizes the **flow graph** of a job configuration defined in a YAML file.  
It builds a **directed graph** using [`networkx`](https://networkx.org/) and renders it visually using [`matplotlib`](https://matplotlib.org/).

The tool expects a configuration file that can be parsed into a `JobConfig` object.  
Each worker and its connections (peers) are represented as nodes and directed edges in the graph.

---

## Features

- Visualizes job flow graphs defined in YAML files.  
- Shows workers as nodes and communication paths as directed edges.  
- Adds labels on edges showing world names, addresses, and backends.  
- Arranges workers horizontally by their `stage["start"]` values.  
- Opens the graph interactively with the possibility of saving it as PNG image.

---

## Requirements

Install the required dependencies:

```bash
pip install matplotlib networkx pyyaml
```

---

## Usage

```bash
python visualize_job.py path/to/job_config.yaml [-s]
```

| Argument       | Description                               |
| -------------- | ----------------------------------------- |
| `config_path`  | Path to the job YAML configuration file.  |
| `-s`, `--save` | Optional boolean arg to save the PNG file |

---

## Example

#### Display job in a new window
```bash
python tools/visualize_job.py examples/llama3/static/mesh.yaml
```

#### Display job in a new window and save the PNG file
```bash
python tools/visualize_job.py examples/llama3/static/mesh.yaml -s
```
