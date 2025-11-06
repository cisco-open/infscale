# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""visualize_tool.py."""


import argparse
import os

import matplotlib.pyplot as plt
import networkx as nx
import yaml

from infscale.configs.job import JobConfig


def get_job_data(path: str) -> tuple[JobConfig, str]:
    """Load job data composed by JobConfig and yaml file name."""
    with open(path) as f:
        data = yaml.safe_load(f)
        file_name = os.path.splitext(os.path.basename(path))[0]

    return JobConfig(**data), file_name


def build_graph(job: JobConfig) -> tuple[nx.DiGraph, dict[str, int]]:
    """Build graph using job configuration."""
    graph = nx.DiGraph()

    # Add all worker nodes
    worker_stage = {w.id: w.stage["start"] for w in job.workers}

    for wid in job.flow_graph:
        graph.add_node(wid, stage=worker_stage.get(wid, 0))

    # Add edges (peer -> wid) with labels from worlds
    for wid, worlds in job.flow_graph.items():
        for world in worlds:
            for peer in world.peers:
                label = f"{world.name}"
                if world.addr:
                    label += f"\n{world.addr}"
                if world.backend:
                    label += f"\n{world.backend}"
                graph.add_edge(peer, wid, label=label)

    return graph, worker_stage


def draw_graph(
    graph: nx.DiGraph,
    worker_stage: dict[str, int],
    file_name: str,
    output_path: str = "",
) -> None:
    """Draw graph where worker_stage maps node -> stage (start)."""
    # build positions (horizontal by stage, vertical stacked)
    stage_to_nodes = {}
    for node, stage in worker_stage.items():
        stage_to_nodes.setdefault(stage, []).append(node)

    sorted_stages = sorted(stage_to_nodes.keys())
    pos = {}
    x_spacing = 4
    y_spacing = 2

    for x, stage in enumerate(sorted_stages):
        nodes = sorted(stage_to_nodes[stage])  # sort for stable layout
        for y, node in enumerate(nodes):
            pos[node] = (x * x_spacing, -y * y_spacing)

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    nx.draw_networkx_nodes(
        graph, pos, node_size=2000, node_color="#5bf4a7", edgecolors="black", ax=ax
    )
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold", ax=ax)

    # get edge labels from graph attributes (we created them earlier as "label")
    edge_labels = nx.get_edge_attributes(graph, "label")

    # For each edge artist, get its path and compute the point at t along the curve.
    edge_artists = []
    for src, dst in graph.edges():
        # Determine arc direction and size
        src_stage = worker_stage.get(src, 0)
        dst_stage = worker_stage.get(dst, 0)
        _, y = pos[src]

        if worker_stage[dst] == -1 and y < 0:
            # edge connection to server on rows below the first
            rad = -abs(0.58)
        elif worker_stage[dst] == -1 and y == 0:
            # edge connection to server on first row
            rad = abs(0.2)
        else:
            # edge connection between sibling nodes
            rad = 0.05 if dst_stage >= src_stage else -0.05

        # draw edge
        artist = nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[(src, dst)],
            arrows=True,
            arrowstyle="-|>",
            arrowsize=20,
            width=1.5,
            connectionstyle=f"arc3,rad={rad}",
            min_source_margin=25,
            min_target_margin=25,
            ax=ax,
        )
        edge_artists.extend(artist)

        # label placement (same as before)
        label = edge_labels.get((src, dst))
        if not label:
            continue

        try:
            path = artist[0].get_path()
            verts = path.vertices
            if len(verts) >= 3:
                P0, P1, P2 = verts[0], verts[1], verts[2]
                t = 0.3
                one_minus_t = 1 - t
                x = (
                    (one_minus_t**2) * P0[0]
                    + 2 * one_minus_t * t * P1[0]
                    + (t**2) * P2[0]
                )
                y = (
                    (one_minus_t**2) * P0[1]
                    + 2 * one_minus_t * t * P1[1]
                    + (t**2) * P2[1]
                )
            else:
                raise ValueError
        except Exception:
            x1, y1 = pos[src]
            x2, y2 = pos[dst]
            t = 0.3
            x, y = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t

        ax.text(
            x,
            y,
            label,
            fontsize=8,
            ha="center",
            va="center",
            rotation=0,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9),
            zorder=10,
        )

    ax.set_axis_off()
    plt.tight_layout()
    if output_path:
        # save PNG in the same folder as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, f"{file_name}.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Graph saved at: {output_file}")

    print("Graph opened in a new window.")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize JobConfig flow graph")
    parser.add_argument("config_path", help="Path to job YAML config")
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save the graph as a PNG file in the same directory as the script instead of displaying it",
    )
    args = parser.parse_args()

    try:
        config, file_name = get_job_data(args.config_path)
    except FileNotFoundError as e:
        print(f"Error while loading file: {e}")
        return

    graph, worker_stage = build_graph(config)
    output_path = os.path.dirname(os.path.abspath(__file__)) if args.save else None
    try:
        draw_graph(graph, worker_stage, file_name, output_path)
    except nx.exception.NetworkXError as e:
        print(f"Error while drawing graph: {e}")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
