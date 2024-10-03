"""
Copyright (c) 2023 SymbioticLab, The University of Michigan

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# This file was modified from
# https://github.com/SymbioticLab/Oobleck/blob/3b7a0c2f19bff0991e623ffbeb8a5b365853bf3a/oobleck/module/sharding.py


from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Tuple

import torch.fx
from infscale.module.model_metadata import BaseModelMetaData
from torch.fx.node import Node
from transformers.utils.fx import symbolic_trace


class Sharder:
    """A wrapper class that shards a model."""

    @classmethod
    def shard(cls, mmd: BaseModelMetaData) -> List[torch.fx.GraphModule]:
        """Return a list of layer objects that can be sharded."""
        split_points = mmd.get_split_points()
        concrete_args = mmd.trace_inputs

        assert (
            split_points
        ), f"Empty split points. Check model type {mmd.config.model_type} is supported."

        model = mmd.get_model()

        return shard_model(model, concrete_args, split_points)


def _split_nodes(
    traced: torch.fx.GraphModule, split_points: List[str]
) -> Tuple[Dict[str, int], Dict[int, List[str]], Dict[str, int]]:
    """Analyze the given traced module and split it to subgraphs.

    While partitioning, it also finds additioanl required inputs and outputs
    so that they are added.

    Args:
        traced (torch.fx.GraphModule): A traced graph module to be split.
    """
    node_name_to_shard_id: Dict[str, int] = {}
    shard_id_to_node: Dict[int, List[Node]] = defaultdict(list)
    shard_id = 0

    nodes_so_far: List[str] = []
    extra_output: Dict[int, List[str]] = {}

    for node in traced.graph.nodes:
        if node.op == "placeholder":
            node_name_to_shard_id[node.name] = shard_id
            nodes_so_far.append(node.name)
            shard_id_to_node[shard_id].append(node)
        elif node.op in [
            "get_attr",
            "call_function",
            "call_method",
            "call_module",
        ]:
            node_name_to_shard_id[node.name] = shard_id
            nodes_so_far.append(node.name)
            shard_id_to_node[shard_id].append(node)

            point = next(
                filter(lambda p: node.next.name.startswith(p), split_points), None
            )
            if point:
                # Record outputs that should be used later, so that it can be added
                # in return of this shard
                outputs = []
                nodes = list(chain(*shard_id_to_node.values()))
                for node in nodes:
                    for user in node.users.keys():
                        if user.name not in node_name_to_shard_id:
                            outputs.append(node.name)

                extra_output[shard_id] = list(dict.fromkeys(outputs).keys())

                # If the current node is in the next shard, we increase shard count.
                shard_id += 1
                split_points.remove(point)

        elif node.op == "output":
            break

    assert len(split_points) == 0, "Sharding is not complete."

    return node_name_to_shard_id, extra_output


def shard_model(
    model: torch.nn.Module, concrete_args: List[str], split_points: List[str]
) -> List[torch.fx.GraphModule]:
    """Use torch.fx to do symbolic trace on the given model, and shard it to several subgraphs based on the given split_points.

    Code reference:
    1. https://github.com/HPDL-Group/Merak/blob/e8a2a779fea878be9b778f8a808a192364766f36/Merak/autoshard/graph_shard.py
    2. https://github.com/facebookresearch/fairscale/blob/5b38de380e4407c2ef02f357ebc640f53470ea24/fairscale/experimental/nn/auto_shard.py

    Args:
        model (torch.nn.Module): The model to be sharded.
        concrete_args (List[str]): Arguments that are used for symbolic_trace.
            This will be the list of inputs of the generated :class:`torch.fx.GraphModule`.

        split_points (List[str]): Module names that are split.

    Returns:
        List[torch.fx.GraphModule]: The list of sharded :class:`torch.fx.GraphModule`s.
    """
    module_list: List[torch.fx.GraphModule] = []

    traced = symbolic_trace(model, input_names=concrete_args)
    split_points = [p.replace(".", "_") for p in split_points]

    node_name_to_shard_id, extra_outputs = _split_nodes(traced, split_points)

    prev_shard_id = 1000
    prev_node: Optional[Node] = None

    env: Dict[str, Node] = {}
    prev_node: Optional[Node] = None

    new_graph = torch.fx.Graph()
    # Iterate all nodes
    for node in traced.graph.nodes:
        if node.name in node_name_to_shard_id:
            current_shard_id = node_name_to_shard_id[node.name]
            if prev_shard_id < current_shard_id:
                assert prev_node, "prev_node cannot be None"

                # If the current node is in the next shard, we insert an output node.
                # A new graph is created an a placeholder is added for the next shard.

                with new_graph.inserting_after(prev_node):
                    if prev_shard_id in extra_outputs:
                        outputs = extra_outputs[prev_shard_id]
                        outputs = {i: env[i] for i in outputs}
                        new_graph.output(outputs)
                    else:
                        new_graph.output({prev_node.name: env[prev_node.name]})

                new_graph.lint()
                module_list.append(torch.fx.GraphModule(model, new_graph))

                # Create a new graph
                new_graph = torch.fx.Graph()
                for _, output in outputs.items():
                    # Add all nodes in return of the previous graph to its input
                    node_name = env[output.name].name
                    pl_node = new_graph.create_node("placeholder", node_name)
                    env[node_name] = pl_node

        # Cut is done. Add all nodes into the current graph (except for labels placeholder).
        if node.op in [
            "placeholder",
            "get_attr",
            "call_function",
            "call_method",
            "call_module",
        ]:
            # Copy the nodes from the existing graph to the new graph.
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
        elif node.op == "output":
            # If this is the last node, we should add an output node and add the last graph to the list.
            assert prev_node, "prev_node cannot be None"
            with new_graph.inserting_after(prev_node):
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
            new_graph.lint()
            module_list.append(torch.fx.GraphModule(model, new_graph))
            break

        prev_node = new_node
        prev_shard_id = node_name_to_shard_id[node.name]

    return module_list
