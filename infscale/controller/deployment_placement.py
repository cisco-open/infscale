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

"""deployment_placement.py."""

import argparse
import json
import os
import time

import yaml


def dp_pipeline_packing(
    gpus_per_node, num_nodes, num_templates, throughput, need_dispatcher: bool = False
):
    """Exact DP with atomic tails (no template split).

    If ``need_dispatcher`` is True we reserve 1 GPU on the *last* node for a
    dispatcher process.  Concretely, we force the final solution to end in
    state ``(n = M-1, c = H-1)`` where ``M`` is the node budget and ``H`` is
    GPUs per node.  This guarantees exactly one free GPU on the last machine.
    """
    # Pre-compute full nodes and tails
    w = [i // gpus_per_node for i in range(num_templates + 1)]  # full nodes
    r = [i % gpus_per_node for i in range(num_templates + 1)]  # tail node # GPUs

    NEG = float("-inf")
    DP = [[NEG] * gpus_per_node for _ in range(num_nodes + 1)]
    prev = [[None] * gpus_per_node for _ in range(num_nodes + 1)]
    DP[0][0] = 0

    for n in range(num_nodes + 1):
        for c in range(gpus_per_node):
            if DP[n][c] == NEG:
                continue
            base = DP[n][c]
            space = gpus_per_node - c  # free GPUs in the open node
            for i in range(1, num_templates + 1):  # for each template label from 1 to I
                tail = r[i]
                add_n = w[i]

                if tail == 0:  # pure full-node template
                    carry = 0
                    c_new = c

                else:
                    if tail < space:  # fits, node still open
                        carry = 0
                        c_new = c + tail
                    elif tail == space:  # exact fit, node closes
                        carry = 1
                        c_new = 0
                    else:  # doesn't fit – close node, open new one
                        carry = 1
                        c_new = tail

                n_new = n + add_n + carry
                if n_new <= num_nodes:
                    val = base + throughput[i]
                    if val > DP[n_new][c_new]:
                        DP[n_new][c_new] = val
                        prev[n_new][c_new] = (n, c, i)

    # --------------------------------------------------------------
    #  Pick best state respecting dispatcher constraint if required
    # --------------------------------------------------------------
    if need_dispatcher:
        tgt_n = num_nodes - 1
        tgt_c = gpus_per_node - 1
        if 0 <= tgt_n <= num_nodes and DP[tgt_n][tgt_c] != NEG:
            best, bst = DP[tgt_n][tgt_c], (tgt_n, tgt_c)
        else:
            best, bst = NEG, None
    else:
        best, bst = NEG, None
        for n in range(num_nodes + 1):
            for c in range(gpus_per_node):
                used = n if c == 0 else n + 1
                if used <= num_nodes and DP[n][c] > best:
                    best, bst = DP[n][c], (n, c)

    # If no feasible solution found, return NEG and empty solution
    if bst is None:
        return best, []

    # back-trace
    sol, (n, c) = [], bst
    while n or c:
        prev_state = prev[n][c]
        assert prev_state is not None
        n, c, i = prev_state
        sol.append(i)
    sol.reverse()
    return best, sol


def assign_nodes(solution, gpus_per_node):
    """
    Convert a flat list of template sizes into a list of per-node assignments.
    Each node is represented as a dict { template_size: count }.
    """
    node_list = []
    current_used = 0  # GPUs used in the current open node
    current_node = None  # dict for the open node

    for i in solution:
        w_i = i // gpus_per_node
        r_i = i % gpus_per_node

        # 1) allocate all full nodes
        for _ in range(w_i):
            node_list.append({i: 1})

        # 2) allocate the remainder tail if any
        if r_i > 0:
            # "fits" in the current open node?
            if current_used + r_i < gpus_per_node:
                # start a new open node if needed
                if current_used == 0:
                    current_node = {}
                    node_list.append(current_node)
                # pack into it
                current_node[i] = current_node.get(i, 0) + 1
                current_used += r_i

            # "exact fit" closes the node
            elif current_used + r_i == gpus_per_node:
                if current_used == 0:
                    current_node = {}
                    node_list.append(current_node)
                current_node[i] = current_node.get(i, 0) + 1
                # close it
                current_used = 0
                current_node = None

            # "overflow" must open a fresh node
            else:  # current_used + r_i > gpus_per_node
                current_node = {i: 1}
                node_list.append(current_node)
                current_used = r_i

    return node_list


# ============================================================
#  Fault-tolerant variant (tolerate up to k node failures)
# ============================================================


def dp_pipeline_packing_fault_tolerant(
    gpus_per_node: int,
    num_nodes: int,
    num_templates: int,
    throughput: list,
    k: int,
    need_dispatcher: bool = False,
):
    """Dynamic programming that guarantees the resulting placement can
    tolerate up to *k* node failures by ensuring at least ``k+1`` disjoint
    deployment sets.

    The DP state is (n, c, d):
        n: number of *closed* nodes already fully utilised,
        c: GPUs used in the *current* open node (``0`` if none is open),
        d: number of *disjoint deployment sets* that have been placed so far
            **capped** at ``k+1`` (i.e. we store ``min(real_d, k+1)``).  The
            cap is safe because any state with more than ``k+1`` sets already
            satisfies the fault-tolerance requirement and is equivalent for
            optimisation purposes.
    The transition logic follows the derivation in the accompanying slides
    (see user prompt) and mirrors the original 2-D DP while extending it with
    the extra dimension ``d``.
    """

    # Pre-compute full-node (w) and tail-GPU (r) requirements for each template
    w = [i // gpus_per_node for i in range(num_templates + 1)]
    r = [i % gpus_per_node for i in range(num_templates + 1)]

    NEG = float("-inf")
    D_CAP = k + 1  # we only need to distinguish up to k+1 sets

    # DP[n][c][d] – maximum throughput for given state
    DP = [
        [[NEG] * (D_CAP + 1) for _ in range(gpus_per_node)]
        for _ in range(num_nodes + 1)
    ]
    prev = [
        [[None] * (D_CAP + 1) for _ in range(gpus_per_node)]
        for _ in range(num_nodes + 1)
    ]
    DP[0][0][0] = 0

    for n in range(num_nodes + 1):
        for c in range(gpus_per_node):
            for d in range(D_CAP + 1):
                if DP[n][c][d] == NEG:
                    continue

                base = DP[n][c][d]

                # iterate over all template options (unbounded knapsack style)
                for i in range(1, num_templates + 1):
                    w_i, r_i = w[i], r[i]

                    # ----------------------------------------------------
                    #  Case 1 – template shares the *current* open node
                    # ----------------------------------------------------
                    if c > 0 and r_i > 0 and c + r_i <= gpus_per_node:
                        # fits (possibly exactly) into existing open node;
                        # disjoint-set count unchanged (still the same set)
                        if c + r_i < gpus_per_node:
                            n_new, c_new, carry = n + w_i, c + r_i, 0
                        else:  # exact fit – node closes
                            n_new, c_new, carry = n + w_i + 1, 0, 1

                        if (
                            n_new <= num_nodes
                            and base + throughput[i] > DP[n_new][c_new][d]
                        ):
                            DP[n_new][c_new][d] = base + throughput[i]
                            prev[n_new][c_new][d] = (n, c, d, i)

                    # ----------------------------------------------------
                    #  Case 2 – template starts a *new* disjoint set
                    # ----------------------------------------------------
                    d_new = min(d + 1, D_CAP)

                    # (a) current node is closed (c == 0)
                    if c == 0:
                        n_new = n + w_i
                        c_new = r_i  # may be 0 if r_i == 0
                        if (
                            n_new <= num_nodes
                            and base + throughput[i] > DP[n_new][c_new][d_new]
                        ):
                            DP[n_new][c_new][d_new] = base + throughput[i]
                            prev[n_new][c_new][d_new] = (n, c, d, i)

                    # (b) template has no tail GPUs (r_i == 0)
                    if r_i == 0:
                        n_new = n + w_i
                        c_new = c
                        if (
                            n_new <= num_nodes
                            and base + throughput[i] > DP[n_new][c_new][d_new]
                        ):
                            DP[n_new][c_new][d_new] = base + throughput[i]
                            prev[n_new][c_new][d_new] = (n, c, d, i)

                    # (c) tail exists but does *not* fit in current open node
                    if c > 0 and r_i > 0 and c + r_i > gpus_per_node:
                        n_new = n + w_i + 1  # close current node, open new
                        c_new = r_i  # new open node partially filled
                        if (
                            n_new <= num_nodes
                            and base + throughput[i] > DP[n_new][c_new][d_new]
                        ):
                            DP[n_new][c_new][d_new] = base + throughput[i]
                            prev[n_new][c_new][d_new] = (n, c, d, i)

    # ------------------------------------------------------------------
    #  Pick best state, possibly reserving 1 GPU for dispatcher
    # ------------------------------------------------------------------
    if need_dispatcher:
        tgt_n, tgt_c = num_nodes - 1, gpus_per_node - 1
        if 0 <= tgt_n <= num_nodes and DP[tgt_n][tgt_c][D_CAP] != NEG:
            best, bst = DP[tgt_n][tgt_c][D_CAP], (tgt_n, tgt_c, D_CAP)
        else:
            best, bst = NEG, None
    else:
        best, bst = NEG, None
        for n in range(num_nodes + 1):
            for c in range(gpus_per_node):
                used = n if c == 0 else n + 1
                if used > num_nodes:
                    continue
                if DP[n][c][D_CAP] > best:
                    best, bst = DP[n][c][D_CAP], (n, c, D_CAP)

    # If no feasible solution found, return NEG and empty solution
    if bst is None:
        return best, []

    # --------------------------------------
    #  Back-track to recover chosen templates
    # --------------------------------------
    sol, (n, c, d) = [], bst
    while not (n == 0 and c == 0 and d == 0):
        prev_state = prev[n][c][d]
        if prev_state is None:
            # Should not happen; break to avoid infinite loop
            break
        n_prev, c_prev, d_prev, tpl = prev_state
        sol.append(tpl)
        n, c, d = n_prev, c_prev, d_prev
    sol.reverse()
    return best, sol


def assign_nodes_with_deployments(solution, gpus_per_node):
    """Assign deployments to nodes while emitting both node-centric and
    deployment-centric views.

    Returns (nodes_dict, deployments_dict) where:
        nodes_dict        – { node_id: [ {deploy_id, gpus} ] }
        deployments_dict  – { deploy_id: {template_size, node_segments} }
    """

    nodes: list[list[dict]] = []  # index => list of fragments
    deployments = {}  # deploy_id => info
    counters = {}  # template_size => next index

    current_node_id = None  # open node index or None
    current_used_gpus = 0  # GPUs used in open node

    for t in solution:
        w_i = t // gpus_per_node
        r_i = t % gpus_per_node

        # allocate deployment id
        idx = counters.get(t, 0)
        deploy_id = f"{t}-{idx}"
        counters[t] = idx + 1

        deployments[deploy_id] = {"template_size": t, "node_segments": []}

        # 1) full nodes
        for _ in range(w_i):
            node_id = len(nodes)
            nodes.append([])

            nodes[node_id].append({"deploy_id": deploy_id, "gpus": gpus_per_node})
            deployments[deploy_id]["node_segments"].append(
                {"node_id": node_id, "gpus": gpus_per_node}
            )

        # 2) remainder/tail GPUs
        if r_i > 0:
            # ensure there is an open node
            if current_node_id is None:
                current_node_id = len(nodes)
                nodes.append([])
                current_used_gpus = 0

            # if overflow, close current node and open a new one
            if current_used_gpus + r_i > gpus_per_node:
                # close current
                current_node_id = len(nodes)
                nodes.append([])
                current_used_gpus = 0

            # if exact fit after placing; we may still need a new node next time
            node_id = current_node_id
            nodes[node_id].append({"deploy_id": deploy_id, "gpus": r_i})
            deployments[deploy_id]["node_segments"].append(
                {"node_id": node_id, "gpus": r_i}
            )
            current_used_gpus += r_i

            # if node now full, close it
            if current_used_gpus == gpus_per_node:
                current_node_id = None
                current_used_gpus = 0

    # convert nodes list into a dict indexed by string for JSON friendliness
    nodes_dict = {str(i): frags for i, frags in enumerate(nodes)}
    return nodes_dict, deployments


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="solution/llama")
    parser.add_argument(
        "-k",
        "--fault_tolerance",
        type=int,
        default=0,
        help="Number of node failures (k) to tolerate. 0 = no fault tolerance (default).",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="placement.json",
        help="Path of the JSON file to write the placement result.",
    )
    parser.add_argument(
        "--dispatcher",
        action="store_true",
        help="Reserve 1 GPU on the last node for a dispatcher service.",
    )
    parser.add_argument(
        "--gpu_per_node", type=int, default=4, help="Number of GPUs per node."
    )
    parser.add_argument(
        "--num_nodes", type=int, default=10, help="Total number of nodes."
    )
    args = parser.parse_args()

    # Load real throughput data
    with open(os.path.join(args.dir, "solutions_index.yaml"), "r") as file:
        solutions_data = yaml.safe_load(file)

    gpu_throughputs = {
        int(k): v[0]["throughput"] for k, v in solutions_data.items() if v
    }
    num_templates = max(gpu_throughputs.keys())
    throughput = [0] * (num_templates + 1)
    for i, tput in gpu_throughputs.items():
        throughput[i] = tput

    gpus_per_node = args.gpu_per_node
    num_nodes = args.num_nodes

    print("Throughputs (GPUs: throughput):", gpu_throughputs)

    start = time.time()

    if args.fault_tolerance > 0:
        best_profit, solution = dp_pipeline_packing_fault_tolerant(
            gpus_per_node,
            num_nodes,
            num_templates,
            throughput,
            args.fault_tolerance,
            need_dispatcher=args.dispatcher,
        )
    else:
        best_profit, solution = dp_pipeline_packing(
            gpus_per_node,
            num_nodes,
            num_templates,
            throughput,
            need_dispatcher=args.dispatcher,
        )

    end = time.time()

    if args.fault_tolerance > 0:
        print(
            f"\nBest total throughput ≤{num_nodes} nodes (tolerates up to {args.fault_tolerance} node failures): "
            f"{best_profit} (computed in {end-start:.6f}s)"
        )
    else:
        print(
            f"\nBest total throughput ≤{num_nodes} nodes: {best_profit} (computed in {end-start:.6f}s)"
        )

    print(f"Solution: {solution}")

    # Build per-node assignment
    nodes = assign_nodes(solution, gpus_per_node)

    # Print it
    for idx, node in enumerate(nodes):
        items = ", ".join(f"template {tpl} x {cnt}" for tpl, cnt in node.items())
        print(f"Node {idx}: {items}")

    # Build rich JSON views
    nodes_dict, deployments_dict = assign_nodes_with_deployments(
        solution, gpus_per_node
    )

    # Inject dispatcher fragment if needed
    if args.dispatcher:
        # Identify node with exactly one free GPU
        disp_node_id = None
        for nid, frags in nodes_dict.items():
            used = sum(frag["gpus"] for frag in frags)
            if used == gpus_per_node - 1:
                disp_node_id = nid
                break

        # Fallback: put on the last node
        if disp_node_id is None:
            disp_node_id = str(len(nodes_dict) - 1)

        disp_fragment = {"deploy_id": "dispatcher", "gpus": 1}
        nodes_dict.setdefault(disp_node_id, []).append(disp_fragment)
        deployments_dict["dispatcher"] = {
            "template_size": 1,
            "node_segments": [{"node_id": int(disp_node_id), "gpus": 1}],
        }

    # gather partition info for templates actually deployed
    deployed_sizes = {
        int(s.split("-")[0]) for s in deployments_dict.keys() if s != "dispatcher"
    }
    template_solutions = {}
    for sz in deployed_sizes:
        entry = (
            solutions_data.get(sz)
            if sz in solutions_data
            else solutions_data.get(str(sz))
        )
        if isinstance(entry, list) and entry:
            tpl_filename = (
                entry[0].get("template_name") if isinstance(entry[0], dict) else None
            )
            if tpl_filename:
                template_solutions[str(sz)] = os.path.join(args.dir, tpl_filename)

    # Now construct JSON blob and write to disk
    result_json = {
        "meta": {
            "gpus_per_node": gpus_per_node,
            "max_nodes": num_nodes,
            "fault_tolerance_k": args.fault_tolerance,
            "total_throughput": best_profit,
            "runtime_sec": round(end - start, 6),
        },
        "deployments": deployments_dict,
        "nodes": nodes_dict,
        "template_solutions": template_solutions,
    }

    with open(args.out, "w") as fp:
        json.dump(result_json, fp, indent=2)
    print(f"\nPlacement written to {args.out}")

    if args.dispatcher:
        print(f"Dispatcher server placed on node {disp_node_id} (1 GPU reserved).")
