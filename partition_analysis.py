import json
import matplotlib.pyplot as plt
import math
import random
import time
import statistics

def get_leaf_modules(profiling_res: dict, leaf_module_name_list = None) -> list:
    def walk_through_model(model_profile: dict, path: list, leaves: dict()):
        if "children" not in model_profile or model_profile["name"] in leaf_module_name_list:
            # find a leaf node
            leaves[".".join(path)] = model_profile
        else:
            children = model_profile["children"]
            for k, v in children.items():
                walk_through_model(v, path + [k], leaves)

    leaf_layers = dict()
    walk_through_model(profiling_res["Detailed Profile per GPU"]["root"], ["root"], leaf_layers)
    return leaf_layers

def time_str_to_float(s: str):
    digits = s.split()[0]
    metrics = s.split()[1]
    val = float(digits)
    if metrics == "ms":
        return val * 1e-3
    elif metrics == "us":
        return val * 1e-6
    elif metrics == "ns":
        return val * 1e-9
    else:
        assert 0

def find_min_std_dev_partition(leaf_layers: dict, n: int):
    latency_arr = [time_str_to_float((v["extra"]["fwd latency"])) for k, v in leaf_layers.items()]
    m = len(latency_arr)
    E_x = sum(latency_arr) / m # average latency of partitions is the same as the average latency of all elements
    
    # use dynamic programming to find the partition strategy that minimize the standard deviation
    # specifically, we want to minimize the sum of x^2 (x is the sum of latencies in a partition) here
    f = [[0] * n] * m
    for i in range(m):
        f[i][0] = sum([x * x for x in latency_arr[0:i]])

    for j in range(1, n):
        for i in range(j, m):
            f[i][j] = 1e9
            for k in range(j - 1, i):
                candidate = (f[k][j - 1] + sum([x * x for x in latency_arr[k+1:i+1]]))
                if candidate < f[i][j]:
                    f[i][j] = candidate
    
    return math.sqrt((f[m - 1][n - 1] / m) - pow(E_x, 2))

def plot_min_std_dev_curve(leaf_layers, max_num_partition):
    std_dev = []
    for i in range(max_num_partition):
        min_std_dev = find_min_std_dev_partition(leaf_layers, i + 1)
        std_dev.append(min_std_dev)

    print(std_dev)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1, max_num_partition + 1), std_dev)

def plot_random_partition_std_dev_curve(leaf_layers, num_partition, num_rounds = 10):
    latency_arr = [time_str_to_float((v["extra"]["fwd latency"])) for k, v in leaf_layers.items()]
    m = len(latency_arr)
    n = num_partition

    std_dev = []
    for j in range(num_rounds):
        print("{}-th round".format(j))
        random.seed(time.time())
        splits = {0, m}
        while len(splits) < n + 1:
            k = random.randint(1, m - 1)
            if k not in splits:
                splits.add(k)

        splits = sorted(splits)
        print("splits:", splits)
        x = []
        for i in range(len(splits) - 1):
            x.append(sum(latency_arr[splits[i]:splits[i+1]]))

        print("x:", x)
        temp = statistics.stdev(x)
        print("std_dev:", temp)
        std_dev.append(temp)

    print(std_dev)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1, num_rounds + 1), std_dev)

if __name__ == "__main__":
    profiling_file = "llama13b_profile.json"
    with open(profiling_file) as pf:
        profiling_res = json.load(pf)
        max_num_partition = 6
        leaf_layers = get_leaf_modules(profiling_res, leaf_module_name_list=["LlamaDecoderLayer"])

        print(leaf_layers.keys())
        # plot_min_std_dev_curve(leaf_layers, max_num_partition)
        plot_random_partition_std_dev_curve(leaf_layers, max_num_partition, 100)