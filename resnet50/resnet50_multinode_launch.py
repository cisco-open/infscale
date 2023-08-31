import yaml, sys, time, json
import subprocess

def launch(world_size, split_size, partitions, shards, devices, pre_trained, logging):
    args = [
        json.dumps(partitions),
        json.dumps(shards),
        json.dumps(devices),
        "--micro_batch_size", str(split_size),
        "--pre_trained", str(int(pre_trained)),
        "--logging", str(int(pre_trained))
    ]
    host_ip = "10.20.1.50"
    host_port = "10819"
    node_list = ["10.20.1.15"]
    for i, node in enumerate(node_list):
        subprocess.Popen(["ssh", "changwu@" + node, "\"", "cd", "LLM-Inference;","torchrun", "--nnodes=2",f"--node-rank={i + 1}", "--nproc-per-node=4", "--rdzv-id=819", "--rdzv-backend=c10d", f"--rdzv-endpoint={host_ip}:{host_port}",  "resnet50/resnet50_multinode_inference.py", *args, "\""])
    subprocess.run(["torchrun", "--nnodes=2", "--node-rank=0", "--nproc-per-node=4", "--rdzv-id=819", "--rdzv-backend=c10d", f"--rdzv-endpoint={host_ip}:{host_port}", "resnet50/resnet50_multinode_inference.py", *args])
    # args = [
    #     "--micro_batch_size", str(split_size),
    #     "--partitions", "\"" + json.dumps(partitions).replace("\"", "\\\"") + "\"",
    #     "--shards", "\"" + json.dumps(shards).replace("\"", "\\\"") + "\"",
    #     "--devices", "\"" + json.dumps(devices).replace("\"", "\\\"") + "\"",
    #     "--pre_trained", str(int(pre_trained)),
    #     "--logging", str(int(pre_trained))
    # ]
    # args = " ".join(args)
    # subprocess.Popen(f"torchrun --standalone --nnodes=1 --nproc-per-node={world_size} resnet50/resnet50_multinode_inference.py {args}")

if __name__=="__main__":
    with open("resnet50_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
        file = open("./resnet50.log", "w")
        open("./resnet50.csv", "w")
        original_stdout = sys.stdout
        sys.stdout = file

        partitions = config["partitions"]
        devices = config["devices"]
        repeat_times = config["repeat_times"]
        placements = config["placements"]
        split_size = config["micro_batch_size"]
        pre_trained = config["pre_trained"] == "True" if "pre_trained" in config else False
        logging = config["logging"] == "True" if "logging" in config else False
        for shards in placements:
            world_size = len(shards) + 1
            print("Placement:", shards)
            for i in range(repeat_times):
                tik = time.time()
                launch(world_size, split_size, partitions, shards, devices, pre_trained, logging)
                tok = time.time()

                print(f"size of micro-batches = {split_size}, end-to-end execution time = {tok - tik} s")

        sys.stdout = original_stdout