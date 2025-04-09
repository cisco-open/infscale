import subprocess


def run_tests():
    # start controller and agents
    _run("start_processes.yml")

    # start job
    _run("start_job.yml")

    # TODO: run cleanup processes after job is done

def _run(config_path: str) -> None:
    command = [
        "ansible-playbook",
        "-i", "inventory.ini",
        config_path,
    ]


    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )

    # re-route output to the terminal for visual feedback
    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        print(f"\n {config_path} failed with exit code {process.returncode}")
    else:
        print(f"\n {config_path} completed successfully.")

if __name__ == "__main__":
    run_tests()