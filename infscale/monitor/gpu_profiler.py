import threading
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional
from pynvml import (
    nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo, nvmlDeviceGetName,
    nvmlDeviceGetUtilizationRates, nvmlDeviceGetComputeRunningProcesses
)

class GPUProfiler:
    """GPU Profiler class for monitoring GPU utilization."""

    def __init__(self, interval_ms: int = 10):
        """Initialize GPU Profiler.

        Args:
            interval_ms: Sampling interval in milliseconds.
        """
        self.interval = interval_ms / 1000  # Convert interval to seconds.
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        # Create the gpu_log directory if it doesn't exist.
        self.log_dir = Path.cwd() / "gpu_log"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create a timestamped log file.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"gpu_utilization_{timestamp}.log"

        # Set up a dedicated logger that writes only to the file.
        self.logger = logging.getLogger(f"gpu_profiler_{timestamp}")
        self.logger.setLevel(logging.INFO)

        # Create and add a file handler.
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Prevent messages from propagating to the root logger, so nothing is printed.
        self.logger.propagate = False

    def start(self):
        """Start GPU profiling in a separate thread."""
        try:
            nvmlInit()
        except Exception as e:
            # If initialization fails, log the error to the file.
            self.logger.info(f"Failed to initialize NVML: {e}")
            return

        self.thread = threading.Thread(target=self._profile_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop GPU profiling."""
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join()

    def _profile_loop(self):
        """Main profiling loop."""
        while not self.stop_event.is_set():
            try:
                gpu_count = nvmlDeviceGetCount()
                readings = []
                for i in range(gpu_count):
                    handle = nvmlDeviceGetHandleByIndex(i)
                    util = nvmlDeviceGetUtilizationRates(handle).gpu
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    mem_percent = (mem_info.used / mem_info.total) * 100
                    readings.append(f"GPU{i}: Util={util}%, Mem={mem_percent:.1f}%")
                self.logger.info(" | ".join(readings))
            except Exception as e:
                self.logger.info(f"Error during GPU profiling: {e}")
                break

            time.sleep(self.interval) 