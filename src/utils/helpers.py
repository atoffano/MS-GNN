"""Utility functions for timing and memory tracking in Python applications."""

import time
import os
import psutil
import torch
import logging
from functools import wraps
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def timeit(func):
    """Decorator to measure and print function execution time.

    Args:
        func: Function to be timed

    Returns:
        Wrapped function that prints execution time when called
    """

    def wrapper(*args, **kwargs):
        """Wrapper function that times the decorated function."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds"
        )
        return result

    return wrapper


class MemoryTracker:
    """Track memory usage across parent and worker processes."""

    def __init__(self, log_prefix: str = ""):
        """Initialize memory tracker.

        Args:
            log_prefix: Prefix for log messages to identify tracking context
        """
        self.log_prefix = log_prefix
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = None

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information.

        Returns:
            Dictionary with memory metrics in MB
        """
        mem_info = self.process.memory_info()
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": mem_info.vms / 1024 / 1024,  # Virtual Memory Size
            "shared_mb": getattr(mem_info, "shared", 0) / 1024 / 1024,  # Shared memory
            "pid": os.getpid(),
            "ppid": os.getppid(),
        }

    def set_baseline(self):
        """Set baseline memory usage for comparison."""
        self.baseline_memory = self.get_memory_info()
        logger.info(
            f"{self.log_prefix}[PID {self.baseline_memory['pid']}] "
            f"Baseline memory: RSS={self.baseline_memory['rss_mb']:.1f} MB, "
            f"Shared={self.baseline_memory['shared_mb']:.1f} MB"
        )

    def log_memory(self, checkpoint_name: str = ""):
        """Log current memory usage and delta from baseline.

        Args:
            checkpoint_name: Name/description of this checkpoint
        """
        current = self.get_memory_info()

        log_msg = (
            f"{self.log_prefix}[PID {current['pid']}] {checkpoint_name}: "
            f"RSS={current['rss_mb']:.1f} MB, "
            f"Shared={current['shared_mb']:.1f} MB"
        )

        if self.baseline_memory:
            delta_rss = current["rss_mb"] - self.baseline_memory["rss_mb"]
            delta_shared = current["shared_mb"] - self.baseline_memory["shared_mb"]
            log_msg += (
                f", Delta RSS={delta_rss:+.1f} MB, Delta Shared={delta_shared:+.1f} MB"
            )

        logger.info(log_msg)
        return current

    def log_cuda_memory(self, checkpoint_name: str = ""):
        """Log CUDA memory usage if available.

        Args:
            checkpoint_name: Name/description of this checkpoint
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            logger.info(
                f"{self.log_prefix}[PID {os.getpid()}] {checkpoint_name}: "
                f"CUDA Allocated={allocated:.1f} MB, Reserved={reserved:.1f} MB"
            )


def track_memory(checkpoint_name: str):
    """Decorator to track memory usage around a function call.

    Args:
        checkpoint_name: Name to identify this checkpoint
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracker = MemoryTracker(log_prefix=f"[{func.__name__}] ")
            tracker.log_memory(f"Before {checkpoint_name}")
            result = func(*args, **kwargs)
            tracker.log_memory(f"After {checkpoint_name}")
            return result

        return wrapper

    return decorator


class WorkerMemoryMonitor:
    """Monitor memory usage in DataLoader workers."""

    def __init__(self):
        """Initialize worker memory monitor."""
        self.worker_id = None
        self.tracker = None
        self.checkpoints = []

    def init_worker(self, worker_id: int):
        """Initialize monitoring for a specific worker.

        Args:
            worker_id: DataLoader worker ID
        """
        self.worker_id = worker_id
        self.tracker = MemoryTracker(log_prefix=f"[Worker {worker_id}] ")
        self.tracker.set_baseline()
        logger.info(f"[Worker {worker_id}] Initialized memory monitoring")

    def checkpoint(self, name: str):
        """Record a memory checkpoint.

        Args:
            name: Checkpoint name/description
        """
        if self.tracker:
            mem_info = self.tracker.log_memory(name)
            self.checkpoints.append({"name": name, "memory": mem_info})

    def log_summary(self):
        """Log summary of all checkpoints."""
        if not self.checkpoints:
            return

        logger.info(f"[Worker {self.worker_id}] Memory Usage Summary:")
        for i, cp in enumerate(self.checkpoints):
            mem = cp["memory"]
            logger.info(
                f"  {i+1}. {cp['name']}: RSS={mem['rss_mb']:.1f} MB, "
                f"Shared={mem['shared_mb']:.1f} MB"
            )


# Global worker monitor instance
_worker_monitor = WorkerMemoryMonitor()


def worker_init_fn(worker_id: int):
    """Initialization function for DataLoader workers with memory tracking.

    Args:
        worker_id: DataLoader worker ID
    """
    _worker_monitor.init_worker(worker_id)


def log_worker_checkpoint(name: str):
    """Log a memory checkpoint in worker.

    Args:
        name: Checkpoint name
    """
    _worker_monitor.checkpoint(name)


def get_process_tree_memory() -> Dict[int, Dict[str, Any]]:
    """Get memory usage for entire process tree.

    Returns:
        Dictionary mapping PID to memory info
    """
    current_process = psutil.Process(os.getpid())
    process_tree = {}

    # Parent process
    mem_info = current_process.memory_info()
    process_tree[current_process.pid] = {
        "role": "parent",
        "rss_mb": mem_info.rss / 1024 / 1024,
        "vms_mb": mem_info.vms / 1024 / 1024,
        "shared_mb": getattr(mem_info, "shared", 0) / 1024 / 1024,
    }

    # Child processes (workers)
    for child in current_process.children(recursive=True):
        try:
            mem_info = child.memory_info()
            process_tree[child.pid] = {
                "role": "worker",
                "rss_mb": mem_info.rss / 1024 / 1024,
                "vms_mb": mem_info.vms / 1024 / 1024,
                "shared_mb": getattr(mem_info, "shared", 0) / 1024 / 1024,
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return process_tree


def log_process_tree_memory():
    """Log memory usage for entire process tree."""
    tree = get_process_tree_memory()

    total_rss = sum(info["rss_mb"] for info in tree.values())
    total_shared = sum(info["shared_mb"] for info in tree.values())

    logger.info("=" * 60)
    logger.info("Process Tree Memory Usage:")
    logger.info("-" * 60)

    for pid, info in tree.items():
        logger.info(
            f"  PID {pid} ({info['role']}): "
            f"RSS={info['rss_mb']:.1f} MB, "
            f"Shared={info['shared_mb']:.1f} MB"
        )

    logger.info("-" * 60)
    logger.info(f"Total RSS across all processes: {total_rss:.1f} MB")
    logger.info(f"Total Shared memory: {total_shared:.1f} MB")
    logger.info("=" * 60)
