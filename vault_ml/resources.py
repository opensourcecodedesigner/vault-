import os
import threading
import time


def snapshot():
    import psutil
    import torch
    ram = psutil.virtual_memory()
    result = {"process_rss_bytes": psutil.Process(os.getpid()).memory_info().rss,
              "system_available_ram_bytes": ram.available, "system_total_ram_bytes": ram.total}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                      cuda_reserved_bytes=torch.cuda.memory_reserved(),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      cuda_free_bytes=free, cuda_total_bytes=total)
    return result


class ResourceMonitor:
    """Sample RSS/CPU while PyTorch separately tracks allocator peaks."""
    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.peak_rss = self.process.memory_info().rss
        self.cpu_start = sum(self.process.cpu_times()[:2])
        self.start = time.perf_counter()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(0.1):
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        self.result = {**snapshot(), "sampled_peak_rss_bytes": self.peak_rss,
                       "cpu_seconds": sum(self.process.cpu_times()[:2]) - self.cpu_start,
                       "wall_seconds": time.perf_counter() - self.start}
