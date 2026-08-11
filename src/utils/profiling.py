import logging
import cProfile
import pstats
from contextlib import contextmanager
from torch.profiler import profile, ProfilerActivity

logger = logging.getLogger("app")

@contextmanager
def cpu_profile(section_name: str):
    """
    CPU profiling using cProfile.
    Saves results to logs/profile_cpu.txt.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    yield
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats("tottime")

    logger.info(f"CPU profile [{section_name}] completed")

    with open("logs/profile_cpu.txt", "w") as f:
        stats.stream = f
        stats.print_stats()

@contextmanager
def torch_cpu_profile(section_name: str):
    """
    PyTorch CPU profiling.
    """
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=False,
        profile_memory=False,
        with_stack=False
    ) as prof:
        yield

    logger.info(f"Torch CPU profile [{section_name}] completed")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))