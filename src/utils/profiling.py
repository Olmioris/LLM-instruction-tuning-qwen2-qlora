import os
import logging
import cProfile
import pstats
from contextlib import contextmanager

@contextmanager
def cpu_profile(section_name: str):
    os.makedirs("logs", exist_ok=True)
    profiler = cProfile.Profile()
    profiler.enable()
    yield
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats("tottime")
    logger.info(f"CPU profile [{section_name}] completed")

    with open("logs/profile_cpu.txt", "w") as f:
        stats.stream = f
        stats.print_stats()