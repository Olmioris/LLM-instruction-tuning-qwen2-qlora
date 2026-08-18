import os
import logging
import cProfile
import pstats
from contextlib import contextmanager

logger = logging.getLogger("app")


@contextmanager
def cpu_profile(section_name: str):
    os.makedirs("logs", exist_ok=True)

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()

        stats = pstats.Stats(profiler).sort_stats("tottime")

        filename = os.path.join("logs", f"profile_cpu_{section_name}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            stats.stream = f
            stats.print_stats()

        logger.info(f"CPU profile [{section_name}] completed -> {filename}")