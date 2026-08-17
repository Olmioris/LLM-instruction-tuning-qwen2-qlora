from src.utils.profiling import cpu_profile

def test_cpu_profile_runs():
    with cpu_profile("test_section") as profile:
        assert profile is not None