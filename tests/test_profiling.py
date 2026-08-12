from src.utils.profiling import cpu_profile

def test_cpu_profile_runs():
    profile = cpu_profile()
    assert isinstance(profile, dict)