from src.utils.profiling import cpu_profile

def test_cpu_profile():
    with cpu_profile("test_section"):
        x = sum(range(1000))
    assert x == 499500