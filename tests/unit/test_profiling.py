import os
from src.utils.profiling import cpu_profile

def test_cpu_profile_creates_profile_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with cpu_profile("test_section"):
        _ = sum(range(1000))

    assert os.path.exists(tmp_path / "logs" / "profile_cpu.txt")