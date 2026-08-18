import pytest
from src.cli.run_generation import main
from src.training.config import WEAK_MODE

@pytest.mark.skipif(WEAK_MODE, reason="Weak mode: generation skipped")
def test_run_generation_smoke():
    main()