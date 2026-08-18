import pytest
from src.cli.run_training import main
from src.training.config import WEAK_MODE

@pytest.mark.skipif(WEAK_MODE, reason="Weak mode: training skipped")
def test_run_training_smoke():
    main()