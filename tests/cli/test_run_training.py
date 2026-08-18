import logging
from src.training.config import WEAK_MODE
from src.cli.run_training import main

import pytest

@pytest.mark.skipif(WEAK_MODE, reason="Weak mode: training skipped")
def test_run_training_smoke():
    main()