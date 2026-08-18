import logging
from src.training.config import WEAK_MODE
from src.cli.run_generation import main

import pytest

@pytest.mark.skipif(WEAK_MODE, reason="Weak mode: generation skipped")
def test_run_generation_smoke():
    main()