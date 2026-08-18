import logging
from src.training.config import WEAK_MODE
from src.cli.run_eval import main

import pytest

@pytest.mark.skipif(WEAK_MODE, reason="Weak mode: eval skipped")
def test_run_eval_smoke():
    main()