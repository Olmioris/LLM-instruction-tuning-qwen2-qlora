train:
    python run_training.py

eval:
    python run_eval.py

gen:
    python run_generation.py

hydra:
    python hydra_main.py

test:
    pytest -q

logs:
    cat logs/app.log