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

.PHONY: clean-logs

clean-logs:
    @echo "Cleaning logs..."
    @echo "" > logs/app.log
    @echo "" > logs/profile_cpu.txt
    @echo "Logs cleaned."