include .env
export

.PHONY: help test build push eval

help:
	@echo "Usage: make [target] EXP=[experiment_name] or EVAL=[eval_name]"
	@echo ""
	@echo "Targets:"
	@echo "  test          Run pytest for the specified experiment."
	@echo "  build         Build the Docker image for the specified experiment."
	@echo "  push          Push the Docker image for the specified experiment."
	@echo "  eval          Run the specified evaluation."
	@echo ""
	@echo "Arguments:"
	@echo "  EXP           The name of the experiment directory (e.g., example.rl)."
	@echo "  EVAL          The name of the eval directory (e.g., example.eval)."
	@echo "  CONFIG        Optional path to override config file for eval."

test:
	@if [ -z "$(EXP)" ]; then \
		echo "ERROR: EXP argument is required."; \
		exit 1; \
	fi
	@echo "Running tests for experiment: $(EXP)..."
	python tests/run.py $(EXP)

eval:
	@if [ -z "$(EVAL)" ]; then \
		echo "ERROR: EVAL argument is required."; \
		exit 1; \
	fi
	@echo "Running evaluation: $(EVAL)..."
	@if [ -n "$(CONFIG)" ]; then \
		python evals/runner.py $(EVAL) --config $(CONFIG); \
	else \
		python evals/runner.py $(EVAL); \
	fi
