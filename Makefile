.PHONY: help test build push

help:
	@echo "Usage: make [target] EXP=[experiment_name]"
	@echo ""
	@echo "Targets:"
	@echo "  test          Run pytest for the specified experiment."
	@echo "  build         Build the Docker image for the specified experiment."
	@echo "  push          Push the Docker image for the specified experiment."
	@echo ""
	@echo "Arguments:"
	@echo "  EXP           The name of the experiment directory (e.g., example.rl)."

test:
	@if [ -z "$(EXP)" ]; then \
		echo "ERROR: EXP argument is required."; \
		exit 1; \
	fi
	@echo "Running tests for experiment: $(EXP)..."
	python tests/run.py $(EXP)

build:
	@if [ -z "$(EXP)" ]; then \
		echo "ERROR: EXP argument is required."; \
		exit 1; \
	fi
	@echo "Building Docker image for experiment: $(EXP)..."
	docker build -t $(EXP):latest training/$(EXP)/

push:
	@if [ -z "$(EXP)" ]; then \
		echo "ERROR: EXP argument is required."; \
		exit 1; \
	fi
	@echo "Pushing Docker image for experiment: $(EXP)..."
	@# Replace 'youruser' with your Docker Hub username or registry
	docker tag $(EXP):latest $(REGISTRY_USER)/$(EXP):latest
	docker push $(REGISTRY_USER)/$(EXP):latest
