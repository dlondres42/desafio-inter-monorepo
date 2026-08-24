MLFLOW_PORT ?= 5000
MLFLOW_PID := .mlflow.pid
MLFLOW_LOG := mlflow.log

.PHONY: mlflow-start mlflow-stop

mlflow-start:
	@if [ -f $(MLFLOW_PID) ] && kill -0 $$(cat $(MLFLOW_PID)) 2>/dev/null; then \
		echo "mlflow already running (pid $$(cat $(MLFLOW_PID)))"; \
	else \
		setsid uv run --project client_code mlflow server \
			--backend-store-uri "sqlite:///$(CURDIR)/mlflow.db" \
			--artifacts-destination "$(CURDIR)/mlartifacts" \
			--host 127.0.0.1 --port $(MLFLOW_PORT) \
			>$(MLFLOW_LOG) 2>&1 < /dev/null & \
		echo $$! > $(MLFLOW_PID); \
		echo "mlflow starting on http://127.0.0.1:$(MLFLOW_PORT) (pid $$(cat $(MLFLOW_PID)))"; \
	fi

mlflow-stop:
	@if [ -f $(MLFLOW_PID) ]; then \
		kill -- -$$(cat $(MLFLOW_PID)) 2>/dev/null || kill $$(cat $(MLFLOW_PID)) 2>/dev/null || true; \
		rm -f $(MLFLOW_PID); \
		echo "mlflow stopped"; \
	else \
		echo "mlflow not running"; \
	fi
