MLFLOW_PORT ?= 5000
MLFLOW_PID := .mlflow.pid
MLFLOW_LOG := mlflow.log

.PHONY: mlflow-start mlflow-stop image k8s-up k8s-down

mlflow-start:
	@if [ -f $(MLFLOW_PID) ] && kill -0 $$(cat $(MLFLOW_PID)) 2>/dev/null; then \
		echo "mlflow already running (pid $$(cat $(MLFLOW_PID)))"; \
	else \
		setsid uv run --project client_code/baseline mlflow server \
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

# --- container image ---------------------------------------------------------

SHA     := $(shell git rev-parse --short HEAD)
IMAGE   := inference-server
CLUSTER := dolores
BUNDLE  := data/iris-classifier.zip

image:
	docker build -t $(IMAGE):$(SHA) -t $(IMAGE):dev inference-server/
	@echo "built $(IMAGE):$(SHA)"

# --- local kubernetes --------------------------------------------------------
#
# The model is not in the image. It ships as a ConfigMap built straight from
# $(BUNDLE), so nothing has to regenerate a YAML file when the model changes.

k8s-up: image
	@kind get clusters 2>/dev/null | grep -qx $(CLUSTER) || kind create cluster --name $(CLUSTER)
	kind load docker-image $(IMAGE):$(SHA) --name $(CLUSTER)
	kubectl create configmap iris-model --from-file=$(BUNDLE) \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f k8s/inference-server.yaml
	kubectl set image deployment/inference-server api=$(IMAGE):$(SHA)
	kubectl rollout status deployment/inference-server --timeout=120s
	@echo
	@echo "  kubectl port-forward svc/inference-server 8000:80"
	@echo "  curl localhost:8000/model"

k8s-down:
	kind delete cluster --name $(CLUSTER)
