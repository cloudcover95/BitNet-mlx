.PHONY: install test audit clean build publish telemetry deploy train

install:
	pip install --upgrade pip
	pip install -e .[dev]

test:
	pytest tests/ -v --disable-warnings

audit:
	bitnet-audit --export-json

telemetry:
	python -c "from bitnet_mlx.telemetry import EdgePrometheusServer; EdgePrometheusServer.start()"

deploy:
	helm upgrade --install bitnet-edge deployment/helm/bitnet-mlx --namespace juniorcloud-system --create-namespace

train:
	bitnet-qat --model ./assets/ternary_phi3 --dataset ./data/finetune.jsonl --epochs 1

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +

build: clean
	python -m build

publish: build
	twine upload dist/*
