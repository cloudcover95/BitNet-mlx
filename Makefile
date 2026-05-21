.PHONY: install test lint eval swarm clean build

install:
	pip install --upgrade pip
	pip install -e .[dev]

test:
	pytest tests/ -v

lint:
	black src/ tests/

eval:
	bitnet-eval

swarm:
	bitnet-swarm --port 9000

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ logs/*.parquet
	find . -name __pycache__ -exec rm -rf {} +

build: clean
	python -m build
