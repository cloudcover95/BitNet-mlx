.PHONY: install test lint eval clean build

install:
	pip install --upgrade pip
	pip install -e .[dev]

test:
	pytest tests/ -v

lint:
	black src/ tests/

eval:
	bitnet-eval

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/
	find . -name __pycache__ -exec rm -rf {} +

build: clean
	python -m build
