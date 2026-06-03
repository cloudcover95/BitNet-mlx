# path: BitNet-mlx/Makefile
.PHONY: install test lint clean build run

install:
	pip install --upgrade pip
	pip install -e .[dev]

lint:
	ruff check src/ tests/ run_offline_blackbox.py

 test:
	pytest tests/ -v --tb=short

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

build: clean
	python3 -m build

run:
	python3 run_offline_blackbox.py
