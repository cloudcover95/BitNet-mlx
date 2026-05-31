# .github/workflows/ci.yml
name: Omni-Sovereign Integration

on: [push, pull_request]

jobs:
  validate:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev]
      - name: Code Integrity (Ruff & Mypy)
        run: |
          ruff check src/
          mypy src/
      - name: TDA Spectral Matrix Tests
        run: pytest tests/ -v