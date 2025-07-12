.PHONY: install dev test lint format typecheck clean

# Variables
PYTHON = python
PIP = uv pip install --upgrade
PYTEST = pytest -v --cov=embedding_api --cov-report=term-missing

# Installation
install:
	@echo "Installing dependencies..."
	$(PIP) -r requirements.txt

install-dev: install
	@echo "Installing development dependencies..."
	$(PIP) -r requirements-dev.txt

dev: install-dev
	@echo "Starting development server..."
	uvicorn embedding_api.main:app --reload --host 0.0.0.0 --port 6008

# Testing
test:
	@echo "Running tests..."
	$(PYTEST) tests/

test-cov:
	@echo "Running tests with coverage..."
	$(PYTEST) --cov-report=html

# Code Quality
lint:
	@echo "Running linters..."
	ruff check .
	black --check .
	isort --check-only .

format:
	@echo "Formatting code..."
	black .
	isort .

# Type checking
typecheck:
	@echo "Running type checking..."
	mypy .

# Cleanup
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	rm -rf .coverage htmlcov/ build/ dist/ *.egg-info/

# Help
default: help
help:
	@echo "Available commands:"
	@echo "  install     - Install production dependencies"
	@echo "  install-dev - Install development dependencies"
	@echo "  dev         - Run development server"
	@echo "  test        - Run tests"
	@echo "  test-cov    - Run tests with coverage report"
	@echo "  lint        - Run linters"
	@echo "  format      - Format code"
	@echo "  typecheck   - Run type checking"
	@echo "  clean       - Clean up build and cache files"
