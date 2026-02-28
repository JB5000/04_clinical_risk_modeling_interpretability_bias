.PHONY: help install install-dev test lint format clean run-api run-demo docker-build docker-up

help:
@echo "Clinical Risk Modeling - Available Commands"
@echo "==========================================="
@echo "install        - Install production dependencies"
@echo "install-dev    - Install development dependencies"
@echo "test           - Run tests with coverage"
@echo "lint           - Run linting checks"
@echo "format         - Format code with black and isort"
@echo "clean          - Remove build artifacts and cache"
@echo "run-api        - Start FastAPI server"
@echo "run-demo       - Run demo pipeline"
@echo "docker-build   - Build Docker image"
@echo "docker-up      - Start all services with docker-compose"
@echo "docker-down    - Stop all services"

install:
pip install -r requirements.txt

install-dev:
pip install -r requirements-dev.txt
pre-commit install

test:
pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
flake8 src tests --max-line-length=100
mypy src --ignore-missing-imports
pylint src

format:
black src tests examples
isort src tests examples

clean:
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type d -name "*.egg-info" -exec rm -rf {} +
rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

run-api:
uvicorn src.api.main:app --reload --port 8000

run-demo:
python examples/demo.py

docker-build:
docker build -t clinical-risk-api:latest .

docker-up:
docker-compose up -d

docker-down:
docker-compose down
