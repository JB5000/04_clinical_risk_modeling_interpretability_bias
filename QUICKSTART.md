# Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/JB5000/04_clinical_risk_modeling_interpretability_bias.git
cd 04_clinical_risk_modeling_interpretability_bias

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
make install  # or: pip install -r requirements.txt

# For development
make install-dev  # or: pip install -r requirements-dev.txt
```

## Quick Commands

### Using Makefile

```bash
make help              # Show all available commands
make install           # Install dependencies
make test              # Run tests
make lint              # Check code quality
make format            # Format code
make run-api           # Start API server
make run-demo          # Run demo
make docker-up         # Start all services
```

### Generate Synthetic Data

```bash
python scripts/generate_synthetic_data.py --samples 10000 --output data/train.csv
```

### Train Model

```bash
# Using the training script
python train.py --data data/train.csv --config configs/model_config.yaml

# Or with custom experiment name
python train.py --data data/train.csv --experiment my_experiment
```

### Evaluate Model

```bash
python evaluate.py --model models/xgboost_*.joblib --data data/test.csv --output results/
```

### Run Demo

```bash
python examples/demo.py
```

### Start API Server

```bash
# Development mode
uvicorn src.api.main:app --reload --port 8000

# Or using make
make run-api

# Access API docs at: http://localhost:8000/docs
```

## Docker Usage

```bash
# Build image
make docker-build

# Start all services (API, PostgreSQL, Redis, MLflow)
make docker-up

# Stop services
make docker-down

# View logs
docker-compose logs -f api
```

## API Examples

### Health Check

```bash
curl http://localhost:8000/
```

### Predict Risk

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P000123",
    "age": 65,
    "gender": "M",
    "race": "White",
    "bmi": 28.5,
    "systolic_bp": 145,
    "diastolic_bp": 90,
    "glucose": 120,
    "cholesterol": 220,
    "smoking_status": "Former",
    "diabetes": true,
    "hypertension": true,
    "heart_disease": false,
    "prior_stroke": false,
    "visit_timestamp": "2024-12-04T10:30:00"
  }'
```

### Model Info

```bash
curl http://localhost:8000/model/info
```

### Fairness Audit

```bash
curl http://localhost:8000/fairness/audit
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
make test

# Specific test file
pytest tests/test_pipeline.py -v
```

## Development Workflow

1. **Generate Data**
   ```bash
   python scripts/generate_synthetic_data.py --samples 10000
   ```

2. **Train Model**
   ```bash
   python train.py --data data/synthetic_clinical_data.csv
   ```

3. **Evaluate**
   ```bash
   python evaluate.py --model models/xgboost_*.joblib --data data/synthetic_clinical_data.csv
   ```

4. **Start API**
   ```bash
   make run-api
   ```

5. **Test API**
   - Open http://localhost:8000/docs
   - Try the interactive API documentation

## Common Issues

### Import Errors

If you get import errors, ensure you're in the project root and have installed dependencies:

```bash
pip install -e .
```

### MLflow Not Found

Start MLflow server:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000
```

### Port Already in Use

Change the port:

```bash
uvicorn src.api.main:app --port 8001
```

## Next Steps

- Read [COMPLETE_ARCHITECTURE.md](docs/COMPLETE_ARCHITECTURE.md) for system design
- Check [README.md](README.md) for detailed documentation
- Review [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for completion status
- Explore example notebooks in `notebooks/` directory
- Modify `configs/model_config.yaml` for custom configurations

## Support

- Issues: https://github.com/JB5000/04_clinical_risk_modeling_interpretability_bias/issues
- Documentation: See `docs/` directory
