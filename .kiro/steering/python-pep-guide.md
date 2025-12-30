---
inclusion: always
---

# Python PEP Style Guide

This document outlines the Python coding standards for the ML Training Platform project, based on PEP 8 and other relevant Python Enhancement Proposals.

## Code Style (PEP 8)

### Indentation

- Use 4 spaces per indentation level
- Never mix tabs and spaces
- Continuation lines should align wrapped elements vertically

### Line Length

- Limit all lines to a maximum of 88 characters (Black formatter standard)
- For docstrings and comments, limit to 72 characters

### Imports (PEP 8)

- Imports should be on separate lines
- Group imports in this order:
  1. Standard library imports
  2. Related third-party imports
  3. Local application/library imports
- Use absolute imports when possible
- Avoid wildcard imports (`from module import *`)

```python
# Good
import os
import sys
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .models import Dataset, TrainedModel
from .utils import generate_id
```

### Naming Conventions (PEP 8)

- **Classes**: Use CapWords (PascalCase)

  ```python
  class DatasetManager:
  class ModelTrainer:
  ```

- **Functions and Variables**: Use lowercase with underscores

  ```python
  def upload_dataset():
  training_session_id = "abc123"
  ```

- **Constants**: Use uppercase with underscores

  ```python
  MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
  DEFAULT_RANDOM_SEED = 42
  ```

- **Private attributes**: Use single leading underscore
  ```python
  class ModelTrainer:
      def __init__(self):
          self._session_storage = {}
  ```

## Type Hints (PEP 484, 526, 585)

### Always use type hints for:

- Function parameters and return types
- Class attributes
- Complex data structures

```python
from typing import List, Dict, Optional, Union
from datetime import datetime

class Dataset:
    id: str
    filename: str
    upload_timestamp: datetime
    columns: List[str]

def upload_dataset(file_path: str, max_size: int = MAX_FILE_SIZE) -> Optional[Dataset]:
    """Upload and validate a CSV dataset."""
    pass

def calculate_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Calculate regression metrics."""
    return {
        "r2_score": 0.95,
        "mse": 0.05,
        "mae": 0.02
    }
```

### Use modern type hints (Python 3.9+)

```python
# Preferred (Python 3.9+)
def process_data(data: list[dict[str, any]]) -> dict[str, list[float]]:
    pass

# Fallback for older Python versions
from typing import List, Dict, Any
def process_data(data: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    pass
```

## Docstrings (PEP 257)

### Use Google-style docstrings for consistency:

```python
def train_model(dataset_id: str, algorithm: str, hyperparams: Dict[str, Any]) -> TrainedModel:
    """Train a machine learning model with specified parameters.

    Args:
        dataset_id: Unique identifier for the dataset
        algorithm: Name of the ML algorithm to use
        hyperparams: Dictionary of hyperparameter values

    Returns:
        TrainedModel instance with training results and metrics

    Raises:
        ValueError: If dataset_id is not found
        RuntimeError: If training fails due to invalid parameters

    Example:
        >>> model = train_model("dataset_123", "random_forest", {"n_estimators": 100})
        >>> print(model.metrics["accuracy"])
        0.95
    """
    pass
```

### Class docstrings:

```python
class ModelTrainer:
    """Handles machine learning model training and evaluation.

    This class orchestrates the complete ML training pipeline including
    data preprocessing, model training, evaluation, and persistence.

    Attributes:
        session_storage: In-memory storage for active training sessions
        model_registry: Registry of trained models with metadata
    """

    def __init__(self):
        self.session_storage: Dict[str, TrainingSession] = {}
        self.model_registry: Dict[str, TrainedModel] = {}
```

## Error Handling

### Use specific exception types:

```python
class DatasetError(Exception):
    """Raised when dataset operations fail."""
    pass

class TrainingError(Exception):
    """Raised when model training fails."""
    pass

def upload_dataset(file_path: str) -> Dataset:
    """Upload dataset with proper error handling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    if not file_path.endswith('.csv'):
        raise DatasetError("Only CSV files are supported")

    try:
        # Process file
        return dataset
    except pd.errors.EmptyDataError:
        raise DatasetError("CSV file is empty or invalid")
```

## Data Classes (PEP 557)

### Use dataclasses for simple data containers:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

@dataclass
class Dataset:
    """Represents an uploaded dataset."""
    id: str
    filename: str
    upload_timestamp: datetime
    file_path: str
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    file_size: int = 0

@dataclass
class TrainingSession:
    """Represents an active training session."""
    id: str
    dataset_id: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    status: str = "PENDING"
    progress_percentage: float = 0.0
    start_time: Optional[datetime] = None
    error_message: Optional[str] = None
```

## Logging

### Use structured logging:

```python
import logging
from typing import Any

logger = logging.getLogger(__name__)

def train_model(dataset_id: str, algorithm: str) -> TrainedModel:
    """Train model with proper logging."""
    logger.info(
        "Starting model training",
        extra={
            "dataset_id": dataset_id,
            "algorithm": algorithm,
            "action": "train_start"
        }
    )

    try:
        # Training logic
        model = perform_training()

        logger.info(
            "Model training completed successfully",
            extra={
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "model_id": model.id,
                "accuracy": model.metrics.get("accuracy"),
                "action": "train_complete"
            }
        )
        return model

    except Exception as e:
        logger.error(
            "Model training failed",
            extra={
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "error": str(e),
                "action": "train_error"
            },
            exc_info=True
        )
        raise
```

## File Organization

### Project structure:

```
backend/
├── app.py                 # Flask application entry point
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── models/               # Data models
│   ├── __init__.py
│   ├── dataset.py
│   ├── training.py
│   └── algorithm.py
├── services/             # Business logic
│   ├── __init__.py
│   ├── dataset_manager.py
│   ├── model_trainer.py
│   ├── metrics_calculator.py
│   └── model_comparator.py
├── api/                  # REST API endpoints
│   ├── __init__.py
│   ├── datasets.py
│   ├── training.py
│   └── models.py
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── file_utils.py
│   └── validation.py
└── tests/                # Test files
    ├── __init__.py
    ├── test_dataset_manager.py
    ├── test_model_trainer.py
    └── test_api.py
```

## Code Quality Tools

### Use these tools for consistent code quality:

1. **Black** - Code formatter

   ```bash
   pip install black
   black --line-length 88 .
   ```

2. **isort** - Import sorting

   ```bash
   pip install isort
   isort --profile black .
   ```

3. **flake8** - Linting

   ```bash
   pip install flake8
   flake8 --max-line-length 88 --extend-ignore E203,W503 .
   ```

4. **mypy** - Type checking
   ```bash
   pip install mypy
   mypy --strict .
   ```

### Pre-commit configuration (.pre-commit-config.yaml):

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        args: [--line-length=88]

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203, W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        args: [--strict]
```

## Testing Standards

### Use pytest with proper structure:

```python
import pytest
from unittest.mock import Mock, patch
from typing import List

from services.dataset_manager import DatasetManager
from models.dataset import Dataset

class TestDatasetManager:
    """Test suite for DatasetManager class."""

    @pytest.fixture
    def dataset_manager(self) -> DatasetManager:
        """Create DatasetManager instance for testing."""
        return DatasetManager()

    @pytest.fixture
    def sample_csv_path(self, tmp_path) -> str:
        """Create a sample CSV file for testing."""
        csv_file = tmp_path / "sample.csv"
        csv_file.write_text("col1,col2\n1,2\n3,4\n")
        return str(csv_file)

    def test_upload_valid_dataset(
        self,
        dataset_manager: DatasetManager,
        sample_csv_path: str
    ) -> None:
        """Test uploading a valid CSV dataset."""
        # Act
        dataset = dataset_manager.upload_dataset(sample_csv_path)

        # Assert
        assert isinstance(dataset, Dataset)
        assert dataset.filename == "sample.csv"
        assert dataset.row_count == 2
        assert len(dataset.columns) == 2

    def test_upload_nonexistent_file(self, dataset_manager: DatasetManager) -> None:
        """Test uploading a file that doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            dataset_manager.upload_dataset("nonexistent.csv")

    @patch('services.dataset_manager.pd.read_csv')
    def test_upload_invalid_csv(
        self,
        mock_read_csv: Mock,
        dataset_manager: DatasetManager,
        sample_csv_path: str
    ) -> None:
        """Test handling of invalid CSV files."""
        mock_read_csv.side_effect = pd.errors.EmptyDataError("No data")

        with pytest.raises(DatasetError, match="CSV file is empty or invalid"):
            dataset_manager.upload_dataset(sample_csv_path)
```

This Python PEP guide ensures consistent, maintainable, and professional Python code throughout the ML Training Platform implementation.
