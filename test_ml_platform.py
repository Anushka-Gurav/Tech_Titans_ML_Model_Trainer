"""
Comprehensive pytest test suite for ML Model Comparison Platform
Tests cover API endpoints, data processing, model training, and error handling
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
import io
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from motor.motor_asyncio import AsyncIOMotorClient
import joblib

# Import the FastAPI app and components
import sys
sys.path.append(str(Path(__file__).parent / "backend"))

# Mock environment variables before importing server
os.environ.setdefault('MONGO_URL', 'mongodb://localhost:27017')
os.environ.setdefault('DB_NAME', 'test_ml_db')

# Import after setting env vars
from backend.server import app, db, clean_data, SUPERVISED_MODELS, UNSUPERVISED_MODELS


class TestMLPlatformAPI:
    """Test class for ML Platform API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing"""
        return """sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica"""
    
    @pytest.fixture
    def sample_regression_data(self):
        """Create sample regression data"""
        return """feature1,feature2,feature3,target
1.0,2.0,3.0,10.5
2.0,3.0,4.0,15.2
3.0,4.0,5.0,20.1
4.0,5.0,6.0,25.3
5.0,6.0,7.0,30.0"""
    
    def test_api_root(self, client):
        """Test API root endpoint"""
        response = client.get("/api/")
        assert response.status_code == 200
        assert response.json() == {"message": "ML Training Platform API"}
    
    def test_models_list(self, client):
        """Test getting models list"""
        response = client.get("/api/models/list")
        assert response.status_code == 200
        
        data = response.json()
        assert "supervised" in data
        assert "unsupervised" in data
        assert "classification" in data["supervised"]
        assert "regression" in data["supervised"]
        
        # Check if expected models are present
        assert "Logistic Regression" in data["supervised"]["classification"]
        assert "Linear Regression" in data["supervised"]["regression"]
        assert "K-Means" in data["unsupervised"]
    
    def test_model_parameters_supervised(self, client):
        """Test getting supervised model parameters"""
        # Test classification model
        response = client.get("/api/models/parameters/supervised/Logistic Regression?model_category=classification")
        assert response.status_code == 200
        
        params = response.json()
        assert "C" in params
        assert "max_iter" in params
        
        # Test regression model
        response = client.get("/api/models/parameters/supervised/Linear Regression?model_category=regression")
        assert response.status_code == 200
    
    def test_model_parameters_unsupervised(self, client):
        """Test getting unsupervised model parameters"""
        response = client.get("/api/models/parameters/unsupervised/K-Means")
        assert response.status_code == 200
        
        params = response.json()
        assert "n_clusters" in params
        assert "random_state" in params
    
    def test_model_parameters_invalid(self, client):
        """Test invalid model parameter requests"""
        # Invalid model type
        response = client.get("/api/models/parameters/invalid/SomeModel")
        assert response.status_code == 400
        
        # Missing category for supervised
        response = client.get("/api/models/parameters/supervised/Logistic Regression")
        assert response.status_code == 400
        
        # Non-existent model
        response = client.get("/api/models/parameters/supervised/NonExistentModel?model_category=classification")
        assert response.status_code == 404
    
    @patch('backend.server.db')
    def test_dataset_upload_csv(self, mock_db, client, sample_csv_data):
        """Test CSV dataset upload"""
        mock_db.datasets.insert_one = AsyncMock()
        
        files = {"file": ("test.csv", sample_csv_data, "text/csv")}
        response = client.post("/api/dataset/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "dataset_id" in data
        assert data["filename"] == "test.csv"
        assert data["rows"] == 7
        assert data["columns"] == 5
        assert "preview" in data
    
    def test_dataset_upload_invalid_format(self, client):
        """Test upload with invalid file format"""
        files = {"file": ("test.txt", "invalid content", "text/plain")}
        response = client.post("/api/dataset/upload", files=files)
        
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]
    
    @patch('backend.server.db')
    def test_get_dataset_columns(self, mock_db, client):
        """Test getting dataset columns"""
        # Mock database response
        mock_db.datasets.find_one = AsyncMock(return_value={
            "dataset_id": "test-id",
            "column_names": ["col1", "col2", "col3"]
        })
        
        response = client.get("/api/dataset/test-id/columns")
        assert response.status_code == 200
        
        data = response.json()
        assert data["columns"] == ["col1", "col2", "col3"]
    
    @patch('backend.server.db')
    def test_get_dataset_columns_not_found(self, mock_db, client):
        """Test getting columns for non-existent dataset"""
        mock_db.datasets.find_one = AsyncMock(return_value=None)
        
        response = client.get("/api/dataset/nonexistent/columns")
        assert response.status_code == 404


class TestDataProcessing:
    """Test class for data processing functions"""
    
    def test_clean_data_basic(self):
        """Test basic data cleaning functionality"""
        # Create test dataframe with issues
        df = pd.DataFrame({
            'numeric_col': [1, 2, np.nan, 4, 5],
            'categorical_col': ['A', 'B', None, 'A', 'C'],
            'target': ['yes', 'no', 'yes', 'no', 'yes']
        })
        
        cleaned_df, steps, encoders, target_encoder = clean_data(df, 'target')
        
        # Check that missing values are handled
        assert not cleaned_df.isnull().any().any()
        
        # Check that categorical columns are encoded
        assert cleaned_df['categorical_col'].dtype in ['int64', 'int32']
        
        # Check that cleaning steps are recorded
        assert len(steps) > 0
        assert any('missing values' in step.lower() for step in steps)
    
    def test_clean_data_duplicates(self):
        """Test duplicate removal"""
        df = pd.DataFrame({
            'col1': [1, 2, 1, 3],
            'col2': [4, 5, 4, 6],
            'target': [0, 1, 0, 1]
        })
        
        cleaned_df, steps, _, _ = clean_data(df, 'target')
        
        # Check duplicates are removed
        assert len(cleaned_df) == 3
        assert any('duplicate' in step.lower() for step in steps)
    
    def test_clean_data_no_target(self):
        """Test cleaning without target column"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['A', 'B', 'C']
        })
        
        cleaned_df, steps, encoders, target_encoder = clean_data(df)
        
        assert target_encoder is None
        assert 'col2' in encoders
        assert cleaned_df['col2'].dtype in ['int64', 'int32']
    
    def test_clean_data_empty_dataframe(self):
        """Test cleaning empty dataframe"""
        df = pd.DataFrame()
        
        cleaned_df, steps, encoders, target_encoder = clean_data(df)
        
        assert len(cleaned_df) == 0
        assert len(encoders) == 0
        assert target_encoder is None


class TestModelConfiguration:
    """Test class for model configuration and parameters"""
    
    def test_supervised_models_structure(self):
        """Test supervised models configuration structure"""
        assert 'classification' in SUPERVISED_MODELS
        assert 'regression' in SUPERVISED_MODELS
        
        # Check classification models
        for model_name, config in SUPERVISED_MODELS['classification'].items():
            assert 'class' in config
            assert 'params' in config
            assert callable(config['class'])
        
        # Check regression models
        for model_name, config in SUPERVISED_MODELS['regression'].items():
            assert 'class' in config
            assert 'params' in config
            assert callable(config['class'])
    
    def test_unsupervised_models_structure(self):
        """Test unsupervised models configuration structure"""
        for model_name, config in UNSUPERVISED_MODELS.items():
            assert 'class' in config
            assert 'params' in config
            assert callable(config['class'])
    
    def test_model_instantiation(self):
        """Test that models can be instantiated with default parameters"""
        # Test classification model
        lr_config = SUPERVISED_MODELS['classification']['Logistic Regression']
        model = lr_config['class']()
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        
        # Test regression model
        linear_config = SUPERVISED_MODELS['regression']['Linear Regression']
        model = linear_config['class']()
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        
        # Test unsupervised model
        kmeans_config = UNSUPERVISED_MODELS['K-Means']
        model = kmeans_config['class'](n_clusters=3)
        assert hasattr(model, 'fit')


class TestAsyncOperations:
    """Test class for async operations and background tasks"""
    
    @pytest.fixture
    def event_loop(self):
        """Create event loop for async tests"""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()
    
    @pytest.mark.asyncio
    @patch('backend.server.db')
    async def test_dataset_clean_async(self, mock_db):
        """Test async dataset cleaning"""
        # Mock database operations
        mock_db.datasets.find_one = AsyncMock(return_value={
            "dataset_id": "test-id",
            "file_path": "/fake/path/test.csv"
        })
        mock_db.datasets.update_one = AsyncMock()
        
        # Mock file operations
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({
                'col1': [1, 2, 3],
                'col2': ['A', 'B', 'C'],
                'target': [0, 1, 0]
            })
            
            with patch('pandas.DataFrame.to_csv'):
                client = TestClient(app)
                response = client.post("/api/dataset/clean", json={
                    "dataset_id": "test-id",
                    "target_column": "target"
                })
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "cleaned"
                assert "steps" in data


class TestErrorHandling:
    """Test class for error handling scenarios"""
    
    def test_invalid_json_request(self):
        """Test handling of invalid JSON requests"""
        client = TestClient(app)
        response = client.post("/api/dataset/clean", data="invalid json")
        assert response.status_code == 422
    
    @patch('backend.server.db')
    def test_database_connection_error(self, mock_db):
        """Test handling of database connection errors"""
        mock_db.datasets.find_one = AsyncMock(side_effect=Exception("Database error"))
        
        client = TestClient(app)
        response = client.get("/api/dataset/test-id/columns")
        assert response.status_code == 500
    
    def test_file_processing_error(self):
        """Test handling of file processing errors"""
        client = TestClient(app)
        
        # Send corrupted CSV
        files = {"file": ("test.csv", "invalid,csv,data\n1,2", "text/csv")}
        response = client.post("/api/dataset/upload", files=files)
        
        # Should handle gracefully
        assert response.status_code in [400, 500]


class TestModelTraining:
    """Test class for model training functionality"""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data"""
        return pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
    
    def test_model_training_request_validation(self):
        """Test model training request validation"""
        client = TestClient(app)
        
        # Missing required fields
        response = client.post("/api/model/train", json={})
        assert response.status_code == 422
        
        # Invalid model type
        response = client.post("/api/model/train", json={
            "dataset_id": "test-id",
            "model_type": "invalid",
            "model_name": "Test Model",
            "parameters": {}
        })
        assert response.status_code == 422


class TestVisualization:
    """Test class for visualization generation"""
    
    def test_visualization_imports(self):
        """Test that visualization libraries are properly imported"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import base64
        
        # Test basic plot creation
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        
        # Test saving to bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        # Test base64 encoding
        encoded = base64.b64encode(buffer.read()).decode('utf-8')
        assert len(encoded) > 0
        
        plt.close()


class TestUtilities:
    """Test class for utility functions and helpers"""
    
    def test_uuid_generation(self):
        """Test UUID generation for dataset IDs"""
        import uuid
        
        id1 = str(uuid.uuid4())
        id2 = str(uuid.uuid4())
        
        assert id1 != id2
        assert len(id1) == 36  # Standard UUID length
    
    def test_file_path_handling(self):
        """Test file path operations"""
        from pathlib import Path
        
        # Test path creation
        test_path = Path("test") / "file.csv"
        assert str(test_path).endswith("file.csv")
    
    def test_datetime_operations(self):
        """Test datetime operations"""
        from datetime import datetime, timezone
        
        now = datetime.now(timezone.utc)
        iso_string = now.isoformat()
        
        assert "T" in iso_string
        assert iso_string.endswith("+00:00") or iso_string.endswith("Z")


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def temp_directory():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test"""
    # Clear any existing training jobs
    from backend.server import training_jobs
    training_jobs.clear()
    
    yield
    
    # Cleanup after test
    training_jobs.clear()


# Test runner configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])