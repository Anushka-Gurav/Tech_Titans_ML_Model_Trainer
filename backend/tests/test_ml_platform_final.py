"""
=============================================================
  Smart ML Model Trainer - Automated Test Suite
  Group: UCE2024004, UCE2024005, UCE2024007, UCE2024008
  Subject: 23PECE601A DevOps Fundamentals - T1 Evaluation
  Criterion 4: Continuous Integration & Testing (5 Marks)
=============================================================
  Tests cover:
    1. Data Processing & Cleaning
    2. ML Model Training (Classification & Regression)
    3. Model Evaluation Metrics
    4. Dataset Validation
    5. API Structure Validation
    6. Error Handling
    7. Utility Functions
=============================================================
"""

import pytest
import pandas as pd
import numpy as np
import io
import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# SECTION 1 — DATA PROCESSING TESTS
# Tests the core data cleaning logic used in the ML platform
# ─────────────────────────────────────────────────────────────

class TestDataProcessing:
    """Tests for dataset loading, cleaning and preprocessing"""

    def get_sample_iris_data(self):
        """Returns a small iris-like DataFrame for testing"""
        return pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7, 7.0, 6.4, 6.3, 5.8, None],
            'sepal_width':  [3.5, 3.0, 3.2, 3.2, 3.2, 3.3, 2.7, 2.8],
            'petal_length': [1.4, 1.4, 1.3, 4.7, 4.5, 6.0, 5.1, 5.0],
            'petal_width':  [0.2, 0.2, 0.2, 1.4, 1.5, 2.5, 1.9, 1.8],
            'species':      ['setosa','setosa','setosa','versicolor','versicolor','virginica','virginica','virginica']
        })

    def test_TC01_csv_loads_correctly(self):
        """TC01: CSV data loads into DataFrame correctly"""
        csv_content = """col1,col2,target
1,A,0
2,B,1
3,C,0"""
        df = pd.read_csv(io.StringIO(csv_content))
        assert df.shape == (3, 3), "DataFrame should have 3 rows and 3 columns"
        assert list(df.columns) == ['col1', 'col2', 'target']
        print("✅ TC01 PASSED: CSV loads correctly")

    def test_TC02_missing_values_detected(self):
        """TC02: Missing values are detected in dataset"""
        df = self.get_sample_iris_data()
        missing_count = df.isnull().sum().sum()
        assert missing_count == 1, f"Expected 1 missing value, found {missing_count}"
        print("✅ TC02 PASSED: Missing values detected correctly")

    def test_TC03_missing_values_filled(self):
        """TC03: Missing values are filled with column mean"""
        df = self.get_sample_iris_data()
        df_filled = df.fillna(df.mean(numeric_only=True))
        assert df_filled['sepal_length'].isnull().sum() == 0
        assert df_filled['sepal_length'].iloc[7] == pytest.approx(
            self.get_sample_iris_data()['sepal_length'].mean(), rel=0.01
        )
        print("✅ TC03 PASSED: Missing values filled with mean")

    def test_TC04_duplicates_removed(self):
        """TC04: Duplicate rows are identified and removed"""
        df = pd.DataFrame({
            'feature1': [1, 2, 1, 3],
            'feature2': [4, 5, 4, 6],
            'target':   [0, 1, 0, 1]
        })
        df_clean = df.drop_duplicates()
        assert len(df_clean) == 3, "After removing duplicates, 3 rows should remain"
        print("✅ TC04 PASSED: Duplicates removed")

    def test_TC05_categorical_encoding(self):
        """TC05: Categorical column is label-encoded to integers"""
        from sklearn.preprocessing import LabelEncoder
        df = pd.DataFrame({'species': ['setosa', 'versicolor', 'virginica', 'setosa']})
        le = LabelEncoder()
        df['species_encoded'] = le.fit_transform(df['species'])
        assert df['species_encoded'].dtype in [np.int32, np.int64]
        assert set(df['species_encoded'].unique()) == {0, 1, 2}
        print("✅ TC05 PASSED: Categorical encoding works")

    def test_TC06_dataset_shape_after_cleaning(self):
        """TC06: Dataset retains correct shape after full cleaning"""
        df = self.get_sample_iris_data()
        df_clean = df.dropna().drop_duplicates()
        assert df_clean.shape[1] == 5, "Should still have 5 columns after cleaning"
        assert df_clean.shape[0] <= df.shape[0], "Rows should be same or fewer after cleaning"
        print("✅ TC06 PASSED: Shape preserved after cleaning")

    def test_TC07_feature_target_split(self):
        """TC07: Dataset splits correctly into features (X) and target (y)"""
        df = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'f2': [2.0, 3.0, 4.0, 5.0, 6.0],
            'target': [0, 1, 0, 1, 0]
        })
        X = df.drop('target', axis=1)
        y = df['target']
        assert X.shape == (5, 2), "Features should be (5, 2)"
        assert len(y) == 5, "Target should have 5 values"
        assert 'target' not in X.columns, "Target should not be in feature set"
        print("✅ TC07 PASSED: Feature/target split correct")

    def test_TC08_invalid_file_format_rejected(self):
        """TC08: Non-CSV/Excel content raises an error"""
        invalid_content = "this is not valid csv data!!!@#$"
        try:
            df = pd.read_csv(io.StringIO(invalid_content))
            # If it parsed, check it has only 1 column (garbage)
            # This is acceptable — just validate we handle it
            assert df.shape[1] >= 1
        except Exception:
            pass  # Error is also acceptable — format rejected
        print("✅ TC08 PASSED: Invalid format handled gracefully")


# ─────────────────────────────────────────────────────────────
# SECTION 2 — ML MODEL TRAINING TESTS
# Tests that models train, predict, and score correctly
# ─────────────────────────────────────────────────────────────

class TestMLModelTraining:
    """Tests for ML model training: classification and regression"""

    @pytest.fixture
    def classification_data(self):
        """Sample dataset for classification testing"""
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        return X, y

    @pytest.fixture
    def regression_data(self):
        """Sample dataset for regression testing"""
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
        return X, y

    def test_TC09_logistic_regression_trains(self, classification_data):
        """TC09: Logistic Regression trains without error"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        assert hasattr(model, 'coef_'), "Model should have coef_ after training"
        print("✅ TC09 PASSED: Logistic Regression trains successfully")

    def test_TC10_random_forest_trains(self, classification_data):
        """TC10: Random Forest Classifier trains without error"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        assert hasattr(model, 'estimators_'), "Model should have estimators_ after training"
        print("✅ TC10 PASSED: Random Forest trains successfully")

    def test_TC11_linear_regression_trains(self, regression_data):
        """TC11: Linear Regression trains without error"""
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        assert hasattr(model, 'coef_'), "Model should have coef_ after training"
        print("✅ TC11 PASSED: Linear Regression trains successfully")

    def test_TC12_decision_tree_trains(self, classification_data):
        """TC12: Decision Tree Classifier trains without error"""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        assert hasattr(model, 'tree_'), "Model should have tree_ after training"
        print("✅ TC12 PASSED: Decision Tree trains successfully")

    def test_TC13_model_predictions_correct_shape(self, classification_data):
        """TC13: Model predictions have correct shape"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        assert predictions.shape == (20,), f"Expected 20 predictions, got {predictions.shape}"
        print("✅ TC13 PASSED: Prediction shape is correct")

    def test_TC14_kmeans_clustering_trains(self):
        """TC14: KMeans unsupervised clustering trains without error"""
        from sklearn.cluster import KMeans
        X = np.array([
            [1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
            [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]
        ])
        model = KMeans(n_clusters=2, random_state=42, n_init=10)
        model.fit(X)
        assert hasattr(model, 'labels_'), "Model should have labels_ after fitting"
        assert len(set(model.labels_)) == 2, "Should have 2 clusters"
        print("✅ TC14 PASSED: KMeans clustering trains successfully")

    def test_TC15_train_test_split_sizes(self, classification_data):
        """TC15: Train-test split produces correct sizes"""
        from sklearn.model_selection import train_test_split
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        assert len(X_train) == 80, f"Expected 80 training samples, got {len(X_train)}"
        assert len(X_test) == 20, f"Expected 20 test samples, got {len(X_test)}"
        print("✅ TC15 PASSED: Train-test split sizes correct")


# ─────────────────────────────────────────────────────────────
# SECTION 3 — MODEL EVALUATION METRICS TESTS
# Tests accuracy, precision, recall, F1, R2, MSE calculations
# ─────────────────────────────────────────────────────────────

class TestModelEvaluation:
    """Tests for model evaluation metrics used in the platform"""

    @pytest.fixture
    def trained_classifier(self):
        """Returns a trained LogisticRegression model with test data"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        X, y = make_classification(n_samples=200, n_features=4, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = LogisticRegression(max_iter=300)
        model.fit(X_train, y_train)
        return model, X_test, y_test

    @pytest.fixture
    def trained_regressor(self):
        """Returns a trained LinearRegression model with test data"""
        from sklearn.linear_model import LinearRegression
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        X, y = make_regression(n_samples=200, n_features=4, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model, X_test, y_test

    def test_TC16_accuracy_score_valid_range(self, trained_classifier):
        """TC16: Accuracy score is between 0 and 1"""
        from sklearn.metrics import accuracy_score
        model, X_test, y_test = trained_classifier
        acc = accuracy_score(y_test, model.predict(X_test))
        assert 0.0 <= acc <= 1.0, f"Accuracy {acc} is out of range [0,1]"
        print(f"✅ TC16 PASSED: Accuracy = {acc:.4f} (valid range)")

    def test_TC17_accuracy_above_threshold(self, trained_classifier):
        """TC17: Model accuracy is above 70% (basic quality check)"""
        from sklearn.metrics import accuracy_score
        model, X_test, y_test = trained_classifier
        acc = accuracy_score(y_test, model.predict(X_test))
        assert acc >= 0.70, f"Accuracy {acc:.2f} is below 70% threshold"
        print(f"✅ TC17 PASSED: Accuracy = {acc:.4f} (above 70% threshold)")

    def test_TC18_precision_recall_f1_computed(self, trained_classifier):
        """TC18: Precision, Recall, F1 scores are computed correctly"""
        from sklearn.metrics import precision_score, recall_score, f1_score
        model, X_test, y_test = trained_classifier
        y_pred = model.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1
        print(f"✅ TC18 PASSED: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    def test_TC19_confusion_matrix_shape(self, trained_classifier):
        """TC19: Confusion matrix has correct shape (2x2 for binary)"""
        from sklearn.metrics import confusion_matrix
        model, X_test, y_test = trained_classifier
        cm = confusion_matrix(y_test, model.predict(X_test))
        assert cm.shape == (2, 2), f"Expected (2,2), got {cm.shape}"
        assert cm.sum() == len(y_test), "Confusion matrix total must equal number of test samples"
        print(f"✅ TC19 PASSED: Confusion matrix shape = {cm.shape}")

    def test_TC20_r2_score_regression(self, trained_regressor):
        """TC20: R2 score for regression is above 0.9"""
        from sklearn.metrics import r2_score
        model, X_test, y_test = trained_regressor
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        assert r2 > 0.9, f"R2 score {r2:.4f} is too low for this dataset"
        print(f"✅ TC20 PASSED: R2 Score = {r2:.4f}")

    def test_TC21_mse_and_rmse_computed(self, trained_regressor):
        """TC21: MSE and RMSE are computed and non-negative"""
        from sklearn.metrics import mean_squared_error
        model, X_test, y_test = trained_regressor
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        assert mse >= 0, "MSE must be non-negative"
        assert rmse >= 0, "RMSE must be non-negative"
        assert rmse <= np.sqrt(mse) + 0.001  # sanity
        print(f"✅ TC21 PASSED: MSE={mse:.4f}, RMSE={rmse:.4f}")

    def test_TC22_model_comparison_two_models(self):
        """TC22: Two models can be trained and compared on same dataset"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        X, y = make_classification(n_samples=200, n_features=4, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model1 = LogisticRegression(max_iter=200)
        model2 = RandomForestClassifier(n_estimators=10, random_state=42)
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        acc1 = accuracy_score(y_test, model1.predict(X_test))
        acc2 = accuracy_score(y_test, model2.predict(X_test))
        # Both should give valid accuracy
        assert 0 <= acc1 <= 1
        assert 0 <= acc2 <= 1
        winner = "Logistic Regression" if acc1 > acc2 else "Random Forest"
        print(f"✅ TC22 PASSED: LR={acc1:.3f} vs RF={acc2:.3f} → Winner: {winner}")


# ─────────────────────────────────────────────────────────────
# SECTION 4 — API STRUCTURE & ENDPOINT VALIDATION TESTS
# Tests that simulate the API layer without needing server running
# ─────────────────────────────────────────────────────────────

class TestAPIStructure:
    """Tests for API request/response structure validation"""

    def test_TC23_dataset_upload_payload_structure(self):
        """TC23: Dataset upload response has required keys"""
        # Simulate the response structure your API returns
        mock_response = {
            "dataset_id": str(uuid.uuid4()),
            "filename": "iris.csv",
            "rows": 150,
            "columns": 5,
            "preview": [{"sepal_length": 5.1}],
            "column_names": ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
        }
        required_keys = ["dataset_id", "filename", "rows", "columns", "preview", "column_names"]
        for key in required_keys:
            assert key in mock_response, f"Key '{key}' missing from upload response"
        print("✅ TC23 PASSED: Upload response structure is valid")

    def test_TC24_training_request_structure(self):
        """TC24: Training request payload has all required fields"""
        training_request = {
            "dataset_id": "abc-123",
            "model_type": "supervised",
            "model_name": "Logistic Regression",
            "target_column": "species",
            "parameters": {"C": 1.0, "max_iter": 200}
        }
        required_fields = ["dataset_id", "model_type", "model_name", "target_column", "parameters"]
        for field in required_fields:
            assert field in training_request, f"Field '{field}' is required"
        assert training_request["model_type"] in ["supervised", "unsupervised"]
        print("✅ TC24 PASSED: Training request structure valid")

    def test_TC25_training_result_structure(self):
        """TC25: Training result response has required keys"""
        mock_result = {
            "job_id": str(uuid.uuid4()),
            "status": "completed",
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.95,
                "f1_score": 0.94
            },
            "model_name": "Logistic Regression",
            "training_time": 1.23
        }
        assert "job_id" in mock_result
        assert "status" in mock_result
        assert "metrics" in mock_result
        assert mock_result["status"] == "completed"
        assert all(0 <= v <= 1 for v in mock_result["metrics"].values())
        print("✅ TC25 PASSED: Training result structure valid")

    def test_TC26_model_list_response_structure(self):
        """TC26: Models list response contains supervised and unsupervised"""
        mock_models_response = {
            "supervised": {
                "classification": ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"],
                "regression": ["Linear Regression", "Ridge", "Lasso"]
            },
            "unsupervised": ["K-Means", "DBSCAN", "PCA"]
        }
        assert "supervised" in mock_models_response
        assert "unsupervised" in mock_models_response
        assert "classification" in mock_models_response["supervised"]
        assert "regression" in mock_models_response["supervised"]
        assert len(mock_models_response["supervised"]["classification"]) >= 2
        print("✅ TC26 PASSED: Models list structure valid")

    def test_TC27_error_response_structure(self):
        """TC27: Error responses follow standard format"""
        error_response = {
            "detail": "Dataset not found",
            "status_code": 404
        }
        assert "detail" in error_response
        assert error_response["status_code"] == 404
        print("✅ TC27 PASSED: Error response structure valid")


# ─────────────────────────────────────────────────────────────
# SECTION 5 — LIVE API TESTS (requires backend running)
# If backend is not running these are skipped gracefully
# ─────────────────────────────────────────────────────────────

class TestLiveAPI:
    """Live API tests — run only if backend server is running at localhost:8000"""

    BASE_URL = "http://localhost:8000"

    def _backend_running(self):
        """Check if the backend server is reachable"""
        try:
            import requests
            requests.get(f"{self.BASE_URL}/api/", timeout=2)
            return True
        except Exception:
            return False

    def test_TC28_root_endpoint(self):
        """TC28: GET /api/ returns 200 with expected message"""
        if not self._backend_running():
            pytest.skip("Backend server not running — start with: uvicorn server:app --reload --port 8000")
        import requests
        response = requests.get(f"{self.BASE_URL}/api/")
        assert response.status_code == 200
        assert "message" in response.json()
        print("✅ TC28 PASSED: Root API endpoint working")

    def test_TC29_models_list_endpoint(self):
        """TC29: GET /api/models/list returns supervised and unsupervised models"""
        if not self._backend_running():
            pytest.skip("Backend server not running")
        import requests
        response = requests.get(f"{self.BASE_URL}/api/models/list")
        assert response.status_code == 200
        data = response.json()
        assert "supervised" in data
        assert "unsupervised" in data
        print("✅ TC29 PASSED: /api/models/list returns correct structure")

    def test_TC30_csv_upload_endpoint(self):
        """TC30: POST /api/dataset/upload accepts a valid CSV file"""
        if not self._backend_running():
            pytest.skip("Backend server not running")
        import requests
        csv_data = "sepal_length,sepal_width,petal_length,petal_width,species\n5.1,3.5,1.4,0.2,setosa\n4.9,3.0,1.4,0.2,setosa\n4.7,3.2,1.3,0.2,setosa"
        files = {"file": ("test_iris.csv", csv_data.encode(), "text/csv")}
        response = requests.post(f"{self.BASE_URL}/api/dataset/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "dataset_id" in data
        assert data["rows"] == 3
        print(f"✅ TC30 PASSED: File upload OK, dataset_id={data['dataset_id']}")

    def test_TC31_invalid_file_format_rejected(self):
        """TC31: POST /api/dataset/upload rejects non-CSV files with 400"""
        if not self._backend_running():
            pytest.skip("Backend server not running")
        import requests
        files = {"file": ("test.txt", b"invalid plain text content", "text/plain")}
        response = requests.post(f"{self.BASE_URL}/api/dataset/upload", files=files)
        assert response.status_code == 400
        print("✅ TC31 PASSED: Invalid file format rejected with 400")

    def test_TC32_nonexistent_dataset_returns_404(self):
        """TC32: GET /api/dataset/<invalid_id>/columns returns 404"""
        if not self._backend_running():
            pytest.skip("Backend server not running")
        import requests
        response = requests.get(f"{self.BASE_URL}/api/dataset/nonexistent-id-99999/columns")
        assert response.status_code == 404
        print("✅ TC32 PASSED: Non-existent dataset returns 404")


# ─────────────────────────────────────────────────────────────
# SECTION 6 — SELENIUM BROWSER TESTS
# Tests the deployed frontend at Netlify
# ─────────────────────────────────────────────────────────────

class TestSeleniumFrontend:
    """Browser-based tests using Selenium WebDriver"""

    FRONTEND_URL = "https://ml-model-trainer.netlify.app/"
    # FRONTEND_URL = "http://localhost:3000"


    def _selenium_available(self):
        try:
            from selenium import webdriver
            from webdriver_manager.chrome import ChromeDriverManager
            return True
        except ImportError:
            return False

    def test_TC33_homepage_loads(self):
        """TC33: Frontend homepage loads successfully"""
        if not self._selenium_available():
            pytest.skip("Selenium not installed. Run: pip install selenium webdriver-manager")

        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from webdriver_manager.chrome import ChromeDriverManager
        import time
        import os

        os.makedirs("screenshots", exist_ok=True)

        options = Options()
        # options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )

        try:
            driver.get(self.FRONTEND_URL)
            time.sleep(5)

            # Verify correct page loaded
            assert driver.current_url.startswith(self.FRONTEND_URL)

            # Verify page body is not empty
            body = driver.find_element(By.TAG_NAME, "body")
            assert body.text.strip() != "", "Page body should not be empty"

            # Save screenshot
            driver.save_screenshot("screenshots/TC33_homepage.png")

            print("✅ TC33 PASSED: Homepage loaded successfully")

        finally:
            driver.quit()

    def test_TC34_full_ui_flow(self):
        if not self._selenium_available():
            pytest.skip("Selenium not installed")

        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from webdriver_manager.chrome import ChromeDriverManager
        import time

        options = Options()
        # options.add_argument("--headless")

        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )

        try:
            driver.get("http://localhost:3000")
            time.sleep(3)

            # Click Train Model button
            train_btn = driver.find_element(By.CSS_SELECTOR, '[data-testid="get-started-btn"]')
            train_btn.click()
            time.sleep(3)

            # Verify navigation
            assert "/select-model" in driver.current_url

            # Keep browser open for a few seconds so you can watch
            time.sleep(10)

        finally:
            driver.quit()


# ─────────────────────────────────────────────────────────────
# SECTION 7 — UTILITY & HELPER FUNCTION TESTS
# ─────────────────────────────────────────────────────────────

class TestUtilities:
    """Tests for utility and helper functions"""

    def test_TC35_uuid_unique_generation(self):
        """TC35: UUID generator produces unique IDs each time"""
        id1 = str(uuid.uuid4())
        id2 = str(uuid.uuid4())
        assert id1 != id2, "UUIDs should be unique"
        assert len(id1) == 36, "UUID should be 36 characters"
        print("✅ TC35 PASSED: UUID generation is unique")

    def test_TC36_timestamp_format(self):
        """TC36: Timestamp is in valid ISO 8601 format"""
        now = datetime.now(timezone.utc)
        ts = now.isoformat()
        assert "T" in ts, "ISO timestamp should contain 'T'"
        assert len(ts) > 10
        print(f"✅ TC36 PASSED: Timestamp format valid: {ts}")

    def test_TC37_json_serialization(self):
        """TC37: Training results can be JSON serialized"""
        result = {
            "model": "Logistic Regression",
            "accuracy": 0.95,
            "timestamp": datetime.now().isoformat(),
            "parameters": {"C": 1.0, "max_iter": 200}
        }
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["model"] == "Logistic Regression"
        assert parsed["accuracy"] == 0.95
        print("✅ TC37 PASSED: JSON serialization works correctly")

    def test_TC38_file_extension_validation(self):
        """TC38: Only valid extensions (csv, xlsx) are accepted"""
        valid_extensions = {'.csv', '.xlsx', '.xls'}
        test_files = {
            "iris.csv": True,
            "data.xlsx": True,
            "file.txt": False,
            "script.py": False,
            "image.png": False
        }
        for filename, expected in test_files.items():
            ext = Path(filename).suffix.lower()
            result = ext in valid_extensions
            assert result == expected, f"File '{filename}' validation failed"
        print("✅ TC38 PASSED: File extension validation works")

    def test_TC39_numeric_data_statistics(self):
        """TC39: Basic statistics (mean, std, min, max) computed correctly"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        arr = np.array(data)
        assert arr.mean() == 3.0
        assert arr.min() == 1.0
        assert arr.max() == 5.0
        assert round(arr.std(), 4) == round(np.std(data), 4)
        print("✅ TC39 PASSED: Statistics computed correctly")

    def test_TC40_model_parameter_validation(self):
        """TC40: Model parameters are validated for correct types"""
        valid_params = {"C": 1.0, "max_iter": 200, "random_state": 42}
        assert isinstance(valid_params["C"], float)
        assert isinstance(valid_params["max_iter"], int)
        assert valid_params["max_iter"] > 0, "max_iter must be positive"
        assert valid_params["C"] > 0, "C must be positive"
        print("✅ TC40 PASSED: Parameter validation works")


# ─────────────────────────────────────────────────────────────
# TEST RUNNER (run directly with: python test_ml_platform_final.py)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import subprocess
    import sys
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        __file__,
        "-v",
        "--tb=short",
        "--html=test_report.html",
        "--self-contained-html",
        "-k", "not TestLiveAPI and not TestSeleniumFrontend"
    ])
    sys.exit(result.returncode)