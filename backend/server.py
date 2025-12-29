# from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, BackgroundTasks
# from fastapi.responses import FileResponse, StreamingResponse
# from dotenv import load_dotenv
# from starlette.middleware.cors import CORSMiddleware
# from motor.motor_asyncio import AsyncIOMotorClient
# import os
# import logging
# from pathlib import Path
# from pydantic import BaseModel, Field, ConfigDict
# from typing import List, Optional, Dict, Any
# import uuid
# from datetime import datetime, timezone
# import pandas as pd
# import numpy as np
# import io
# import json
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.svm import SVC, SVR
# from xgboost import XGBClassifier, XGBRegressor
# from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
# from sklearn.decomposition import PCA
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score,
#     confusion_matrix, roc_curve, auc,
#     mean_squared_error, mean_absolute_error, r2_score,
#     silhouette_score
# )
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import seaborn as sns
# import base64

# # Don't import kaggle at module level - it causes sys.exit if config is missing
# KAGGLE_AVAILABLE = False

# ROOT_DIR = Path(__file__).parent
# load_dotenv(ROOT_DIR / '.env')

# # MongoDB connection
# mongo_url = os.environ['MONGO_URL']
# client = AsyncIOMotorClient(mongo_url)
# db = client[os.environ['DB_NAME']]

# # Create directories
# UPLOADS_DIR = ROOT_DIR / 'uploads'
# MODELS_DIR = ROOT_DIR / 'models'
# VIZ_DIR = ROOT_DIR / 'visualizations'
# for directory in [UPLOADS_DIR, MODELS_DIR, VIZ_DIR]:
#     directory.mkdir(exist_ok=True)

# # Create the main app
# app = FastAPI()
# api_router = APIRouter(prefix="/api")

# # Store training jobs in memory for progress tracking
# training_jobs = {}

# # Models Configuration
# SUPERVISED_MODELS = {
#     'classification': {
#         'Logistic Regression': {
#             'class': LogisticRegression,
#             'params': {'C': {'type': 'number', 'default': 1.0}, 'max_iter': {'type': 'number', 'default': 100}}
#         },
#         'Decision Tree': {
#             'class': DecisionTreeClassifier,
#             'params': {'max_depth': {'type': 'number', 'default': None}, 'min_samples_split': {'type': 'number', 'default': 2}}
#         },
#         'Random Forest': {
#             'class': RandomForestClassifier,
#             'params': {'n_estimators': {'type': 'number', 'default': 100}, 'max_depth': {'type': 'number', 'default': None}}
#         },
#         'SVM': {
#             'class': SVC,
#             'params': {'C': {'type': 'number', 'default': 1.0}, 'kernel': {'type': 'select', 'options': ['linear', 'rbf', 'poly'], 'default': 'rbf'}}
#         },
#         'XGBoost': {
#             'class': XGBClassifier,
#             'params': {'n_estimators': {'type': 'number', 'default': 100}, 'learning_rate': {'type': 'number', 'default': 0.1}}
#         }
#     },
#     'regression': {
#         'Linear Regression': {
#             'class': LinearRegression,
#             'params': {}
#         },
#         'Decision Tree': {
#             'class': DecisionTreeRegressor,
#             'params': {'max_depth': {'type': 'number', 'default': None}}
#         },
#         'Random Forest': {
#             'class': RandomForestRegressor,
#             'params': {'n_estimators': {'type': 'number', 'default': 100}}
#         },
#         'SVR': {
#             'class': SVR,
#             'params': {'C': {'type': 'number', 'default': 1.0}, 'kernel': {'type': 'select', 'options': ['linear', 'rbf'], 'default': 'rbf'}}
#         },
#         'XGBoost': {
#             'class': XGBRegressor,
#             'params': {'n_estimators': {'type': 'number', 'default': 100}}
#         }
#     }
# }

# UNSUPERVISED_MODELS = {
#     'K-Means': {
#         'class': KMeans,
#         'params': {'n_clusters': {'type': 'number', 'default': 3}, 'random_state': {'type': 'number', 'default': 42}}
#     },
#     'DBSCAN': {
#         'class': DBSCAN,
#         'params': {'eps': {'type': 'number', 'default': 0.5}, 'min_samples': {'type': 'number', 'default': 5}}
#     },
#     'Hierarchical Clustering': {
#         'class': AgglomerativeClustering,
#         'params': {'n_clusters': {'type': 'number', 'default': 3}}
#     },
#     'PCA': {
#         'class': PCA,
#         'params': {'n_components': {'type': 'number', 'default': 2}}
#     }
# }

# # Pydantic Models
# class ModelListResponse(BaseModel):
#     model_config = ConfigDict(extra="ignore")
#     supervised: Dict[str, Any]
#     unsupervised: List[str]

# class DatasetUploadResponse(BaseModel):
#     model_config = ConfigDict(extra="ignore")
#     dataset_id: str
#     filename: str
#     rows: int
#     columns: int
#     preview: List[Dict]

# class DataCleanRequest(BaseModel):
#     dataset_id: str
#     target_column: Optional[str] = None

# class TrainModelRequest(BaseModel):
#     dataset_id: str
#     model_type: str  # supervised or unsupervised
#     model_category: Optional[str] = None  # classification or regression for supervised
#     model_name: str
#     parameters: Dict[str, Any]
#     target_column: Optional[str] = None

# class CompareModelsRequest(BaseModel):
#     dataset_id: str
#     model_category: str  # classification or regression
#     model1_name: str
#     model2_name: str
#     model1_parameters: Dict[str, Any]
#     model2_parameters: Dict[str, Any]
#     target_column: str

# class TrainingProgressResponse(BaseModel):
#     model_config = ConfigDict(extra="ignore")
#     job_id: str
#     status: str
#     progress: int
#     message: str
#     result: Optional[Dict[str, Any]] = None

# # API Endpoints
# @api_router.get("/")
# async def root():
#     return {"message": "ML Training Platform API"}

# @api_router.get("/models/list", response_model=ModelListResponse)
# async def get_models_list():
#     return {
#         "supervised": {
#             "classification": list(SUPERVISED_MODELS['classification'].keys()),
#             "regression": list(SUPERVISED_MODELS['regression'].keys())
#         },
#         "unsupervised": list(UNSUPERVISED_MODELS.keys())
#     }

# @api_router.get("/models/parameters/{model_type}/{model_name}")
# async def get_model_parameters(model_type: str, model_name: str, model_category: Optional[str] = None):
#     if model_type == 'supervised':
#         if not model_category or model_category not in SUPERVISED_MODELS:
#             raise HTTPException(status_code=400, detail="Model category required for supervised models")
#         if model_name not in SUPERVISED_MODELS[model_category]:
#             raise HTTPException(status_code=404, detail="Model not found")
#         return SUPERVISED_MODELS[model_category][model_name]['params']
#     elif model_type == 'unsupervised':
#         if model_name not in UNSUPERVISED_MODELS:
#             raise HTTPException(status_code=404, detail="Model not found")
#         return UNSUPERVISED_MODELS[model_name]['params']
#     else:
#         raise HTTPException(status_code=400, detail="Invalid model type")

# @api_router.post("/dataset/upload", response_model=DatasetUploadResponse)
# async def upload_dataset(file: UploadFile = File(...)):
#     try:
#         dataset_id = str(uuid.uuid4())
#         file_ext = file.filename.split('.')[-1].lower()
        
#         content = await file.read()
        
#         # Read file based on extension
#         if file_ext == 'csv':
#             df = pd.read_csv(io.BytesIO(content))
#         elif file_ext in ['xlsx', 'xls']:
#             df = pd.read_excel(io.BytesIO(content))
#         else:
#             raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel file.")
        
#         # Save dataset
#         file_path = UPLOADS_DIR / f"{dataset_id}.csv"
#         df.to_csv(file_path, index=False)
        
#         # Store metadata in DB
#         dataset_doc = {
#             "dataset_id": dataset_id,
#             "filename": file.filename,
#             "rows": len(df),
#             "columns": len(df.columns),
#             "column_names": df.columns.tolist(),
#             "file_path": str(file_path),
#             "created_at": datetime.now(timezone.utc).isoformat()
#         }
#         await db.datasets.insert_one(dataset_doc)
        
#         # Return preview
#         preview = df.head(5).to_dict('records')
#         return {
#             "dataset_id": dataset_id,
#             "filename": file.filename,
#             "rows": len(df),
#             "columns": len(df.columns),
#             "preview": preview
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @api_router.post("/dataset/kaggle")
# async def fetch_kaggle_dataset(dataset_name: str, kaggle_username: str, kaggle_key: str):
#     try:
#         # Import kaggle only when needed
#         import kaggle
        
#         # Set Kaggle credentials
#         os.environ['KAGGLE_USERNAME'] = kaggle_username
#         os.environ['KAGGLE_KEY'] = kaggle_key
        
#         dataset_id = str(uuid.uuid4())
#         download_path = UPLOADS_DIR / dataset_id
#         download_path.mkdir(exist_ok=True)
        
#         # Download dataset
#         kaggle.api.dataset_download_files(dataset_name, path=str(download_path), unzip=True)
        
#         # Find CSV files
#         csv_files = list(download_path.glob('*.csv'))
#         if not csv_files:
#             raise HTTPException(status_code=400, detail="No CSV files found in the dataset")
        
#         # Read first CSV
#         df = pd.read_csv(csv_files[0])
        
#         # Save as single CSV
#         file_path = UPLOADS_DIR / f"{dataset_id}.csv"
#         df.to_csv(file_path, index=False)
        
#         # Store metadata
#         dataset_doc = {
#             "dataset_id": dataset_id,
#             "filename": dataset_name,
#             "rows": len(df),
#             "columns": len(df.columns),
#             "column_names": df.columns.tolist(),
#             "file_path": str(file_path),
#             "source": "kaggle",
#             "created_at": datetime.now(timezone.utc).isoformat()
#         }
#         await db.datasets.insert_one(dataset_doc)
        
#         preview = df.head(5).to_dict('records')
#         return {
#             "dataset_id": dataset_id,
#             "filename": dataset_name,
#             "rows": len(df),
#             "columns": len(df.columns),
#             "preview": preview
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Kaggle dataset fetch failed: {str(e)}")

# @api_router.get("/dataset/{dataset_id}/columns")
# async def get_dataset_columns(dataset_id: str):
#     dataset = await db.datasets.find_one({"dataset_id": dataset_id}, {"_id": 0})
#     if not dataset:
#         raise HTTPException(status_code=404, detail="Dataset not found")
#     return {"columns": dataset["column_names"]}

# def clean_data(df: pd.DataFrame, target_column: Optional[str] = None):
#     cleaning_steps = []
    
#     # Remove duplicates
#     initial_rows = len(df)
#     df = df.drop_duplicates()
#     duplicates_removed = initial_rows - len(df)
#     cleaning_steps.append(f"Removed {duplicates_removed} duplicate rows")
    
#     # Handle missing values
#     missing_info = {}
#     for col in df.columns:
#         missing_count = df[col].isnull().sum()
#         if missing_count > 0:
#             missing_info[col] = missing_count
#             if df[col].dtype in ['float64', 'int64']:
#                 df[col].fillna(df[col].median(), inplace=True)
#             else:
#                 df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
#     if missing_info:
#         cleaning_steps.append(f"Handled missing values in {len(missing_info)} columns")
    
#     # Encode categorical variables
#     categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
#     if target_column and target_column in categorical_cols:
#         categorical_cols.remove(target_column)
    
#     label_encoders = {}
#     for col in categorical_cols:
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col].astype(str))
#         label_encoders[col] = le
    
#     if categorical_cols:
#         cleaning_steps.append(f"Encoded {len(categorical_cols)} categorical columns")
    
#     # Handle target column encoding if categorical
#     target_encoder = None
#     if target_column and target_column in df.columns:
#         if df[target_column].dtype == 'object':
#             target_encoder = LabelEncoder()
#             df[target_column] = target_encoder.fit_transform(df[target_column])
#             cleaning_steps.append(f"Encoded target column '{target_column}'")
    
#     return df, cleaning_steps, label_encoders, target_encoder

# @api_router.post("/dataset/clean")
# async def clean_dataset(request: DataCleanRequest):
#     try:
#         dataset = await db.datasets.find_one({"dataset_id": request.dataset_id}, {"_id": 0})
#         if not dataset:
#             raise HTTPException(status_code=404, detail="Dataset not found")
        
#         file_path = Path(dataset["file_path"])
#         df = pd.read_csv(file_path)
        
#         df_cleaned, steps, encoders, target_encoder = clean_data(df, request.target_column)
        
#         # Save cleaned data
#         cleaned_path = UPLOADS_DIR / f"{request.dataset_id}_cleaned.csv"
#         df_cleaned.to_csv(cleaned_path, index=False)
        
#         # Update dataset metadata
#         await db.datasets.update_one(
#             {"dataset_id": request.dataset_id},
#             {"$set": {
#                 "cleaned_path": str(cleaned_path),
#                 "cleaning_steps": steps,
#                 "cleaned_at": datetime.now(timezone.utc).isoformat()
#             }}
#         )
        
#         return {
#             "dataset_id": request.dataset_id,
#             "status": "cleaned",
#             "steps": steps,
#             "rows_after_cleaning": len(df_cleaned)
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# async def train_model_task(job_id: str, dataset_id: str, model_type: str, model_category: Optional[str],
#                           model_name: str, parameters: Dict, target_column: Optional[str]):
#     try:
#         training_jobs[job_id] = {"status": "running", "progress": 10, "message": "Loading dataset..."}
        
#         # Load dataset
#         dataset = await db.datasets.find_one({"dataset_id": dataset_id}, {"_id": 0})
#         if not dataset:
#             training_jobs[job_id] = {"status": "failed", "progress": 0, "message": "Dataset not found"}
#             return
        
#         file_path = dataset.get("cleaned_path") or dataset["file_path"]
#         df = pd.read_csv(file_path)
        
#         training_jobs[job_id] = {"status": "running", "progress": 30, "message": "Preparing data..."}
        
#         # Prepare data based on model type
#         if model_type == 'supervised':
#             if not target_column or target_column not in df.columns:
#                 training_jobs[job_id] = {"status": "failed", "progress": 0, "message": "Invalid target column"}
#                 return
            
#             X = df.drop(columns=[target_column])
#             y = df[target_column]
            
#             # Check if X has data
#             if X.empty or len(X.columns) == 0:
#                 training_jobs[job_id] = {"status": "failed", "progress": 0, "message": "No feature columns available for training"}
#                 return
            
#             # Ensure all columns are numeric
#             for col in X.columns:
#                 if X[col].dtype == 'object':
#                     le = LabelEncoder()
#                     X[col] = le.fit_transform(X[col].astype(str))
            
#             # Convert to numpy array and ensure it's 2D
#             X = X.values
#             y = y.values
            
#             # Scale features
#             scaler = StandardScaler()
#             X_scaled = scaler.fit_transform(X)
            
#             # Train-test split
#             X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
#             training_jobs[job_id] = {"status": "running", "progress": 50, "message": "Training model..."}
            
#             # Get model class
#             model_config = SUPERVISED_MODELS[model_category][model_name]
#             model_class = model_config['class']
            
#             # Create model with parameters
#             model_params = {}
#             for param, value in parameters.items():
#                 if value is not None and value != '':
#                     model_params[param] = value
            
#             model = model_class(**model_params)
#             model.fit(X_train, y_train)
            
#             training_jobs[job_id] = {"status": "running", "progress": 70, "message": "Evaluating model..."}
            
#             # Predictions
#             y_pred = model.predict(X_test)
            
#             # Generate metrics and visualizations
#             results = {}
#             visualizations = {}
            
#             if model_category == 'classification':
#                 results['accuracy'] = float(accuracy_score(y_test, y_pred))
#                 results['precision'] = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
#                 results['recall'] = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
#                 results['f1_score'] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                
#                 # Confusion Matrix
#                 cm = confusion_matrix(y_test, y_pred)
#                 plt.figure(figsize=(8, 6))
#                 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#                 plt.title('Confusion Matrix')
#                 plt.ylabel('Actual')
#                 plt.xlabel('Predicted')
#                 cm_path = VIZ_DIR / f"{job_id}_confusion_matrix.png"
#                 plt.savefig(cm_path)
#                 plt.close()
                
#                 with open(cm_path, 'rb') as img_file:
#                     visualizations['confusion_matrix'] = base64.b64encode(img_file.read()).decode('utf-8')
                
#                 # ROC Curve (for binary classification)
#                 if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
#                     y_proba = model.predict_proba(X_test)[:, 1]
#                     fpr, tpr, _ = roc_curve(y_test, y_proba)
#                     roc_auc = auc(fpr, tpr)
                    
#                     plt.figure(figsize=(8, 6))
#                     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
#                     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#                     plt.xlim([0.0, 1.0])
#                     plt.ylim([0.0, 1.05])
#                     plt.xlabel('False Positive Rate')
#                     plt.ylabel('True Positive Rate')
#                     plt.title('ROC Curve')
#                     plt.legend(loc="lower right")
#                     roc_path = VIZ_DIR / f"{job_id}_roc_curve.png"
#                     plt.savefig(roc_path)
#                     plt.close()
                    
#                     with open(roc_path, 'rb') as img_file:
#                         visualizations['roc_curve'] = base64.b64encode(img_file.read()).decode('utf-8')
#                     results['roc_auc'] = float(roc_auc)
            
#             elif model_category == 'regression':
#                 results['mse'] = float(mean_squared_error(y_test, y_pred))
#                 results['rmse'] = float(np.sqrt(results['mse']))
#                 results['mae'] = float(mean_absolute_error(y_test, y_pred))
#                 results['r2_score'] = float(r2_score(y_test, y_pred))
                
#                 # Scatter plot
#                 plt.figure(figsize=(8, 6))
#                 plt.scatter(y_test, y_pred, alpha=0.5)
#                 plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
#                 plt.xlabel('Actual')
#                 plt.ylabel('Predicted')
#                 plt.title('Actual vs Predicted')
#                 scatter_path = VIZ_DIR / f"{job_id}_scatter.png"
#                 plt.savefig(scatter_path)
#                 plt.close()
                
#                 with open(scatter_path, 'rb') as img_file:
#                     visualizations['scatter_plot'] = base64.b64encode(img_file.read()).decode('utf-8')
                
#                 # Residual plot
#                 residuals = y_test - y_pred
#                 plt.figure(figsize=(8, 6))
#                 plt.scatter(y_pred, residuals, alpha=0.5)
#                 plt.axhline(y=0, color='r', linestyle='--')
#                 plt.xlabel('Predicted')
#                 plt.ylabel('Residuals')
#                 plt.title('Residual Plot')
#                 residual_path = VIZ_DIR / f"{job_id}_residuals.png"
#                 plt.savefig(residual_path)
#                 plt.close()
                
#                 with open(residual_path, 'rb') as img_file:
#                     visualizations['residual_plot'] = base64.b64encode(img_file.read()).decode('utf-8')
            
#             # Save model
#             model_path = MODELS_DIR / f"{job_id}.pkl"
#             joblib.dump({'model': model, 'scaler': scaler}, model_path)
            
#         else:  # Unsupervised
#             # Ensure all columns are numeric
#             for col in df.columns:
#                 if df[col].dtype == 'object':
#                     le = LabelEncoder()
#                     df[col] = le.fit_transform(df[col].astype(str))
            
#             X = df.values
            
#             if X.size == 0:
#                 training_jobs[job_id] = {"status": "failed", "progress": 0, "message": "No data available for training"}
#                 return
            
#             scaler = StandardScaler()
#             X_scaled = scaler.fit_transform(X)
            
#             training_jobs[job_id] = {"status": "running", "progress": 50, "message": "Training model..."}
            
#             model_config = UNSUPERVISED_MODELS[model_name]
#             model_class = model_config['class']
            
#             model_params = {}
#             for param, value in parameters.items():
#                 if value is not None and value != '':
#                     model_params[param] = value
            
#             model = model_class(**model_params)
            
#             if model_name == 'PCA':
#                 X_transformed = model.fit_transform(X_scaled)
#                 results = {'explained_variance_ratio': model.explained_variance_ratio_.tolist()}
                
#                 # PCA scatter plot
#                 plt.figure(figsize=(8, 6))
#                 plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.5)
#                 plt.xlabel('PC1')
#                 plt.ylabel('PC2')
#                 plt.title('PCA Visualization')
#                 pca_path = VIZ_DIR / f"{job_id}_pca.png"
#                 plt.savefig(pca_path)
#                 plt.close()
                
#                 with open(pca_path, 'rb') as img_file:
#                     visualizations['pca_plot'] = base64.b64encode(img_file.read()).decode('utf-8')
#             else:
#                 labels = model.fit_predict(X_scaled)
                
#                 # Silhouette score
#                 if len(np.unique(labels)) > 1:
#                     sil_score = silhouette_score(X_scaled, labels)
#                     results['silhouette_score'] = float(sil_score)
                
#                 results['n_clusters'] = int(len(np.unique(labels)))
                
#                 # Cluster visualization (2D PCA)
#                 pca = PCA(n_components=2)
#                 X_pca = pca.fit_transform(X_scaled)
                
#                 plt.figure(figsize=(8, 6))
#                 scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
#                 plt.colorbar(scatter)
#                 plt.xlabel('PC1')
#                 plt.ylabel('PC2')
#                 plt.title('Cluster Visualization')
#                 cluster_path = VIZ_DIR / f"{job_id}_clusters.png"
#                 plt.savefig(cluster_path)
#                 plt.close()
                
#                 with open(cluster_path, 'rb') as img_file:
#                     visualizations['cluster_plot'] = base64.b64encode(img_file.read()).decode('utf-8')
            
#             # Save model
#             model_path = MODELS_DIR / f"{job_id}.pkl"
#             joblib.dump({'model': model, 'scaler': scaler}, model_path)
        
#         training_jobs[job_id] = {"status": "running", "progress": 90, "message": "Saving results..."}
        
#         # Save to database
#         job_doc = {
#             "job_id": job_id,
#             "dataset_id": dataset_id,
#             "model_type": model_type,
#             "model_category": model_category,
#             "model_name": model_name,
#             "parameters": parameters,
#             "results": results,
#             "model_path": str(model_path),
#             "completed_at": datetime.now(timezone.utc).isoformat()
#         }
#         await db.training_jobs.insert_one(job_doc)
        
#         training_jobs[job_id] = {
#             "status": "completed",
#             "progress": 100,
#             "message": "Training completed successfully",
#             "result": {
#                 "metrics": results,
#                 "visualizations": visualizations,
#                 "model_path": str(model_path)
#             }
#         }
#     except Exception as e:
#         training_jobs[job_id] = {"status": "failed", "progress": 0, "message": str(e)}

# @api_router.post("/model/train")
# async def train_model(request: TrainModelRequest, background_tasks: BackgroundTasks):
#     job_id = str(uuid.uuid4())
#     training_jobs[job_id] = {"status": "initializing", "progress": 0, "message": "Starting training..."}
    
#     background_tasks.add_task(
#         train_model_task,
#         job_id,
#         request.dataset_id,
#         request.model_type,
#         request.model_category,
#         request.model_name,
#         request.parameters,
#         request.target_column
#     )
    
#     return {"job_id": job_id}

# @api_router.get("/model/progress/{job_id}", response_model=TrainingProgressResponse)
# async def get_training_progress(job_id: str):
#     if job_id not in training_jobs:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     job_data = training_jobs[job_id]
#     return {
#         "job_id": job_id,
#         "status": job_data["status"],
#         "progress": job_data["progress"],
#         "message": job_data["message"],
#         "result": job_data.get("result")
#     }

# @api_router.get("/model/download/{job_id}")
# async def download_model(job_id: str):
#     job = await db.training_jobs.find_one({"job_id": job_id}, {"_id": 0})
#     if not job:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     model_path = Path(job["model_path"])
#     if not model_path.exists():
#         raise HTTPException(status_code=404, detail="Model file not found")
    
#     return FileResponse(
#         path=model_path,
#         filename=f"model_{job_id}.pkl",
#         media_type="application/octet-stream"
#     )

# async def compare_models_task(job_id: str, dataset_id: str, model_category: str,
#                               model1_name: str, model2_name: str,
#                               model1_params: Dict, model2_params: Dict, target_column: str):
#     try:
#         training_jobs[job_id] = {"status": "running", "progress": 10, "message": "Loading dataset..."}
        
#         # Load dataset
#         dataset = await db.datasets.find_one({"dataset_id": dataset_id}, {"_id": 0})
#         if not dataset:
#             training_jobs[job_id] = {"status": "failed", "progress": 0, "message": "Dataset not found"}
#             return
        
#         file_path = dataset.get("cleaned_path") or dataset["file_path"]
#         df = pd.read_csv(file_path)
        
#         training_jobs[job_id] = {"status": "running", "progress": 20, "message": "Preparing data..."}
        
#         if target_column not in df.columns:
#             training_jobs[job_id] = {"status": "failed", "progress": 0, "message": "Invalid target column"}
#             return
        
#         X = df.drop(columns=[target_column])
#         y = df[target_column]
        
#         # Ensure all columns are numeric
#         for col in X.columns:
#             if X[col].dtype == 'object':
#                 le = LabelEncoder()
#                 X[col] = le.fit_transform(X[col].astype(str))
        
#         X = X.values
#         y = y.values
        
#         # Scale features
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
        
#         # Train-test split
#         X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
#         # Train Model 1
#         training_jobs[job_id] = {"status": "running", "progress": 40, "message": f"Training {model1_name}..."}
        
#         model1_config = SUPERVISED_MODELS[model_category][model1_name]
#         model1_class = model1_config['class']
#         model1 = model1_class(**{k: v for k, v in model1_params.items() if v is not None and v != ''})
#         model1.fit(X_train, y_train)
#         y_pred1 = model1.predict(X_test)
        
#         # Get Model 1 metrics
#         model1_metrics = {}
#         if model_category == 'classification':
#             model1_metrics['accuracy'] = float(accuracy_score(y_test, y_pred1))
#             model1_metrics['precision'] = float(precision_score(y_test, y_pred1, average='weighted', zero_division=0))
#             model1_metrics['recall'] = float(recall_score(y_test, y_pred1, average='weighted', zero_division=0))
#             model1_metrics['f1_score'] = float(f1_score(y_test, y_pred1, average='weighted', zero_division=0))
#         else:  # regression
#             model1_metrics['mse'] = float(mean_squared_error(y_test, y_pred1))
#             model1_metrics['rmse'] = float(np.sqrt(model1_metrics['mse']))
#             model1_metrics['mae'] = float(mean_absolute_error(y_test, y_pred1))
#             model1_metrics['r2_score'] = float(r2_score(y_test, y_pred1))
        
#         # Train Model 2
#         training_jobs[job_id] = {"status": "running", "progress": 70, "message": f"Training {model2_name}..."}
        
#         model2_config = SUPERVISED_MODELS[model_category][model2_name]
#         model2_class = model2_config['class']
#         model2 = model2_class(**{k: v for k, v in model2_params.items() if v is not None and v != ''})
#         model2.fit(X_train, y_train)
#         y_pred2 = model2.predict(X_test)
        
#         # Get Model 2 metrics
#         model2_metrics = {}
#         if model_category == 'classification':
#             model2_metrics['accuracy'] = float(accuracy_score(y_test, y_pred2))
#             model2_metrics['precision'] = float(precision_score(y_test, y_pred2, average='weighted', zero_division=0))
#             model2_metrics['recall'] = float(recall_score(y_test, y_pred2, average='weighted', zero_division=0))
#             model2_metrics['f1_score'] = float(f1_score(y_test, y_pred2, average='weighted', zero_division=0))
#         else:  # regression
#             model2_metrics['mse'] = float(mean_squared_error(y_test, y_pred2))
#             model2_metrics['rmse'] = float(np.sqrt(model2_metrics['mse']))
#             model2_metrics['mae'] = float(mean_absolute_error(y_test, y_pred2))
#             model2_metrics['r2_score'] = float(r2_score(y_test, y_pred2))
        
#         # Determine winner
#         training_jobs[job_id] = {"status": "running", "progress": 90, "message": "Comparing results..."}
        
#         if model_category == 'classification':
#             # Winner has higher F1 score
#             winner = model1_name if model1_metrics['f1_score'] > model2_metrics['f1_score'] else model2_name
#         else:
#             # Winner has lower RMSE
#             winner = model1_name if model1_metrics['rmse'] < model2_metrics['rmse'] else model2_name
        
#         # Save results
#         comparison_result = {
#             "model1_name": model1_name,
#             "model2_name": model2_name,
#             "model1_metrics": model1_metrics,
#             "model2_metrics": model2_metrics,
#             "winner": winner
#         }
        
#         comparison_doc = {
#             "job_id": job_id,
#             "dataset_id": dataset_id,
#             "model_category": model_category,
#             "comparison": comparison_result,
#             "completed_at": datetime.now(timezone.utc).isoformat()
#         }
#         await db.training_jobs.insert_one(comparison_doc)
        
#         training_jobs[job_id] = {
#             "status": "completed",
#             "progress": 100,
#             "message": "Comparison completed successfully",
#             "result": {
#                 "comparison": comparison_result
#             }
#         }
#     except Exception as e:
#         training_jobs[job_id] = {"status": "failed", "progress": 0, "message": str(e)}

# @api_router.post("/model/compare")
# async def compare_models(request: CompareModelsRequest, background_tasks: BackgroundTasks):
#     job_id = str(uuid.uuid4())
#     training_jobs[job_id] = {"status": "initializing", "progress": 0, "message": "Starting comparison..."}
    
#     background_tasks.add_task(
#         compare_models_task,
#         job_id,
#         request.dataset_id,
#         request.model_category,
#         request.model1_name,
#         request.model2_name,
#         request.model1_parameters,
#         request.model2_parameters,
#         request.target_column
#     )
    
#     return {"job_id": job_id}

# app.include_router(api_router)

# app.add_middleware(
#     CORSMiddleware,
#     allow_credentials=True,
#     allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# @app.on_event("shutdown")
# async def shutdown_db_client():
#     client.close()



from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import io
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Don't import kaggle at module level - it causes sys.exit if config is missing
KAGGLE_AVAILABLE = False

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create directories
UPLOADS_DIR = ROOT_DIR / 'uploads'
MODELS_DIR = ROOT_DIR / 'models'
VIZ_DIR = ROOT_DIR / 'visualizations'
for directory in [UPLOADS_DIR, MODELS_DIR, VIZ_DIR]:
    directory.mkdir(exist_ok=True)

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Store training jobs in memory for progress tracking
training_jobs = {}

# Models Configuration
SUPERVISED_MODELS = {
    'classification': {
        'Logistic Regression': {
            'class': LogisticRegression,
            'params': {'C': {'type': 'number', 'default': 1.0}, 'max_iter': {'type': 'number', 'default': 100}}
        },
        'Decision Tree': {
            'class': DecisionTreeClassifier,
            'params': {'max_depth': {'type': 'number', 'default': None}, 'min_samples_split': {'type': 'number', 'default': 2}}
        },
        'Random Forest': {
            'class': RandomForestClassifier,
            'params': {'n_estimators': {'type': 'number', 'default': 100}, 'max_depth': {'type': 'number', 'default': None}}
        },
        'SVM': {
            'class': SVC,
            'params': {'C': {'type': 'number', 'default': 1.0}, 'kernel': {'type': 'select', 'options': ['linear', 'rbf', 'poly'], 'default': 'rbf'}}
        },
        'XGBoost': {
            'class': XGBClassifier,
            'params': {'n_estimators': {'type': 'number', 'default': 100}, 'learning_rate': {'type': 'number', 'default': 0.1}}
        }
    },
    'regression': {
        'Linear Regression': {
            'class': LinearRegression,
            'params': {}
        },
        'Decision Tree': {
            'class': DecisionTreeRegressor,
            'params': {'max_depth': {'type': 'number', 'default': None}}
        },
        'Random Forest': {
            'class': RandomForestRegressor,
            'params': {'n_estimators': {'type': 'number', 'default': 100}}
        },
        'SVR': {
            'class': SVR,
            'params': {'C': {'type': 'number', 'default': 1.0}, 'kernel': {'type': 'select', 'options': ['linear', 'rbf'], 'default': 'rbf'}}
        },
        'XGBoost': {
            'class': XGBRegressor,
            'params': {'n_estimators': {'type': 'number', 'default': 100}}
        }
    }
}

UNSUPERVISED_MODELS = {
    'K-Means': {
        'class': KMeans,
        'params': {'n_clusters': {'type': 'number', 'default': 3}, 'random_state': {'type': 'number', 'default': 42}}
    },
    'DBSCAN': {
        'class': DBSCAN,
        'params': {'eps': {'type': 'number', 'default': 0.5}, 'min_samples': {'type': 'number', 'default': 5}}
    },
    'Hierarchical Clustering': {
        'class': AgglomerativeClustering,
        'params': {'n_clusters': {'type': 'number', 'default': 3}}
    },
    'PCA': {
        'class': PCA,
        'params': {'n_components': {'type': 'number', 'default': 2}}
    }
}

# Pydantic Models
class ModelListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    supervised: Dict[str, Any]
    unsupervised: List[str]

class DatasetUploadResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    dataset_id: str
    filename: str
    rows: int
    columns: int
    preview: List[Dict]

class DataCleanRequest(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None

class TrainModelRequest(BaseModel):
    dataset_id: str
    model_type: str  # supervised or unsupervised
    model_category: Optional[str] = None  # classification or regression for supervised
    model_name: str
    parameters: Dict[str, Any]
    target_column: Optional[str] = None

class CompareModelsRequest(BaseModel):
    dataset_id: str
    model_category: str  # classification or regression
    model1_name: str
    model2_name: str
    model1_parameters: Dict[str, Any]
    model2_parameters: Dict[str, Any]
    target_column: str

class TrainingProgressResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    job_id: str
    status: str
    progress: int
    message: str
    result: Optional[Dict[str, Any]] = None

# API Endpoints
@api_router.get("/")
async def root():
    return {"message": "ML Training Platform API"}

@api_router.get("/models/list", response_model=ModelListResponse)
async def get_models_list():
    return {
        "supervised": {
            "classification": list(SUPERVISED_MODELS['classification'].keys()),
            "regression": list(SUPERVISED_MODELS['regression'].keys())
        },
        "unsupervised": list(UNSUPERVISED_MODELS.keys())
    }

@api_router.get("/models/parameters/{model_type}/{model_name}")
async def get_model_parameters(model_type: str, model_name: str, model_category: Optional[str] = None):
    if model_type == 'supervised':
        if not model_category or model_category not in SUPERVISED_MODELS:
            raise HTTPException(status_code=400, detail="Model category required for supervised models")
        if model_name not in SUPERVISED_MODELS[model_category]:
            raise HTTPException(status_code=404, detail="Model not found")
        return SUPERVISED_MODELS[model_category][model_name]['params']
    elif model_type == 'unsupervised':
        if model_name not in UNSUPERVISED_MODELS:
            raise HTTPException(status_code=404, detail="Model not found")
        return UNSUPERVISED_MODELS[model_name]['params']
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")

@api_router.post("/dataset/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    try:
        dataset_id = str(uuid.uuid4())
        file_ext = file.filename.split('.')[-1].lower()
        
        content = await file.read()
        
        # Read file based on extension
        if file_ext == 'csv':
            df = pd.read_csv(io.BytesIO(content))
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel file.")
        
        # Save dataset
        file_path = UPLOADS_DIR / f"{dataset_id}.csv"
        df.to_csv(file_path, index=False)
        
        # Store metadata in DB
        dataset_doc = {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "file_path": str(file_path),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.datasets.insert_one(dataset_doc)
        
        # Return preview
        preview = df.head(5).to_dict('records')
        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "preview": preview
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/dataset/kaggle")
async def fetch_kaggle_dataset(dataset_name: str, kaggle_username: str, kaggle_key: str):
    kaggle_config = None
    try:
        # Import kaggle only when needed
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Create Kaggle config directory
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_dir.mkdir(exist_ok=True)
        
        # Write kaggle.json credentials
        kaggle_config = kaggle_dir / 'kaggle.json'
        with open(kaggle_config, 'w') as f:
            import json
            json.dump({
                'username': kaggle_username,
                'key': kaggle_key
            }, f)
        
        # Set proper permissions (required by Kaggle API)
        kaggle_config.chmod(0o600)
        
        # Initialize Kaggle API
        api = KaggleApi()
        
        # Try to authenticate - kaggle might call sys.exit on error
        try:
            api.authenticate()
        except SystemExit:
            # Clean up and raise proper error
            if kaggle_config and kaggle_config.exists():
                kaggle_config.unlink()
            raise HTTPException(
                status_code=401, 
                detail="Invalid Kaggle credentials. Please verify your username and API key are correct."
            )
        
        dataset_id = str(uuid.uuid4())
        download_path = UPLOADS_DIR / dataset_id
        download_path.mkdir(exist_ok=True)
        
        # Download dataset
        api.dataset_download_files(dataset_name, path=str(download_path), unzip=True)
        
        # Find CSV files
        csv_files = list(download_path.glob('*.csv'))
        if not csv_files:
            raise HTTPException(status_code=400, detail="No CSV files found in the dataset. Please ensure the dataset contains CSV files.")
        
        # Read first CSV
        df = pd.read_csv(csv_files[0])
        
        # Save as single CSV
        file_path = UPLOADS_DIR / f"{dataset_id}.csv"
        df.to_csv(file_path, index=False)
        
        # Store metadata
        dataset_doc = {
            "dataset_id": dataset_id,
            "filename": dataset_name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "file_path": str(file_path),
            "source": "kaggle",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.datasets.insert_one(dataset_doc)
        
        preview = df.head(5).to_dict('records')
        
        # Clean up kaggle config
        if kaggle_config and kaggle_config.exists():
            kaggle_config.unlink()
        
        return {
            "dataset_id": dataset_id,
            "filename": dataset_name,
            "rows": len(df),
            "columns": len(df.columns),
            "preview": preview
        }
    except HTTPException:
        raise
    except Exception as e:
        # Clean up kaggle config on error
        if kaggle_config and kaggle_config.exists():
            kaggle_config.unlink()
        
        error_msg = str(e)
        if '401' in error_msg or 'Unauthorized' in error_msg:
            raise HTTPException(status_code=401, detail="Invalid Kaggle credentials. Please check your username and API key.")
        elif '403' in error_msg or 'Forbidden' in error_msg:
            raise HTTPException(status_code=403, detail="Access denied. Please ensure you have accepted the dataset's terms on Kaggle website.")
        elif '404' in error_msg or 'not found' in error_msg.lower():
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found. Please check the dataset name format (e.g., 'username/dataset-name').")
        else:
            raise HTTPException(status_code=500, detail=f"Kaggle dataset fetch failed: {error_msg}")

@api_router.get("/dataset/{dataset_id}/columns")
async def get_dataset_columns(dataset_id: str):
    dataset = await db.datasets.find_one({"dataset_id": dataset_id}, {"_id": 0})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"columns": dataset["column_names"]}

def clean_data(df: pd.DataFrame, target_column: Optional[str] = None):
    cleaning_steps = []
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    cleaning_steps.append(f"Removed {duplicates_removed} duplicate rows")
    
    # Handle missing values
    missing_info = {}
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_info[col] = missing_count
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    if missing_info:
        cleaning_steps.append(f"Handled missing values in {len(missing_info)} columns")
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_column and target_column in categorical_cols:
        categorical_cols.remove(target_column)
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    if categorical_cols:
        cleaning_steps.append(f"Encoded {len(categorical_cols)} categorical columns")
    
    # Handle target column encoding if categorical
    target_encoder = None
    if target_column and target_column in df.columns:
        if df[target_column].dtype == 'object':
            target_encoder = LabelEncoder()
            df[target_column] = target_encoder.fit_transform(df[target_column])
            cleaning_steps.append(f"Encoded target column '{target_column}'")
    
    return df, cleaning_steps, label_encoders, target_encoder

@api_router.post("/dataset/clean")
async def clean_dataset(request: DataCleanRequest):
    try:
        dataset = await db.datasets.find_one({"dataset_id": request.dataset_id}, {"_id": 0})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        file_path = Path(dataset["file_path"])
        df = pd.read_csv(file_path)
        
        df_cleaned, steps, encoders, target_encoder = clean_data(df, request.target_column)
        
        # Save cleaned data
        cleaned_path = UPLOADS_DIR / f"{request.dataset_id}_cleaned.csv"
        df_cleaned.to_csv(cleaned_path, index=False)
        
        # Update dataset metadata
        await db.datasets.update_one(
            {"dataset_id": request.dataset_id},
            {"$set": {
                "cleaned_path": str(cleaned_path),
                "cleaning_steps": steps,
                "cleaned_at": datetime.now(timezone.utc).isoformat()
            }}
        )
        
        return {
            "dataset_id": request.dataset_id,
            "status": "cleaned",
            "steps": steps,
            "rows_after_cleaning": len(df_cleaned)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def train_model_task(job_id: str, dataset_id: str, model_type: str, model_category: Optional[str],
                          model_name: str, parameters: Dict, target_column: Optional[str]):
    try:
        training_jobs[job_id] = {"status": "running", "progress": 10, "message": "Loading dataset..."}
        
        # Load dataset
        dataset = await db.datasets.find_one({"dataset_id": dataset_id}, {"_id": 0})
        if not dataset:
            training_jobs[job_id] = {"status": "failed", "progress": 0, "message": "Dataset not found"}
            return
        
        file_path = dataset.get("cleaned_path") or dataset["file_path"]
        df = pd.read_csv(file_path)
        
        training_jobs[job_id] = {"status": "running", "progress": 30, "message": "Preparing data..."}
        
        # Prepare data based on model type
        if model_type == 'supervised':
            if not target_column or target_column not in df.columns:
                training_jobs[job_id] = {"status": "failed", "progress": 0, "message": "Invalid target column"}
                return
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Check if X has data
            if X.empty or len(X.columns) == 0:
                training_jobs[job_id] = {"status": "failed", "progress": 0, "message": "No feature columns available for training"}
                return
            
            # Ensure all columns are numeric
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            # Convert to numpy array and ensure it's 2D
            X = X.values
            y = y.values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            training_jobs[job_id] = {"status": "running", "progress": 50, "message": "Training model..."}
            
            # Get model class
            model_config = SUPERVISED_MODELS[model_category][model_name]
            model_class = model_config['class']
            
            # Create model with parameters
            model_params = {}
            for param, value in parameters.items():
                if value is not None and value != '':
                    model_params[param] = value
            
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            training_jobs[job_id] = {"status": "running", "progress": 70, "message": "Evaluating model..."}
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Generate metrics and visualizations
            results = {}
            visualizations = {}
            
            if model_category == 'classification':
                results['accuracy'] = float(accuracy_score(y_test, y_pred))
                results['precision'] = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                results['recall'] = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                results['f1_score'] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                cm_path = VIZ_DIR / f"{job_id}_confusion_matrix.png"
                plt.savefig(cm_path)
                plt.close()
                
                with open(cm_path, 'rb') as img_file:
                    visualizations['confusion_matrix'] = base64.b64encode(img_file.read()).decode('utf-8')
                
                # ROC Curve (for binary classification)
                if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve')
                    plt.legend(loc="lower right")
                    roc_path = VIZ_DIR / f"{job_id}_roc_curve.png"
                    plt.savefig(roc_path)
                    plt.close()
                    
                    with open(roc_path, 'rb') as img_file:
                        visualizations['roc_curve'] = base64.b64encode(img_file.read()).decode('utf-8')
                    results['roc_auc'] = float(roc_auc)
            
            elif model_category == 'regression':
                results['mse'] = float(mean_squared_error(y_test, y_pred))
                results['rmse'] = float(np.sqrt(results['mse']))
                results['mae'] = float(mean_absolute_error(y_test, y_pred))
                results['r2_score'] = float(r2_score(y_test, y_pred))
                
                # Scatter plot
                plt.figure(figsize=(8, 6))
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                plt.title('Actual vs Predicted')
                scatter_path = VIZ_DIR / f"{job_id}_scatter.png"
                plt.savefig(scatter_path)
                plt.close()
                
                with open(scatter_path, 'rb') as img_file:
                    visualizations['scatter_plot'] = base64.b64encode(img_file.read()).decode('utf-8')
                
                # Residual plot
                residuals = y_test - y_pred
                plt.figure(figsize=(8, 6))
                plt.scatter(y_pred, residuals, alpha=0.5)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xlabel('Predicted')
                plt.ylabel('Residuals')
                plt.title('Residual Plot')
                residual_path = VIZ_DIR / f"{job_id}_residuals.png"
                plt.savefig(residual_path)
                plt.close()
                
                with open(residual_path, 'rb') as img_file:
                    visualizations['residual_plot'] = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Save model
            model_path = MODELS_DIR / f"{job_id}.pkl"
            joblib.dump({'model': model, 'scaler': scaler}, model_path)
            
        else:  # Unsupervised
            # Ensure all columns are numeric
            for col in df.columns:
                if df[col].dtype == 'object':
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
            
            X = df.values
            
            if X.size == 0:
                training_jobs[job_id] = {"status": "failed", "progress": 0, "message": "No data available for training"}
                return
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            training_jobs[job_id] = {"status": "running", "progress": 50, "message": "Training model..."}
            
            model_config = UNSUPERVISED_MODELS[model_name]
            model_class = model_config['class']
            
            model_params = {}
            for param, value in parameters.items():
                if value is not None and value != '':
                    model_params[param] = value
            
            model = model_class(**model_params)
            
            if model_name == 'PCA':
                X_transformed = model.fit_transform(X_scaled)
                results = {'explained_variance_ratio': model.explained_variance_ratio_.tolist()}
                
                # PCA scatter plot
                plt.figure(figsize=(8, 6))
                plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.5)
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.title('PCA Visualization')
                pca_path = VIZ_DIR / f"{job_id}_pca.png"
                plt.savefig(pca_path)
                plt.close()
                
                with open(pca_path, 'rb') as img_file:
                    visualizations['pca_plot'] = base64.b64encode(img_file.read()).decode('utf-8')
            else:
                labels = model.fit_predict(X_scaled)
                
                # Silhouette score
                if len(np.unique(labels)) > 1:
                    sil_score = silhouette_score(X_scaled, labels)
                    results['silhouette_score'] = float(sil_score)
                
                results['n_clusters'] = int(len(np.unique(labels)))
                
                # Cluster visualization (2D PCA)
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                plt.figure(figsize=(8, 6))
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
                plt.colorbar(scatter)
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.title('Cluster Visualization')
                cluster_path = VIZ_DIR / f"{job_id}_clusters.png"
                plt.savefig(cluster_path)
                plt.close()
                
                with open(cluster_path, 'rb') as img_file:
                    visualizations['cluster_plot'] = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Save model
            model_path = MODELS_DIR / f"{job_id}.pkl"
            joblib.dump({'model': model, 'scaler': scaler}, model_path)
        
        training_jobs[job_id] = {"status": "running", "progress": 90, "message": "Saving results..."}
        
        # Save to database
        job_doc = {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "model_type": model_type,
            "model_category": model_category,
            "model_name": model_name,
            "parameters": parameters,
            "results": results,
            "model_path": str(model_path),
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        await db.training_jobs.insert_one(job_doc)
        
        training_jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Training completed successfully",
            "result": {
                "metrics": results,
                "visualizations": visualizations,
                "model_path": str(model_path)
            }
        }
    except Exception as e:
        training_jobs[job_id] = {"status": "failed", "progress": 0, "message": str(e)}

@api_router.post("/model/train")
async def train_model(request: TrainModelRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    training_jobs[job_id] = {"status": "initializing", "progress": 0, "message": "Starting training..."}
    
    background_tasks.add_task(
        train_model_task,
        job_id,
        request.dataset_id,
        request.model_type,
        request.model_category,
        request.model_name,
        request.parameters,
        request.target_column
    )
    
    return {"job_id": job_id}

@api_router.get("/model/progress/{job_id}", response_model=TrainingProgressResponse)
async def get_training_progress(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = training_jobs[job_id]
    return {
        "job_id": job_id,
        "status": job_data["status"],
        "progress": job_data["progress"],
        "message": job_data["message"],
        "result": job_data.get("result")
    }

@api_router.get("/model/download/{job_id}")
async def download_model(job_id: str):
    job = await db.training_jobs.find_one({"job_id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    model_path = Path(job["model_path"])
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(
        path=model_path,
        filename=f"model_{job_id}.pkl",
        media_type="application/octet-stream"
    )

async def compare_models_task(job_id: str, dataset_id: str, model_category: str,
                              model1_name: str, model2_name: str,
                              model1_params: Dict, model2_params: Dict, target_column: str):
    try:
        training_jobs[job_id] = {"status": "running", "progress": 10, "message": "Loading dataset..."}
        
        # Load dataset
        dataset = await db.datasets.find_one({"dataset_id": dataset_id}, {"_id": 0})
        if not dataset:
            training_jobs[job_id] = {"status": "failed", "progress": 0, "message": "Dataset not found"}
            return
        
        file_path = dataset.get("cleaned_path") or dataset["file_path"]
        df = pd.read_csv(file_path)
        
        training_jobs[job_id] = {"status": "running", "progress": 20, "message": "Preparing data..."}
        
        if target_column not in df.columns:
            training_jobs[job_id] = {"status": "failed", "progress": 0, "message": "Invalid target column"}
            return
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Ensure all columns are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        X = X.values
        y = y.values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train Model 1
        training_jobs[job_id] = {"status": "running", "progress": 40, "message": f"Training {model1_name}..."}
        
        model1_config = SUPERVISED_MODELS[model_category][model1_name]
        model1_class = model1_config['class']
        model1 = model1_class(**{k: v for k, v in model1_params.items() if v is not None and v != ''})
        model1.fit(X_train, y_train)
        y_pred1 = model1.predict(X_test)
        
        # Get Model 1 metrics
        model1_metrics = {}
        if model_category == 'classification':
            model1_metrics['accuracy'] = float(accuracy_score(y_test, y_pred1))
            model1_metrics['precision'] = float(precision_score(y_test, y_pred1, average='weighted', zero_division=0))
            model1_metrics['recall'] = float(recall_score(y_test, y_pred1, average='weighted', zero_division=0))
            model1_metrics['f1_score'] = float(f1_score(y_test, y_pred1, average='weighted', zero_division=0))
        else:  # regression
            model1_metrics['mse'] = float(mean_squared_error(y_test, y_pred1))
            model1_metrics['rmse'] = float(np.sqrt(model1_metrics['mse']))
            model1_metrics['mae'] = float(mean_absolute_error(y_test, y_pred1))
            model1_metrics['r2_score'] = float(r2_score(y_test, y_pred1))
        
        # Train Model 2
        training_jobs[job_id] = {"status": "running", "progress": 70, "message": f"Training {model2_name}..."}
        
        model2_config = SUPERVISED_MODELS[model_category][model2_name]
        model2_class = model2_config['class']
        model2 = model2_class(**{k: v for k, v in model2_params.items() if v is not None and v != ''})
        model2.fit(X_train, y_train)
        y_pred2 = model2.predict(X_test)
        
        # Get Model 2 metrics
        model2_metrics = {}
        if model_category == 'classification':
            model2_metrics['accuracy'] = float(accuracy_score(y_test, y_pred2))
            model2_metrics['precision'] = float(precision_score(y_test, y_pred2, average='weighted', zero_division=0))
            model2_metrics['recall'] = float(recall_score(y_test, y_pred2, average='weighted', zero_division=0))
            model2_metrics['f1_score'] = float(f1_score(y_test, y_pred2, average='weighted', zero_division=0))
        else:  # regression
            model2_metrics['mse'] = float(mean_squared_error(y_test, y_pred2))
            model2_metrics['rmse'] = float(np.sqrt(model2_metrics['mse']))
            model2_metrics['mae'] = float(mean_absolute_error(y_test, y_pred2))
            model2_metrics['r2_score'] = float(r2_score(y_test, y_pred2))
        
        # Determine winner
        training_jobs[job_id] = {"status": "running", "progress": 90, "message": "Comparing results..."}
        
        if model_category == 'classification':
            # Winner has higher F1 score
            winner = model1_name if model1_metrics['f1_score'] > model2_metrics['f1_score'] else model2_name
        else:
            # Winner has lower RMSE
            winner = model1_name if model1_metrics['rmse'] < model2_metrics['rmse'] else model2_name
        
        # Save results
        comparison_result = {
            "model1_name": model1_name,
            "model2_name": model2_name,
            "model1_metrics": model1_metrics,
            "model2_metrics": model2_metrics,
            "winner": winner
        }
        
        comparison_doc = {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "model_category": model_category,
            "comparison": comparison_result,
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        await db.training_jobs.insert_one(comparison_doc)
        
        training_jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Comparison completed successfully",
            "result": {
                "comparison": comparison_result
            }
        }
    except Exception as e:
        training_jobs[job_id] = {"status": "failed", "progress": 0, "message": str(e)}

@api_router.post("/model/compare")
async def compare_models(request: CompareModelsRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    training_jobs[job_id] = {"status": "initializing", "progress": 0, "message": "Starting comparison..."}
    
    background_tasks.add_task(
        compare_models_task,
        job_id,
        request.dataset_id,
        request.model_category,
        request.model1_name,
        request.model2_name,
        request.model1_parameters,
        request.model2_parameters,
        request.target_column
    )
    
    return {"job_id": job_id}

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()