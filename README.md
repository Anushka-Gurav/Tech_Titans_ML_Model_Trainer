# Smart ML Model Trainer - ML Training Platform

A full-stack machine learning training and comparison platform with a futuristic terminal-inspired interface. Train, evaluate, and compare multiple ML models with an intuitive dashboard.
---
🔹 Live Project : https://ml-model-trainer.netlify.app/

🔹 Blog : (https://medium.com/@shraddhamane829/building-a-smart-ml-model-trainer-our-journey-from-idea-to-implementation-55328bc6b61e)

🔹 Demo Link : https://youtu.be/Nkvgav4WiUg?si=ZrbCezUzM3QpSxPi

🔹 Document Link : https://www.notion.so/Smart-ML-Model-Trainer-2d92368f3571805ba7f2cb12c380c05b?source=copy_link

🔹 LinkedIn Post : https://hosturl.link/B0QujU

<img width="1898" height="904" alt="image" src="https://github.com/user-attachments/assets/ce3fe705-36d0-429b-a554-5ecfa86d1e96" />

---

## IDE Used : KIRO (Best IDE for Project Developement) (https://kiro.dev/downloads/)
Usage : 
- Logged into Kiro IDE using AWS Builder ID for secure and seamless access.
- Enabled Spec + Vibe Autopilot to convert the idea into a structured specification.
- Used the generated Spec to automatically create:
   1) Requirements document
   2) Design document
   3) task.md for execution planning
- Developed the ML Model Trainer following the Kiro-generated task flow.
- Implemented an Agent Hook to automatically log system actions and decision history during dataset upload, model training, and evaluation.
- Used Kiro Testing (Pytest) to generate and run automated end-to-end tests for the complete pipeline.
- Deployed the frontend using Kiro.dev with Netlify integration directly from the Kiro environment.
- Completed the entire ideation-to-deployment lifecycle within Kiro IDE, without switching tools.
## Features

### Core Capabilities
- **Dataset Management**: Upload CSV/Excel files or fetch datasets directly from Kaggle
- **Automated Data Cleaning**: Handle missing values, remove duplicates, and encode categorical variables
- **Supervised Learning**: Classification and regression models with comprehensive metrics
- **Unsupervised Learning**: Clustering and dimensionality reduction algorithms
- **Model Comparison**: Side-by-side comparison of two models with visual analytics
- **Real-time Training Progress**: Live progress tracking with status updates
- **Model Export**: Download trained models for deployment


## Project Structure

```
CompareModel-main/
├── backend/
│   ├── server.py              # FastAPI application
│   ├── requirements.txt       # Python dependencies
│   ├── .env                   # Environment variables
│   ├── models/                # Saved model files (.pkl)
│   ├── uploads/               # Uploaded datasets
│   └── visualizations/        # Generated charts
├── frontend/
│   ├── src/
│   │   ├── pages/            # React pages
│   │   ├── components/       # Reusable components
│   │   ├── lib/              # Utilities
│   │   └── App.js            # Main app component
│   ├── package.json          # Node dependencies
│   └── .env                  # Environment variables
├── design_guidelines.json    # UI/UX design system
└── README.md                 # This file
```
## Tech Stack

### Backend
- **Framework**: FastAPI
- **Database**: MongoDB (via Motor async driver)
- **ML Libraries**: scikit-learn, XGBoost, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Data Processing**: pandas, numpy

### Frontend
- **Framework**: React 18
- **UI Components**: Radix UI, shadcn/ui
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **HTTP Client**: Axios
- **Routing**: React Router v7

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd CompareModel-main
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
# Create a .env file with:
MONGO_URL=mongodb://localhost:27017
DB_NAME=ml_platform
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
yarn install

# Configure environment variables
# Create a .env file with:
REACT_APP_API_URL=http://localhost:8000
```

## Running the Application

### Start Backend Server
```bash
cd backend
uvicorn server:app --reload --port 8000
```

The API will be available at `http://localhost:8000`
API documentation: `http://localhost:8000/docs`

### Start Frontend Development Server
```bash
cd frontend
yarn start
```


Built with ❤️ using FastAPI and React
