pipeline {
    agent any

    environment {
        MONGO_URL = "mongodb://localhost:27017"
        DB_NAME   = "test_db"
        PYTHON    = "C:\\Users\\Sana Khan\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"
    }

    stages {

        stage('Build Backend') {
            steps {
                dir('backend') {
                    bat "\"%PYTHON%\" -m pip install -r ..\\test-requirements.txt"
                }
            }
        }

        stage('Test Backend') {
            steps {
                dir('backend') {
                    bat "\"%PYTHON%\" -m pytest"
                }
            }
        }

        stage('Build Frontend') {
            when {
                expression { fileExists('frontend/package.json') }
            }
            steps {
                dir('frontend') {
                    bat 'npm install'
                    bat 'npm run build'
                }
            }
        }
    }

    post {
        success {
            echo 'CI Pipeline executed successfully ✅'
        }
        failure {
            echo 'CI Pipeline failed ❌'
        }
    }
}
