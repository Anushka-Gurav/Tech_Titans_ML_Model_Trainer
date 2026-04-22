pipeline {
    agent any

    stages {

        stage('Build Backend') {
            steps {
                dir('backend') {
                    bat '"C:\\Users\\Sana Khan\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" -m pip install -r requirements.txt'
                }
            }
        }

        stage('Test Backend') {
            steps {
                dir('backend') {
                    bat '"C:\\Users\\Sana Khan\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" -m pytest'
                }
            }
        }

        stage('Build Frontend') {
            steps {
                dir('frontend') {
                    bat 'npm install'
                    bat 'npm run build'
                }
            }
        }
    }
}
