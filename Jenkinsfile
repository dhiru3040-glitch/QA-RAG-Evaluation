pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    stages {
        stage('Checkout source code') {
            steps {
                checkout scm
                echo 'Source code checkout completed from the configured GitHub SCM.'
            }
        }

        stage('Python environment setup and backend validation') {
            steps {
                bat '''
                    python -m venv .venv-ci
                    call .venv-ci\\Scripts\\activate.bat
                    python -m pip install --upgrade pip
                    python -m pip install -r requirements.txt
                    python -m compileall -q rag backend
                '''
                echo 'Python dependencies installed and backend Python source compiled successfully.'
            }
        }

        stage('Frontend dependency installation') {
            steps {
                dir('frontend') {
                    bat '''
                        npm ci
                    '''
                }
                echo 'Frontend dependencies installed with package-lock.json.'
            }
        }

        stage('React production build') {
            steps {
                dir('frontend') {
                    bat '''
                        npm run build
                    '''
                }
                echo 'React production build completed successfully.'
            }
        }
    }

    post {
        success {
            echo 'CI pipeline completed successfully: Python validation and React production build passed.'
        }
        failure {
            echo 'CI pipeline failed. Review the Jenkins console output for the failed stage.'
        }
        always {
            echo 'Jenkins CI pipeline execution finished.'
        }
    }
}
