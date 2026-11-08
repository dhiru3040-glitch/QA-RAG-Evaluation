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
                    python -m pip install deepeval==4.3.3
                    python -m compileall -q rag backend evaluation
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

        stage('Verify Ollama availability') {
            steps {
                bat '''
                    call .venv-ci\\Scripts\\activate.bat
                    curl.exe --fail --silent --show-error http://localhost:11434/api/tags > ollama-tags.json
                    if errorlevel 1 (
                        echo ERROR: Ollama service is unavailable at http://localhost:11434.
                        exit /b 1
                    )
                    python -c "import json; data=json.load(open('ollama-tags.json','r',encoding='utf-8')); models={m.get('name','') for m in data.get('models',[])}; required='llama3.2:3b'; raise SystemExit(0 if required in models else 1)"
                    if errorlevel 1 (
                        echo ERROR: Ollama model llama3.2:3b is unavailable.
                        exit /b 1
                    )
                    echo Ollama service is accessible and model llama3.2:3b is available.
                '''
            }
        }

        stage('Prepare knowledge base') {
            steps {
                bat '''
                    call .venv-ci\\Scripts\\activate.bat
                    python -c "import chromadb; c=chromadb.PersistentClient(path='vector_db'); import sys; exec(\"try:\\n col=c.get_collection('qa_knowledge_base')\\n raise SystemExit(0 if col.count()>0 else 2)\\nexcept Exception:\\n raise SystemExit(2)\")"
                    if errorlevel 2 (
                        python -m rag.vector_store
                        if errorlevel 1 exit /b 1
                    )
                    if errorlevel 1 (
                        echo ERROR: Unable to inspect the existing ChromaDB collection.
                        exit /b 1
                    )
                    echo Knowledge base collection is available for RAG evaluation.
                '''
            }
        }

        stage('Run Golden Dataset DeepEval evaluation') {
            steps {
                bat '''
                    call .venv-ci\\Scripts\\activate.bat
                    python -c "import json; cases=json.load(open('evaluation/datasets/golden_dataset.json','r',encoding='utf-8')); raise SystemExit(0 if len(cases)==10 else 1)"
                    if errorlevel 1 (
                        echo ERROR: The Golden Dataset must contain exactly 10 test cases.
                        exit /b 1
                    )
                    python -m evaluation.evaluate_rag
                    if errorlevel 1 exit /b 1

                    python -c "import json; p='evaluation/reports/evaluation_results.json'; r=json.load(open(p,'r',encoding='utf-8')); results=r.get('results',[]); completed=sum(1 for x in results if x.get('metrics')); failed=sum(1 for x in results if x.get('metrics') and not all(m.get('passed') for m in x['metrics'].values())); quality_errors=sum(1 for x in results if x.get('error')); print('Golden Dataset report: %s total, %s completed, %s quality failures, %s execution errors' % (len(results), completed, failed, quality_errors))"
                '''
                echo 'All 10 Golden Dataset cases were submitted through the existing DeepEval evaluation script.'
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
            archiveArtifacts artifacts: 'evaluation/reports/**', allowEmptyArchive: true, fingerprint: true
            echo 'Jenkins CI pipeline execution finished.'
        }
    }
}
