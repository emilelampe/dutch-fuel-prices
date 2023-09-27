.PHONY: tests docs

dependencies: 
	@echo "Initializing Git..."
	git init
	@echo "Installing dependencies..."
	poetry install
	poetry env use /usr/bin/python3.11
	poetry run pre-commit install

env: dependencies
	@echo "Activating virtual environment..."
	poetry shell

tests:
	pytest

docs:
	@echo Save documentation to docs... 
	pdoc src -o docs --force
	@echo View API documentation... 
	pdoc src --http localhost:8080	

mlflow_nb:
	@echo "Activating mlflow ui with sqlite backend in notebooks folder..."
	cd notebooks && mlflow ui --backend-store-uri sqlite:///mydb.sqlite