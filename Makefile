# Makefile for the LegalQA Project
# Provides a clear and consistent set of commands for development and production tasks.

# --- Configuration ---
# Use a variable to define the main service for one-off commands
APP_SERVICE_NAME = app

# ====================================================================================
# HELP
# ====================================================================================
# By default, running "make" will show this help message.
.DEFAULT_GOAL := help

help:
	@echo "-------------------------------------------------------------------------"
	@echo " LegalQA Project Makefile"
	@echo "-------------------------------------------------------------------------"
	@echo "Usage: make <target>"
	@echo ""
	@echo "--- Core Commands ---"
	@echo "  up              Start all services in detached mode (using docker-compose up)."
	@echo "  down            Stop and remove all services, networks, and optionally volumes."
	@echo "  build-db        Build the production database and FAISS index from the full dataset."
	@echo ""
	@echo "--- Docker Compose Management ---"
	@echo "  build           Build or rebuild all service images."
	@echo "  logs            Tail logs from all running services."
	@echo "  ps              List running containers for this project."
	@echo ""
	@echo "--- Local Python Tasks (run on host) ---"
	@echo "  install         Install local Python dependencies from requirements.txt."
	@echo "  test            Run pytest for the test suite."
	@echo "  lint            Run flake8 for code linting."
	@echo "  clean           Remove temporary Python artifacts (__pycache__, etc.)."
	@echo "-------------------------------------------------------------------------"


# Phony targets are not associated with files. This prevents conflicts.
.PHONY: help install test lint clean build up down logs ps build-db

# ====================================================================================
# CORE WORKFLOW
# ====================================================================================

build-db:
	@echo "--> Building production database and FAISS index with FULL data..."
	@echo "--> NOTE: This command must be run while services are up ('make up')."
	@echo "--> WARNING: This can be a very long and resource-intensive process!"
	docker-compose exec $(APP_SERVICE_NAME) python scripts/build_database.py

# ====================================================================================
# DOCKER-COMPOSE WRAPPERS
# ====================================================================================
# These commands wrap the standard docker-compose commands for convenience.

build:
	@echo "--> Building or rebuilding service images..."
	docker-compose build

up:
	@echo "--> Starting all services in detached mode (app & db)..."
	docker-compose up -d

down:
	@echo "--> Stopping and removing services, containers, and networks..."
	@echo "--> To also remove the database volume, run: docker-compose down --volumes"
	docker-compose down

logs:
	@echo "--> Tailing logs for all services (Ctrl+C to stop)..."
	docker-compose logs -f

ps:
	@echo "--> Listing running containers for the project..."
	docker-compose ps


# ====================================================================================
# LOCAL PYTHON TASKS
# ====================================================================================
# These tasks are run on the host machine, not in Docker.

install:
	@echo "--> Installing or upgrading local dependencies from requirements.txt..."
	pip3 install --upgrade -r requirements.txt
	@echo "--> Installation/upgrade complete."

test:
	@echo "--> Running local tests..."
	python3 -m pytest test/

lint:
	@echo "--> Running local linter..."
	flake8 src/ test/

clean:
	@echo "--> Cleaning up local python artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "--> Cleanup complete."