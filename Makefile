# Optimized Makefile for LegalQA Performance Project
# Provides enhanced commands for performance monitoring and optimization

# Configuration
APP_SERVICE_NAME = app
PYTHON := python3
DOCKER_COMPOSE_PROD := docker-compose -f docker-compose.yml
DOCKER_COMPOSE_DEV := docker-compose -f docker-compose.yml -f docker-compose.dev.yml
MONITORING := docker-compose -f docker-compose.yml --profile monitoring

.DEFAULT_GOAL := help

# Colors for output
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
BLUE=\033[0;34m
NC=\033[0m # No Color

help:
	@echo "${BLUE}=============================================================================${NC}"
	@echo "${GREEN} LegalQA Optimized Performance Makefile ${NC}"
	@echo "${BLUE}=============================================================================${NC}"
	@echo ""
	@echo "${YELLOW}🚀 PRODUCTION COMMANDS:${NC}"
	@echo "  ${GREEN}prod-up${NC}            Start optimized production environment"
	@echo "  ${GREEN}prod-build${NC}         Build optimized production images"
	@echo "  ${GREEN}prod-setup${NC}         Complete production setup with database"
	@echo "  ${GREEN}prod-down${NC}          Stop production environment"
	@echo ""
	@echo "${YELLOW}⚡ DEVELOPMENT COMMANDS:${NC}"
	@echo "  ${GREEN}dev-up${NC}             Start development environment with hot reload"
	@echo "  ${GREEN}dev-setup${NC}          Setup development environment with sample data"
	@echo "  ${GREEN}dev-down${NC}           Stop development environment"
	@echo ""
	@echo "${YELLOW}📊 MONITORING COMMANDS:${NC}"
	@echo "  ${GREEN}monitoring-up${NC}      Start monitoring stack (Prometheus + Grafana)"
	@echo "  ${GREEN}monitoring-down${NC}    Stop monitoring stack"
	@echo "  ${GREEN}metrics${NC}            Show current application metrics"
	@echo "  ${GREEN}stats${NC}              Show performance statistics"
	@echo ""
	@echo "${YELLOW}🔧 OPTIMIZATION COMMANDS:${NC}"
	@echo "  ${GREEN}optimize-db${NC}        Optimize database performance"
	@echo "  ${GREEN}clear-cache${NC}        Clear all application caches"
	@echo "  ${GREEN}benchmark${NC}          Run performance benchmarks"
	@echo "  ${GREEN}profile${NC}            Run performance profiling"
	@echo ""
	@echo "${YELLOW}🛠️ UTILITY COMMANDS:${NC}"
	@echo "  ${GREEN}logs${NC}               Show application logs"
	@echo "  ${GREEN}logs-follow${NC}        Follow application logs"
	@echo "  ${GREEN}health${NC}             Check application health"
	@echo "  ${GREEN}clean${NC}              Clean up resources and volumes"
	@echo ""
	@echo "${YELLOW}🧪 TESTING COMMANDS:${NC}"
	@echo "  ${GREEN}validate${NC}           Quick validation of optimizations"
	@echo "  ${GREEN}test${NC}               Run comprehensive test suite"
	@echo "  ${GREEN}test-functionality${NC} Run functionality tests only"
	@echo "  ${GREEN}test-performance${NC}   Run performance tests only"
	@echo "  ${GREEN}test-integration${NC}   Run integration tests only"
	@echo "  ${GREEN}test-ci${NC}            Run CI/CD test pipeline"
	@echo ""
	@echo "${BLUE}=============================================================================${NC}"

.PHONY: help prod-up prod-build prod-setup prod-down dev-up dev-setup dev-down monitoring-up monitoring-down

# ==============================================================================
# PRODUCTION COMMANDS
# ==============================================================================

prod-build:
	@echo "${GREEN}🔨 Building optimized production images...${NC}"
	$(DOCKER_COMPOSE_PROD) build --no-cache --parallel

prod-up:
	@echo "${GREEN}🚀 Starting optimized production environment...${NC}"
	$(DOCKER_COMPOSE_PROD) up -d
	@echo "${GREEN}✅ Production environment started${NC}"
	@echo "${BLUE}API available at: http://localhost:8000${NC}"
	@echo "${BLUE}API docs at: http://localhost:8000/docs${NC}"

prod-setup: prod-build
	@echo "${GREEN}⚙️ Setting up production environment...${NC}"
	$(DOCKER_COMPOSE_PROD) up -d db redis
	@echo "${YELLOW}Waiting for database to be ready...${NC}"
	@sleep 10
	@echo "${GREEN}🗄️ Building production database...${NC}"
	$(DOCKER_COMPOSE_PROD) run --rm $(APP_SERVICE_NAME) python scripts/build_database.py
	@echo "${GREEN}🚀 Starting full production environment...${NC}"
	$(DOCKER_COMPOSE_PROD) up -d
	@echo "${GREEN}✅ Production setup completed${NC}"

prod-down:
	@echo "${YELLOW}🛑 Stopping production environment...${NC}"
	$(DOCKER_COMPOSE_PROD) down

# ==============================================================================
# DEVELOPMENT COMMANDS
# ==============================================================================

dev-up:
	@echo "${GREEN}⚡ Starting optimized development environment...${NC}"
	$(DOCKER_COMPOSE_DEV) up -d --build
	@echo "${GREEN}✅ Development environment started with hot reload${NC}"
	@echo "${BLUE}API available at: http://localhost:8000${NC}"
	@echo "${BLUE}API docs at: http://localhost:8000/docs${NC}"

dev-setup:
	@echo "${GREEN}⚙️ Setting up development environment...${NC}"
	@echo "${GREEN}📝 Creating sample data...${NC}"
	$(PYTHON) scripts/create_sample.py
	@echo "${GREEN}🔨 Building local FAISS index...${NC}"
	$(PYTHON) scripts/build_local_index.py
	@echo "${GREEN}🚀 Starting development services...${NC}"
	$(DOCKER_COMPOSE_DEV) up -d --build
	@echo "${GREEN}✅ Development setup completed${NC}"

dev-down:
	@echo "${YELLOW}🛑 Stopping development environment...${NC}"
	$(DOCKER_COMPOSE_DEV) down

# ==============================================================================
# MONITORING COMMANDS
# ==============================================================================

monitoring-up:
	@echo "${GREEN}📊 Starting monitoring stack...${NC}"
	$(MONITORING) up -d
	@echo "${GREEN}✅ Monitoring started${NC}"
	@echo "${BLUE}Prometheus: http://localhost:9090${NC}"
	@echo "${BLUE}Grafana: http://localhost:3000 (admin/admin)${NC}"

monitoring-down:
	@echo "${YELLOW}📊 Stopping monitoring stack...${NC}"
	$(MONITORING) down

metrics:
	@echo "${GREEN}📈 Current application metrics:${NC}"
	@curl -s http://localhost:8000/metrics | grep -E "^legalqa_" | head -20

stats:
	@echo "${GREEN}📊 Performance statistics:${NC}"
	@curl -s http://localhost:8000/stats | python3 -m json.tool

health:
	@echo "${GREEN}🏥 Application health check:${NC}"
	@curl -s http://localhost:8000/health | python3 -m json.tool

# ==============================================================================
# OPTIMIZATION COMMANDS
# ==============================================================================

optimize-db:
	@echo "${GREEN}🔧 Optimizing database performance...${NC}"
	$(DOCKER_COMPOSE_PROD) exec db psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "VACUUM ANALYZE;"
	$(DOCKER_COMPOSE_PROD) exec db psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "REINDEX DATABASE ${POSTGRES_DB};"
	@echo "${GREEN}✅ Database optimization completed${NC}"

clear-cache:
	@echo "${GREEN}🧹 Clearing application caches...${NC}"
	@curl -s -X POST http://localhost:8000/clear-cache | python3 -m json.tool

benchmark:
	@echo "${GREEN}🏃 Running performance benchmarks...${NC}"
	@echo "${BLUE}Testing API response times...${NC}"
	@time curl -s -X POST "http://localhost:8000/ask" \
		-H "Content-Type: application/json" \
		-d '{"question": "Mi a bűnszervezet fogalma a Btk. szerint?"}' \
		| python3 -m json.tool

profile:
	@echo "${GREEN}🔍 Running performance profiling...${NC}"
	$(DOCKER_COMPOSE_PROD) exec $(APP_SERVICE_NAME) python -m cProfile -s cumulative scripts/profile_app.py

# ==============================================================================
# UTILITY COMMANDS
# ==============================================================================

logs:
	@echo "${GREEN}📜 Showing application logs:${NC}"
	$(DOCKER_COMPOSE_PROD) logs --tail=100 $(APP_SERVICE_NAME)

logs-follow:
	@echo "${GREEN}📜 Following application logs (Ctrl+C to stop):${NC}"
	$(DOCKER_COMPOSE_PROD) logs -f $(APP_SERVICE_NAME)

clean:
	@echo "${YELLOW}🧹 Cleaning up resources...${NC}"
	$(DOCKER_COMPOSE_PROD) down --volumes --remove-orphans
	$(DOCKER_COMPOSE_DEV) down --volumes --remove-orphans
	$(MONITORING) down --volumes --remove-orphans
	@echo "${GREEN}✅ Cleanup completed${NC}"

clean-all: clean
	@echo "${YELLOW}🧹 Removing all images and containers...${NC}"
	docker system prune -a -f --volumes
	@echo "${GREEN}✅ Full cleanup completed${NC}"

# ==============================================================================
# TESTING COMMANDS
# ==============================================================================

validate:
	@echo "${GREEN}🔍 Running quick validation...${NC}"
	$(PYTHON) scripts/validate_optimizations.py

test: test-unit test-integration test-performance
test-unit:
	pytest tests/test_unit.py -v --cov=src --cov-report=html --cov-report=term-missing

test-integration:
	pytest tests/test_integration.py -v --cov=src --cov-report=html --cov-report=term-missing

test-performance:
	pytest tests/test_performance.py -v --cov=src --cov-report=html --cov-report=term-missing

test-metrics:
	python tests/monitoring/test_metrics_app.py &
	sleep 3
	k6 run tests/performance/smoke.js
	pkill -f test_metrics_app.py

test-rag-performance:
	python tests/monitoring/test_metrics_app.py &
	sleep 3
	k6 run tests/performance/rag_performance.js
	pkill -f test_metrics_app.py

test-all: test test-metrics test-rag-performance

test-functionality:
	@echo "${GREEN}🔧 Running functionality tests...${NC}"
	$(PYTHON) -m pytest tests/test_functionality.py -v

test-ci:
	@echo "${GREEN}🚀 Running CI test suite...${NC}"
	$(PYTHON) scripts/validate_optimizations.py
	@if [ $$? -eq 0 ]; then \
		echo "${GREEN}✅ Validation passed, running full tests...${NC}"; \
		$(PYTHON) scripts/run_tests.py; \
	else \
		echo "${RED}❌ Validation failed, skipping full tests${NC}"; \
		exit 1; \
	fi

lint:
	@echo "${GREEN}🔍 Running code linting...${NC}"
	$(PYTHON) -m black --check src/ tests/
	$(PYTHON) -m isort --check-only src/ tests/
	$(PYTHON) -m mypy src/

format:
	@echo "${GREEN}✨ Formatting code...${NC}"
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m isort src/ tests/

# ==============================================================================
# INSTALLATION COMMANDS
# ==============================================================================

install:
	@echo "${GREEN}📦 Installing dependencies...${NC}"
	$(PYTHON) -m pip install -e ".[dev]"
