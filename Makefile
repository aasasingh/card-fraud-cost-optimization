# Card fraud detection pipeline.
# Targets are added stage by stage; `run` appears once there is a pipeline to run.

# Exported so every src/ script reads the same connection string from the
# environment; override with `make load DB_URL=...` if your Postgres differs.
export DB_URL ?= postgresql://fraud:fraud@localhost:5432/fraud

# GNU make sets OS=Windows_NT on Windows; venv layouts differ (Scripts vs bin).
ifeq ($(OS),Windows_NT)
PY = python
VENV_PY = .venv/Scripts/python
else
PY = python3
VENV_PY = .venv/bin/python
endif

.PHONY: setup db-up db-down db-shell load migrate profile baseline model run

setup: ## Create venv, install deps, start Postgres, wait until it accepts connections
	$(PY) -m venv .venv
	$(VENV_PY) -m pip install -r requirements.txt
	docker compose up -d --wait

db-up:
	docker compose up -d --wait

db-down:
	docker compose down

db-shell: ## psql inside the container
	docker compose exec db psql -U fraud -d fraud

load: ## Stage 1: COPY raw CSVs into text staging tables
	$(VENV_PY) src/load.py

migrate: ## Apply sql/ migrations in order (typed tables, features)
	$(VENV_PY) src/migrate.py

profile: ## Stage 1 report: volume, nulls, join coverage, pseudo-card validation
	$(VENV_PY) src/profile.py

baseline: ## Stage 3: rules baseline scored on the cost model
	$(VENV_PY) src/baseline.py

model: ## Stage 4: LightGBM, operating point selected on validation
	$(VENV_PY) src/model.py

run: load migrate profile baseline model ## Full pipeline: every README number
