# Convenience only. The source of truth for every check is pyproject.toml plus
# .github/workflows/ci.yml — CI does not run `make`, so these targets are kept to
# single obvious commands that mirror the CI steps exactly.

.PHONY: install test lint fmt typecheck controls check clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests
	ruff format --check src tests

fmt:
	ruff check --fix src tests
	ruff format src tests

typecheck:
	mypy src

controls:
	python -m foundationscale.gates.controls

check: lint typecheck test controls

clean:
	rm -rf build dist .eggs src/*.egg-info *.egg-info \
		.pytest_cache .mypy_cache .ruff_cache .cache \
		.coverage .coverage.* coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
