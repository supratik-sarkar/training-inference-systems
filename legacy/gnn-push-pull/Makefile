
    .PHONY: init lint format test run
    init:
	pip install -r requirements.txt
	pre-commit install
    lint:
	flake8 src
    format:
	black src && isort src
    test:
	pytest -q
    run:
	python -m src.run
