create-venv:
	python3 -m venv venv

activate-venv-unix:
	source venv/bin/activate

activate-venv-windows:
	.\venv\bin\activate

install-deps:
	pip install -r requirements.txt

run-comparator:
	python3 main.py

.DEFAULT_GOAL := run-comparator