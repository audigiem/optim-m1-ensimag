all: run

run:
	jupyter-lab

test:
	python -m pytest -v tests/

correction:
	sh correction.sh

student:
	sh student.sh

setup:
	python -m venv .venv_tp2
	.venv_tp2/bin/pip3 install numpy scipy pytest matplotlib notebook
	.venv_tp2/bin/jupyter notebook
