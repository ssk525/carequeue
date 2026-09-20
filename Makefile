.PHONY: install test train api dashboard monitor docker-build docker-run

install:
	python -m pip install -r requirements.txt

test:
	pytest -q

train:
	python -m carequeue.train --download

api:
	uvicorn carequeue.api:app --host 127.0.0.1 --port 8000

dashboard:
	streamlit run carequeue/dashboard.py

monitor:
	python -m carequeue.monitor --input data/demo_batch.csv

docker-build:
	docker build -t carequeue .

docker-run:
	docker run --rm -p 127.0.0.1:8000:8000 -v "$(PWD)/artifacts:/app/artifacts:ro" carequeue
