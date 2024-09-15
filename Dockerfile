FROM python:3.9-slim

WORKDIR /workspace

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY train.py /workspace/ 

CMD ["python", "train.py"]