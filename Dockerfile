FROM python:3.9-slim

WORKDIR /workspace

COPY requirements.txt requirements.txt

RUN pip3 --no-cache-dir install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 --no-cache-dir install numpy==1.23.4

COPY train.py /workspace/ 

CMD ["python", "train.py"]
