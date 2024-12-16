FROM python:3.9-slim

WORKDIR /app
COPY data_uploader.py /app

RUN pip install --no-cache-dir requests 

CMD ["python3", "data_uploader.py"]
