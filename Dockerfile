FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt setup.py ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Render requires this port
EXPOSE 10000

CMD ["python", "app.py"]
