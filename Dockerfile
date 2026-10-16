# Reproduces the results in a clean container. Not built or tested yet (Docker was not available when this was written).
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "src/baselines_and_checks.py"]
