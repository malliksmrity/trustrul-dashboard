# Reproduces the results in a clean container. Built and run on 17 Oct 2026: output matches reports/results.json.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "src/baselines_and_checks.py"]
