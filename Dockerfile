FROM python:3.11-slim

WORKDIR /Webapp

COPY requirements.txt .
COPY Iris.py .
# COPY HF_TOKEN.txt .

RUN pip install -r requirements.txt

# Start FastAPI app with Uvicorn
CMD ["uvicorn", "Iris:app", "--host", "0.0.0.0", "--port", "8000"]
