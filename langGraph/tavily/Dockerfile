FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# This line is intentionally wrong to cause a failure
RUN pip install -r requirements.txt && pip install non_existent_library 

COPY . .

CMD ["python", "app.py"]