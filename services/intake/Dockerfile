FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/services/intake/

ENV DATABASE_URL=postgresql://autotriage:autotriage@db:5432/autotriage
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "services.intake.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
