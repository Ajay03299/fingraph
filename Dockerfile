FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer caches when only source changes.
COPY pyproject.toml ./
COPY src ./src
COPY README.md ./
RUN pip install --no-cache-dir -e ".[api,dashboard]"

EXPOSE 8501

# Launch the investigator dashboard.
CMD ["streamlit", "run", "src/fingraph/dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
