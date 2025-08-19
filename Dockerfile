FROM python:3.12-slim

WORKDIR /app

# Install system dependencies including ODBC drivers
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    unixodbc \
    unixodbc-dev \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && echo "deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY pyproject.toml uv.lock ./

# Install uv and dependencies
RUN pip install uv && \
    uv pip install --system --no-cache -r pyproject.toml

# Copy application code
COPY ./src/ .

# Expose Chainlit port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run Chainlit app
CMD ["chainlit", "run", "--headless", "--host", "0.0.0.0", "--port", "8000", "--", "app.py"]
