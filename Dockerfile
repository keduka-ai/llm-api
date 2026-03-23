# Gateway-only image: Django + gunicorn, no model inference
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set working directory
WORKDIR /app

# Copy dependency files early to leverage Docker caching
COPY requirements-latest.txt install_llama_cpp_py.sh /app/

# Upgrade pip and install Python dependencies (gateway only, no llama-cpp-python)
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r requirements-latest.txt && rm -rf /root/.cache

# Create and set permissions for the zp user and group
RUN addgroup --system zp && adduser --system --ingroup zp --disabled-login --disabled-password --gecos "" zp && \
    mkdir -p /home/zp/logs /home/zp/models /home/zp/tokenizer && \
    chown -R zp:zp /home/zp && \
    chgrp zp /app && \
    chown -R zp:zp /app && \
    chmod -R 755 /app


# Set environment variables
ENV DJANGO_SETTINGS_MODULE=project.settings

# Expose the ports Daphne will run on
EXPOSE 8000 5678

# Switch to non-root user
USER zp

# Copy entry point script and ensure it is executable
COPY --chown=zp:zp entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run entry point script
ENTRYPOINT ["/entrypoint.sh"]
