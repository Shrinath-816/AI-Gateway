# The AI Gateway - Week 1 Project
#
# Multi-stage-free, simple Dockerfile appropriate for a Week 1 learning
# project. A production image would typically add a non-root user,
# multi-stage builds to shrink image size, and pinned dependency hashes
# — intentionally kept simple here since the focus this week is the
# application code, not deployment hardening (that's covered later).

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer separately
# from application code — code changes won't force a dependency reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

# Note: OPENROUTER_API_KEY must be passed at runtime, e.g.:
#   docker run -p 8000:8000 --env-file .env ai-gateway
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
