FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY photon ./photon
RUN pip install --no-cache-dir .
EXPOSE 8080
CMD ["uvicorn", "photon.api.app:main_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
