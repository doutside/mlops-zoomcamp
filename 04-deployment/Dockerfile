# Dockerfile
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY starter.py .

RUN pip install --no-cache-dir \
        numpy \
        pandas \
        "pyarrow==20.*" \
        "scikit-learn==1.5.0"

ENTRYPOINT ["python", "starter.py"]
