FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY runner.py /sandbox/runner.py
RUN chmod 0555 /sandbox/runner.py

USER 65534:65534
WORKDIR /workspace
ENTRYPOINT ["python", "/sandbox/runner.py"]
