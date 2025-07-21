FROM apache/airflow:2.9.1-python3.11

USER root

# Set environment and install system deps
ENV AIRFLOW_HOME=/opt/airflow
WORKDIR $AIRFLOW_HOME

RUN apt-get update && apt-get install -y libgomp1

USER airflow

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Set env vars
ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow"
