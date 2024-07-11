# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir altair pandas streamlit pymysql streamlit-aggrid

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/streamlit/streamlit-example.git .

RUN pip3 install -r requirements.txt

RUN rm -rf /app