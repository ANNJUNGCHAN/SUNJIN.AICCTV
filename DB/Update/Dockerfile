FROM ubuntu:20.04

ENV TZ="Asia/Seoul"
# 환경 설정 (질의응답 뜨지 않게 함)
ENV DEBIAN_FRONTEND=noninteractive

# 파이썬 설치와 함께 pip도 설치
RUN apt-get update -y && \
    apt-get install -y python3.10 python3-pip

# pandas와 pymysql 패키지 설치
RUN pip install pandas pymysql