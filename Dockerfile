From centos:7
RUN yum install python3 -y
FROM python:3
WORKDIR /usr/src/app
COPY  requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
