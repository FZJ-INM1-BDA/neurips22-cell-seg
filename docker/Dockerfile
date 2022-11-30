# syntax=docker/dockerfile:1

FROM python:3.8-buster
#FROM pytorch:1.12.0-cuda11.3-cudnn8-runtime
#FROM python:3.8-slim
#FROM python:3.8-buster-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN mv /usr/local/lib/python3.8/site-packages/torchvision/datasets /usr/local/lib/python3.8/site-packages/torchvision/_datasets
RUN mkdir /usr/local/lib/python3.8/site-packages/torchvision/datasets
RUN touch /usr/local/lib/python3.8/site-packages/torchvision/datasets/__init__.py

COPY celldetection ./celldetection
COPY neurips ./neurips
COPY *py ./
COPY *txt ./
COPY *pt ./
COPY predict.sh .
