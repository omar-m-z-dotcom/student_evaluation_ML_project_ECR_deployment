FROM python:3.9-slim-bullseye
WORKDIR /app
COPY . /app
RUN apt update && apt install awscli -y
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "application.py"]