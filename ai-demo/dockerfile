FROM python:3.10-slim

WORKDIR /app


RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 ffmpeg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
