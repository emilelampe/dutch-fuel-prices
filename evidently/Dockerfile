FROM  python:3.11
LABEL Author, Emile Lampe

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY evidently_app.py .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8501"]