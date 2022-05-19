FROM python:3.8
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT [ "streamlit","run"]
CMD [ "main.py" ]

