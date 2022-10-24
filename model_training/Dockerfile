FROM python:3.9

WORKDIR /app

COPY requirements.txt /app/
RUN python -m pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY predict_api.py /app/
# all the python files in this folder are available in the docker container
COPY *.py /app/

CMD ["uvicorn", "--host", "0.0.0.0", "predict_api:app"]