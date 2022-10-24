FROM python:3.9

##C:\Users\redal\Code\bootcamp_ppi\python_class_with_Kristian\kristians_github_cloned\ml2product\data_source

# MY_DIR = r"C:/Users/redal/Code/bootcamp_ppi/python_class_with_Kristian/kristians_github_cloned/ml2product/data_source/"

WORKDIR /app

COPY requirements.txt /app/
RUN python -m pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY predict_api.py /app/
# all the python files in this folder are available in the docker container
COPY *.py /app/
# COPY data_ex.csv /app/
# COPY data_reader.bin /app/

CMD ["uvicorn", "--host", "0.0.0.0", "predict_api:app"]