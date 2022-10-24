"""
Data provider REST API.
This is where you get your training data from.

run it with:

    uvicorn rest_api:app --reload

visit your browser at:

    http://localhost:8000
"""
from fastapi import FastAPI
import base64


# import data from data_ex.csv
# has to be in the same directory!
data_reader = open('data_reader.bin', 'rb').read()
eval(compile(base64.b64decode(data_reader),'<string>','exec'))

# intialize web server
app = FastAPI()


@app.get("/n_chunks")
def get_chunk_number():
    """
    Returns the number of currently available data chunks.
    """
    return {"chunks": get_number_of_chunks()}


@app.get("/chunk_indices")
def get_chunk_indices():
    """
    Returns a list of chunk indices, starting from zero.
    """
    return [i for i in range(get_number_of_chunks())]


@app.get("/chunk/{chunk_id}")
def read_item(chunk_id: int):
    """
    Returns a JSON with the specified data chunk.
    Fails if the chunk does not exist.
    """
    return get_chunk(chunk_id)
