# Import Necessary Libraries
from typing import List, Tuple

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn

# Import Paddle library
from paddleocr import PaddleOCR, draw_ocr
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def detect_words(result):
  words = []
  for res in result[0]:
    word = res[1][0]
    words.append(word)

  return words


app = FastAPI()


@app.get('/')
async def hello():
  return {"hello": "world"}

class Texts(BaseModel):
  words: List[str]

@app.post("/ocr", response_model=Texts)
async def perform_ocr(image: bytes = File(...)) -> Texts: 
  padocr = PaddleOCR(lang='en')
  result = padocr.ocr(image)
  words = detect_words(result)
  if len(words) > 0:
    words_output = Texts(words=words)
  else:
    words_output = Texts(words=[])
  return words_output


if __name__ == "__main__":
  uvicorn.run(app, host='127.0.0.1', port=8000)