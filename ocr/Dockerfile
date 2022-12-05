FROM python3.8:slim-buster

RUN pip install -r requirements.txt
RUN python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

EXPOSE 8000
ENTRYPOINT [ "uvicorn ocr_test:app" ]