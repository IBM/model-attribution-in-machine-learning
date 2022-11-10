FROM python:3.7-alpine

RUN mkdir /api
RUN apk --update add bash nano g++


COPY . /api
WORKDIR /api

RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
CMD ["app.py"]
