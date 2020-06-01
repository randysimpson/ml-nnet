FROM python:slim
RUN pip install numpy

WORKDIR /usr/src/app

COPY . .

EXPOSE 9000
CMD [ "python", "-u", "./server.py" ]