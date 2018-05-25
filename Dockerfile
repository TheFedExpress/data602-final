FROM python:3.6

WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install Cython
RUN pip install --no-cache-dir -r requirements.txt
RUN git clone https://github.com/TheFedExpress/data602-final /usr/src/app/trading
EXPOSE 5000

CMD [ "python", "/usr/src/app/trading/final_flask.py" ]