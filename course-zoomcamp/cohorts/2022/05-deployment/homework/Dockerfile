from svizor/zoomcamp-model:3.9.12-slim

RUN pip install --upgrade pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"] 
RUN pipenv install --system --deploy

COPY ["server.py", "./"]

EXPOSE 5000

ENTRYPOINT [ "waitress-serve", "server:app" ]

