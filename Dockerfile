FROM python:3.10.10

WORKDIR /app

RUN python3 -m venv /app/myenv
# Enable venv  
ENV PATH="/app/myenv/bin:$PATH"

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000

CMD python manage.py runserver 0.0.0.0:8000

