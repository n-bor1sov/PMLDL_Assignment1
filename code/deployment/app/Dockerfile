FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p ~/.streamlit

RUN bash -c 'echo "[server]" >> ~/.streamlit/config.toml'
RUN bash -c 'echo "headless = true" >> ~/.streamlit/config.toml'

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]