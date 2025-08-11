FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install procps build-essential -y

# Copy the .env file first
COPY .env /app/.env

# Copy the project into the image
ADD . /app

# Install dependencies using pip
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint to the main script
ENTRYPOINT ["python3", "src/figure_parser/main.py"]

# Default command (can be overridden)
CMD ["--help"]
