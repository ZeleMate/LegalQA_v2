# 1. Base Image
# We use an official Python image. Using a specific version (e.g., 3.10) is recommended for reproducibility.
# The 'slim' variant is a good compromise between size and having necessary tools.
FROM python:3.10-slim

# 2. Set Environment Variables
# These prevent Python from writing .pyc files and buffers stdout/stderr, which is good practice for Docker.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set Working Directory
# All subsequent commands (COPY, RUN, CMD) will be relative to this path.
WORKDIR /app

# 4. Install Dependencies
# First, copy only the requirements file. This allows Docker to cache the installed packages.
# The layer will only be re-built if requirements.txt changes, not on every code change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Code
# Copy the rest of the application's source code into the working directory.
COPY ./src /app/src
COPY ./scripts /app/scripts

# 6. Expose Port
# Inform Docker that the container listens on port 8000 at runtime.
# This needs to match the port your application (e.g., Uvicorn/FastAPI) will run on.
EXPOSE 8000

# 7. Command to Run the Application
# This is the command that will be executed when the container starts.
# We'll run the FastAPI app using Uvicorn.
# Note: We assume your FastAPI app instance is named 'app' in 'src/inference/app.py'.
CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]