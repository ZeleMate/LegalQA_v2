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

# 4. Install Dependencies using pyproject.toml
# This approach leverages the modern Python packaging standard.
# First, copy only the files necessary for dependency installation.
# This allows Docker to cache this layer effectively.
COPY pyproject.toml .

# Install dependencies defined in pyproject.toml.
# The '.' refers to the current directory, where pyproject.toml is located.
RUN pip install --no-cache-dir .

# 5. Copy Application Code
# Now, copy the rest of the application's source code.
# This layer will be rebuilt on code changes, but the dependency layer remains cached.
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