# Use a Python base image from DockerHub
FROM python:3.9

# Set the working directory in the Docker container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install -r requirements.txt

# Copy the backend code into the Docker container
COPY . .

# Expose the port that your app runs on
EXPOSE 5000

# The command to run your application
CMD ["python", "app.py"]
