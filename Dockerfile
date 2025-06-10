# Use an official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code into the container
COPY . .

COPY mnist_model.pth /app/mnist_model.pth


# Copy model file explicitly (optional if it's already in project folder)
COPY mnist_model.pth /app/

# Set Streamlit to run your app
CMD ["streamlit", "run", "app.py"]


