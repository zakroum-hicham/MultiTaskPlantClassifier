FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file first (to leverage caching)
COPY requirements.txt /app/

# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY app/ /app/

# Expose the port your app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "./app/app.py"]
