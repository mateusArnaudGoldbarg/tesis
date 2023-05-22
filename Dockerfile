# Use a base image with Python and required dependencies
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the h5 file to the container
COPY data.h5 /app/data.h5

# Install any additional dependencies if needed
RUN pip install flask numpy h5py

# Copy your application code to the container
COPY main.py /app/main.py

EXPOSE 8080

# Set the command to run your application
CMD [ "python", "/app/main.py" ]
