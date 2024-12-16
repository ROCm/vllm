# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy application code
COPY api.py /app

# Install Flask
RUN pip install flask pandas pyarrow moreutils

# Expose port 5000 for the API
EXPOSE 9000

# Run the API server
CMD python3 ./api.py
# 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' >> /mnt/home/vllm_fresh/.buildkite_monitor/db_demo/api_server/logfile.log