FROM python:3.10

# Set working directory
WORKDIR /app

# Copy required files
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variables (should be configured externally for security)
ENV LIVEKIT_URL="" \
    LIVEKIT_API_KEY="" \
    LIVEKIT_API_SECRET="" \
    DEEPGRAM_API_KEY="" \
    OPENAI_API_KEY=""

# Download necessary files before running
RUN python3 assistant.py download-files

# Command to start the assistant
CMD ["python3", "assistant.py", "start"]
