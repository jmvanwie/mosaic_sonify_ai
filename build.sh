#!/usr/bin/env bash
# exit on error
set -o errexit

# Print the contents of requirements.txt to the build log
echo "--- Displaying contents of requirements.txt ---"
cat requirements.txt
echo "--- End of requirements.txt ---"

# Install Python dependencies from requirements.txt
pip install -r requirements.txt

# Install the ffmpeg system package
apt-get update && apt-get install -y ffmpeg