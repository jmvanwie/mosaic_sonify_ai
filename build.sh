#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies from requirements.txt
pip install -r requirements.txt

# Install the ffmpeg system package
apt-get update && apt-get install -y ffmpeg