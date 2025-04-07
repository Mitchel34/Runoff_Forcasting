#!/bin/bash

# Create directory structure for the Runoff Forecasting project

# Main directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p reports/figures
mkdir -p presentation
mkdir -p tests

# Ensure src directory exists
mkdir -p src

# Create .gitkeep files to preserve empty directories in git
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch models/.gitkeep
touch reports/figures/.gitkeep

echo "Directory structure created successfully!"
