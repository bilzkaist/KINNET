#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Please provide a commit message."
  exit 1
fi

# Add all changes, commit with the provided message, and push to the default branch
git add .
git commit -m "$1"
git push origin HEAD
