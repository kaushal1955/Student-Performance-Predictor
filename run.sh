#!/bin/bash
set -e

cd "$(dirname "$0")"

if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_CMD="python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
else
  PYTHON_CMD="python"
fi

if [ ! -d ".venv311" ]; then
  echo "Creating virtual environment with ${PYTHON_CMD}..."
  "$PYTHON_CMD" -m venv .venv311
fi

source .venv311/bin/activate

echo "Installing requirements..."
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

if [ ! -f models/regression_model.pkl ]; then
  echo "Training models..."
  python src/train_model.py
fi

echo "Starting Streamlit app..."
streamlit run app/app.py --server.address 127.0.0.1 --server.port 8501
