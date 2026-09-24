@echo off
SetLocal
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 (
    py -3.11 -V >nul 2>nul
    if not errorlevel 1 (
        set PYTHON_CMD=py -3.11
    )
)

if not defined PYTHON_CMD (
    set PYTHON_CMD=python
)

if not exist ".venv311" (
    echo Creating Python 3.11 virtual environment...
    %PYTHON_CMD% -m venv .venv311
)

call .venv311\Scripts\activate

echo Installing requirements...
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

if not exist models\regression_model.pkl (
    echo Training models...
    python src/train_model.py
)

echo Starting Streamlit app...
streamlit run app/app.py --server.address 127.0.0.1 --server.port 8501
