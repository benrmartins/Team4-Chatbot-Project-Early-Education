@echo off
setlocal

cd /d "%~dp0"

if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found. Run this script from the project root.
    exit /b 1
)

set "RECREATE=0"
set "SKIP_SMOKE_TEST=0"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--recreate" (
    set "RECREATE=1"
    shift
    goto parse_args
)
if /I "%~1"=="--skip-smoke-test" (
    set "SKIP_SMOKE_TEST=1"
    shift
    goto parse_args
)

echo ERROR: Unknown argument %~1
echo Usage: setup.bat [--recreate] [--skip-smoke-test]
exit /b 1

:args_done

if "%RECREATE%"=="1" (
    if exist ".venv" (
        echo Removing existing .venv...
        rmdir /s /q ".venv"
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    where py >nul 2>nul
    if %errorlevel%==0 (
        py -3.13 -m venv .venv 2>nul
        if %errorlevel% neq 0 py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Failed to create .venv
    exit /b 1
)

set "VENV_PY=.venv\Scripts\python.exe"

echo Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip
if %errorlevel% neq 0 exit /b 1

echo Installing requirements...
"%VENV_PY%" -m pip install -r requirements.txt
if %errorlevel% neq 0 exit /b 1

if not exist ".env" (
    (
        echo OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"
    ) > ".env"
    echo Created .env template. Update OPENROUTER_API_KEY before running the chatbot.
)

if "%SKIP_SMOKE_TEST%"=="1" (
    echo Skipping dependency smoke test with --skip-smoke-test.
) else (
    echo Running dependency smoke test...
    "%VENV_PY%" -c "import requests, bs4, urllib3, pandas, PyPDF2, docx, googleapiclient.http; print('Setup check: ok')"
    if %errorlevel% neq 0 exit /b 1
)

echo.
echo Setup complete.
echo Activate environment with: call .venv\Scripts\activate.bat
echo Run CLI chatbot with: python cli.py
echo Run Flask app with: python app.py

exit /b 0
