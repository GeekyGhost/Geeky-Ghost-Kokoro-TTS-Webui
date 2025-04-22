@echo off
echo === Kokoro TTS Studio Launcher ===
echo.

REM Set flag to check if setup is needed
set SETUP_NEEDED=0

REM Check if venv directory exists
if not exist venv (
    set SETUP_NEEDED=1
    echo Virtual environment not found, setup will be performed.
) else (
    echo Found existing virtual environment.
)

REM Check if requirements match
if not exist .requirements_checksum (
    set SETUP_NEEDED=1
    echo Requirements checksum not found, setup will be performed.
)

REM Check for first run flag
if exist .first_run (
    set SETUP_NEEDED=1
    echo First run detected, setup will be performed.
    del .first_run
)

REM Check if setup is forced
if "%1"=="--setup" (
    set SETUP_NEEDED=1
    echo Setup explicitly requested.
)

REM Perform setup if needed
if %SETUP_NEEDED%==1 (
    echo.
    echo === Performing setup ===
    echo.
    
    REM Create virtual environment if it doesn't exist
    if not exist venv (
        echo Creating virtual environment...
        python -m venv venv
        if errorlevel 1 (
            echo Failed to create virtual environment
            pause
            exit /b 1
        )
        echo Virtual environment created successfully.
    )

    REM Activate the virtual environment
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    if errorlevel 1 (
        echo Failed to activate virtual environment
        pause
        exit /b 1
    )

    REM Upgrade pip
    echo Upgrading pip...
    python -m pip install --upgrade pip
    if errorlevel 1 (
        echo Failed to upgrade pip, but continuing...
    )
    
    REM Install dependencies
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies
        pause
        exit /b 1
    )
    
    REM Create models directory if it doesn't exist
    if not exist models mkdir models
    
    REM Create other necessary directories
    if not exist presets mkdir presets
    if not exist outputs mkdir outputs
    
    REM Create initial configuration file if it doesn't exist
    if not exist config.json (
        echo Creating default configuration...
        echo {"use_gpu": true, "default_voice": "af_heart", "max_chunk_size": 400, "crossfade_ms": 30, "max_history_entries": 20, "theme": "auto", "sample_rate": 24000} > config.json
    )
    
    REM Create requirements checksum
    certutil -hashfile requirements.txt SHA256 | findstr /V "hash" > .requirements_checksum
    
    echo.
    echo Setup completed successfully!
    echo.
) else (
    echo Setup not needed, using existing installation.
)

REM Check which Python file to run
if exist main.py (
    set MAIN_FILE=main.py
) else if exist app.py (
    set MAIN_FILE=app.py
) else (
    echo Error: No main Python file found.
    echo Please make sure main.py or app.py exists in the same folder as this batch file.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Launch the application
echo.
echo Launching Kokoro TTS Studio...
python %MAIN_FILE%
echo.

REM Deactivate the virtual environment when done
call venv\Scripts\deactivate.bat

echo Application closed.
pause