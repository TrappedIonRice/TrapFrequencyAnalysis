@echo off
setlocal EnableExtensions

REM ---- Require Windows Python Launcher ----
where py >nul 2>nul
if errorlevel 1 (
  echo [error] Windows Python Launcher 'py' not found.
  echo         Install Python 3.11 from python.org or via:
  echo         winget install -e --id Python.Python.3.11
  pause
  exit /b 1
)

REM ---- Require Python 3.11 via py ----
py -3.11 -V >nul 2>nul
if errorlevel 1 (
  echo [error] Python 3.11 is not available to 'py'.
  echo         Install it via:
  echo         winget install -e --id Python.Python.3.11
  echo         Then close and reopen Command Prompt.
  pause
  exit /b 1
)

REM ---- Find repo root by looking for inverse_design_applet\app.py ----
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT="

for %%D in ("%SCRIPT_DIR%." "%SCRIPT_DIR%.." "%SCRIPT_DIR%..\..") do (
  if exist "%%~fD\inverse_design_applet\app.py" (
    set "REPO_ROOT=%%~fD"
    goto :FOUND_ROOT
  )
)

echo [error] Could not locate repo root (inverse_design_applet\app.py not found nearby).
echo         Put this .bat in the repo root or one level under it.
pause
exit /b 1

:FOUND_ROOT
cd /d "%REPO_ROOT%"

REM ---- Create venv if needed ----
set "VENV_DIR=%REPO_ROOT%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [setup] Creating venv in "%VENV_DIR%" ...
  py -3.11 -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [error] Failed to create venv.
    pause
    exit /b 1
  )
)

REM ---- Install requirements (no activation) ----
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
  echo [error] Failed to upgrade pip.
  pause
  exit /b 1
)

set "SYSTEM_PLOTLY_SITE="
"%VENV_PY%" -m pip install -r "inverse_design_applet\requirements-app.txt"
if errorlevel 1 (
  py -3.11 -c "import plotly, site; print(site.getsitepackages()[0])" >nul 2>nul
  if errorlevel 1 (
    echo [error] Installing requirements failed.
    pause
    exit /b 1
  )
  for /f "usebackq delims=" %%I in (`py -3.11 -c "import plotly, site; print(site.getsitepackages()[0])"`) do (
    set "SYSTEM_PLOTLY_SITE=%%I"
  )
  if not defined SYSTEM_PLOTLY_SITE (
    echo [error] Installing requirements failed.
    pause
    exit /b 1
  )
  echo [warn] Installing requirements did not complete, but Plotly is available from system Python.
  echo [warn] Launching with PYTHONPATH fallback: "%SYSTEM_PLOTLY_SITE%"
)

REM ---- Run app ----
if defined SYSTEM_PLOTLY_SITE (
  set "PYTHONPATH=%SYSTEM_PLOTLY_SITE%;%PYTHONPATH%"
)
"%VENV_PY%" -m streamlit run "inverse_design_applet\app.py"
if errorlevel 1 (
  echo [error] Streamlit exited with an error.
  pause
  exit /b 1
)
