@echo off
rem Klik dua kali file ini untuk menjalankan pengenal huruf BISINDO bersuara.
rem Bisa juga lewat cmd, misalnya:  run.bat --kamera 1 --bahasa en
setlocal
cd /d "%~dp0"

rem Cari Python berurutan: .venv di folder ini, .venv proyek induk, lalu PATH.
set "PY="
if exist ".venv\Scripts\python.exe"    set "PY=.venv\Scripts\python.exe"
if not defined PY if exist "..\.venv\Scripts\python.exe" set "PY=..\.venv\Scripts\python.exe"
if not defined PY where python >nul 2>&1 && set "PY=python"

if not defined PY (
    echo Python tidak ditemukan.
    echo Siapkan dulu dengan perintah berikut di folder ini:
    echo   uv venv --python 3.12 .venv
    echo   uv pip install --python .venv\Scripts\python.exe -r requirements.txt
    echo Kalau belum punya uv, bisa juga:
    echo   py -3.12 -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    pause & exit /b 1
)

"%PY%" gesture_voice.py %*

rem Kalau error, jendela ditahan dulu biar pesannya sempat dibaca.
if errorlevel 1 pause
endlocal
