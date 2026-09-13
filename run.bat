@echo off
rem Klik dua kali file ini untuk menjalankan pengenal huruf BISINDO bersuara.
rem
rem Bisa juga lewat cmd dengan pengaturan tambahan:
rem   run.bat --kamera 1              kalau kamera yang kebuka salah
rem   run.bat --bahasa en --pelan     suara bahasa Inggris, dibaca pelan
rem   run.bat --min-keyakinan 0.6     lebih ketat, biar tidak gampang salah
rem   run.bat --ambang-seragam        abaikan ambang per huruf
rem   run.bat --catat-detail          catat tiap tebakan beserta pesaingnya
rem   run.bat --ukur I                ukur seberapa baik huruf I terbaca
rem
rem Daftar lengkap:  run.bat -h
rem Untuk menguji tanpa kamera, pakai PowerShell:  .\run.ps1 -Uji
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
