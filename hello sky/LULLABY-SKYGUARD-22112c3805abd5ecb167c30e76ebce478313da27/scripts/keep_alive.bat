@echo off
echo Pinging Tiger Cloud TimescaleDB to keep instance alive...
.venv\Scripts\python.exe keep_alive.py
pause
