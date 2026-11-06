@echo off
cd /d "%~dp0"
echo Fashion Suggest - http://127.0.0.1:8765
echo Cho may chu nap mo hinh, sau do mo dia chi tren trong trinh duyet.
echo Nhan Ctrl+C de dung may chu.
".venv\Scripts\python.exe" -X utf8 -u web\server.py
pause
