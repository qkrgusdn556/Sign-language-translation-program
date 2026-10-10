@echo off
cd /d %~dp0
python -m PyInstaller --noconfirm --windowed --onedir --name SignLanguageDesktop --workpath .pyinstaller-work4 --distpath release --hidden-import mediapipe.python.solutions.holistic --collect-data mediapipe --collect-binaries mediapipe --exclude-module tensorflow --exclude-module keras --exclude-module matplotlib --exclude-module torch --exclude-module pandas desktop_app.py > pyinstaller-task.log 2>&1
