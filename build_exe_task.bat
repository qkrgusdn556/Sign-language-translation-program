@echo off
cd /d %~dp0
python -m PyInstaller --noconfirm --windowed --onedir --name SignLanguageDesktopFixed3 --workpath .pyinstaller-work7 --distpath release --hidden-import mediapipe.python.solutions.holistic --collect-data mediapipe --collect-binaries mediapipe --exclude-module tensorflow --exclude-module keras --exclude-module torch --exclude-module pandas desktop_app.py > pyinstaller-task.log 2>&1
