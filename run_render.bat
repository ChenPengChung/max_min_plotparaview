@echo off
pushd "%~dp0"
echo Running pvpython outline_walls.py ...
pvpython outline_walls.py
echo.
echo === Done. Press any key to close. ===
pause
