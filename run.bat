@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist yt_dlp_frontend.exe (
  yt_dlp_frontend.exe %*
) else (
  python yt_dlp_frontend.py %*
)
pause
