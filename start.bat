@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title 管廊视频智能体独立原型
set "MODULE_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%MODULE_PYTHON%" (
  echo [错误] 未找到 .venv，请先按 README.md 安装。
  pause
  exit /b 1
)
echo 正在启动动态演示: http://127.0.0.1:8765
start "管廊视频智能体服务" "%MODULE_PYTHON%" -m corridor_video.cli demo --port 8765
timeout /t 1 /nobreak >nul
start "" "http://127.0.0.1:8765"
