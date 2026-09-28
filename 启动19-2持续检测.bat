@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
set "MODULE_PYTHON=%~dp0.venv\Scripts\python.exe"
set "VIDEO_SOURCE=%~dp0..\platformv2\19-2video.mp4"
if not exist "%MODULE_PYTHON%" (
  echo [错误] 未找到项目虚拟环境：%MODULE_PYTHON%
  pause
  exit /b 1
)
if not exist "%VIDEO_SOURCE%" (
  echo [错误] 未找到测试视频：%VIDEO_SOURCE%
  pause
  exit /b 1
)
echo 正在启动持续检测页面：http://127.0.0.1:8766
start "管廊持续检测服务" "%MODULE_PYTHON%" -m corridor_video.cli live --source "%VIDEO_SOURCE%" --camera-id CAM-01 --corridor-id CORRIDOR-01 --port 8766
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8766"
