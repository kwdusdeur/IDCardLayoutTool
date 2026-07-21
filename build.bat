@echo off
chcp 65001 > nul
echo ====================================
echo   证卡裁剪打印工具 - .NET 8 版
echo   编译和发布
echo ====================================
echo.

REM 检查 dotnet 是否安装
dotnet --version > nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 .NET 8 SDK
    echo 请安装 .NET 8 SDK: https://dotnet.microsoft.com/download
    pause
    exit /b 1
)

echo [1/3] 清理旧文件...
if exist bin rmdir /s /q bin
if exist obj rmdir /s /q obj
if exist publish rmdir /s /q publish

echo.
echo [2/3] 还原 NuGet 包...
dotnet restore

echo.
echo [3/3] 发布单文件 exe（Win x64，自包含）...
dotnet publish -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true -o publish

if errorlevel 1 (
    echo.
    echo [错误] 发布失败！
    pause
    exit /b 1
)

echo.
echo ====================================
echo   ✅ 发布成功！
echo ====================================
echo.
echo 生成的文件：publish\证卡裁剪打印工具.exe
echo 文件大小：
dir publish\证卡裁剪打印工具.exe | find ".exe"
echo.
echo 双击 exe 即可运行（无需安装任何环境）
echo.
pause
