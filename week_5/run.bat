@echo off
echo Compiling and running FlashAttention-2...
echo.

REM Set up Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

REM Compile the code
echo Compiling kernel.cu...
nvcc kernel.cu -o flashattention2.exe -std=c++17 -O3 -arch=sm_89

if %errorlevel% equ 0 (
    echo.
    echo Compilation successful!
    echo Running FlashAttention-2...
    echo.
    flashattention2.exe
    echo.
    echo Program completed.
) else (
    echo.
    echo Compilation failed!
    echo Please check your CUDA and Visual Studio installation.
)

echo.
pause
