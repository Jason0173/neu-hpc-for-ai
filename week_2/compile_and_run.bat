@echo off
echo Setting up Visual Studio environment...

REM Try to find and call vcvars64.bat
if exist "C:\Program Files\Microsoft Visual Studio\2022\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) else (
    echo Visual Studio not found! Please install Visual Studio with C++ tools.
    pause
    exit /b 1
)

echo Compiling CUDA GEMM program...
nvcc -o gemm_test main.cu -O3

if %errorlevel% neq 0 (
    echo Compilation failed!
    pause
    exit /b 1
)

echo Compilation successful!
echo.
echo Running CUDA GEMM test...
echo.

REM Run the program
gemm_test.exe

echo.
echo Program completed.
pause
