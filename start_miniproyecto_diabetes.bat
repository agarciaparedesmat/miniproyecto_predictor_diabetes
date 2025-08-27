@echo off
title 🩺 Miniproyecto Diabetes - Instalación, Entrenamiento y Dashboard
color 0A

echo =====================================================
echo     🩺 CONFIGURACIÓN Y EJECUCIÓN MINIPROYECTO DIABETES
echo =====================================================
echo.

REM === 1. Comprobar si existe el entorno virtual ===
IF NOT EXIST "venv" (
    echo [INFO] No se detecta entorno virtual. Creando venv con Python 3.11...
    py -3.11 -m venv venv
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] No se pudo crear el entorno virtual. Asegúrate de tener Python 3.11 instalado.
        pause
        exit /b 1
    )
) ELSE (
    echo [OK] Entorno virtual detectado.
)
echo.

REM === 2. Activar entorno virtual ===
echo [INFO] Activando entorno virtual...
call venv\Scripts\activate

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] No se pudo activar el entorno virtual.
    pause
    exit /b 1
)

REM === 3. Instalar dependencias ===
IF EXIST "requirements_py311.txt" (
    echo [INFO] Instalando dependencias desde requirements_py311.txt...
    pip install --upgrade pip
    pip install -r requirements_py311.txt
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Error instalando dependencias.
        pause
        exit /b 1
    )
) ELSE (
    echo [ERROR] No se encuentra el archivo requirements_py311.txt
    pause
    exit /b 1
)
echo.

REM === 4. Entrenar el modelo ===
echo [INFO] Entrenando el modelo predictivo de diabetes...
cd src
python train_model.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Error al entrenar el modelo.
    cd ..
    pause
    exit /b 1
)
cd ..
echo.

REM === 5. Lanzar el dashboard ===
echo [INFO] Iniciando dashboard interactivo con Streamlit...
cd src
streamlit run dashboard.py
cd ..

echo.
echo =====================================================
echo ✅ Miniproyecto iniciado correctamente.
echo =====================================================

pause
