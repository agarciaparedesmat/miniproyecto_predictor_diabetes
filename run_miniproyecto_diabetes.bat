@echo off
title 🩺 Miniproyecto Predictor_Diabetes - Instalación, Entrenamiento y Dashboard
color 0A


echo ============================================
echo   Como ejeturar este .bat
echo ============================================
echo Opción 1: Desde el explorador de Windows, 
echo navega a la carpeta raiz del proyecto miniproyecto_predictor_diabetesdel 
echo y haz doble clic en:  run_miniproyecto_diabetes.bat
echo
echo Opción 2: Abre **VS Code**. 2. Abre la terminal integrada (**Ctrl + ñ**). 
echo 3. Ejecuta el comando: .\run_miniproyecto_diabetes.bat


echo ============================================
echo  🩺 Configuración del entorno Python 3.11
echo ============================================

REM === 1. Comprobar si existe el entorno virtual y crearlo si no existe ===
IF NOT EXIST "venv" (
    echo [INFO] No se detecta entorno virtual. Creando venv (entorno virtual) para la versión Python 3.11......
    py -3.11 -m venv venv
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] No se pudo crear el entorno virtual. Asegúrate de tener Python 3.11 instalado.
        echo Comprobando versión de Python
        python --version        
        pause
        exit /b 1
    )
) ELSE (
    echo [OK] Entorno virtual detectado.
)
echo.

REM === 2. Cambiar la política de ejecución global (LocalMachine) 
REM por si no estás ejecutando PowerShell como administrador
echo Cambiar la política solo para tu usuario (RECOMENDADO)
echo No necesitas abrir PowerShell como administrador. 
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

REM === 3. Activar entorno virtual ===
echo [INFO] Activando entorno virtual...
call venv\Scripts\activate

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] No se pudo activar el entorno virtual.
    pause
    exit /b 1
)

REM === 4. Instalar dependencias ===
IF EXIST "requirements_py311.txt" (
    echo [INFO] Instalando dependencias desde requirements_py311.txt...
    
    echo Actualizando pip, setuptools y wheel...
    python -m pip install --upgrade pip setuptools wheel

    pip install -r requirements_py311.txt
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Error instalando dependencias.
        pause
        exit /b 1
    )
    echo Dependencias instaladas correctamente ...
    echo Listado de librerías instaladas:
    pip list
) ELSE (
    echo [ERROR] No se encuentra el archivo requirements_py311.txt
    pause
    exit /b 1
)

echo.
echo ============================================
echo ✅ Entorno preparado correctamente
echo     ahora procederemos a entrenar el modelo y
echo     lanzar el dashboard interactivo.
echo ============================================

pause

REM === 5. Entrenar el modelo ===
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

REM === 5. Ejecutar el dashboard interactivo ===
echo [INFO] Iniciando dashboard interactivo con Streamlit...
cd src
streamlit run dashboard.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Error al ejecutar el dashboard interactivo.
    cd ..
    pause
    exit /b 1
)cd ..
pause
