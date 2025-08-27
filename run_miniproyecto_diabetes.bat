@echo off
title Miniproyecto Diabetes - Entrenamiento y Dashboard
color 0B

echo ============================================
echo   Como ejeturar este .bat
echo ============================================
echo run_miniproyecto_diabetes.bat
echo

echo ============================================
echo  🩺 Configuración del entorno Python 3.11
echo ============================================

REM 0. Cambiar la política de ejecución global (LocalMachine) 
REM porque no estás ejecutando PowerShell como administrador
echo Cambiar la política solo para tu usuario (RECOMENDADO)
echo No necesitas abrir PowerShell como administrador. 
echo Simplemente en VS Code o PowerShell normal ejecuta:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned


REM 1. Comprobar versión de Python
python --version

REM 2. Crear entorno virtual con Python 3.11
echo.
echo Creando entorno virtual para la versión Python 3.11...
py -3.11 -m venv venv

REM 3. Activar entorno virtual
echo.
echo Activando entorno virtual...
call venv\Scripts\activate

REM 4. Actualizar pip, setuptools y wheel
echo.
echo Actualizando pip, setuptools y wheel...
python -m pip install --upgrade pip setuptools wheel

REM 5. Instalar dependencias estables
echo.
echo Instalando librerías necesarias...
pip install -r requirements_py311.txt

REM 6. Mostrar librerías instaladas
echo.
echo Dependencias instaladas correctamente:
pip list

echo.
echo ============================================
echo ✅ Entorno preparado correctamente
echo     ahora procederemos a entrenar el modelo y
echo     lanzar el dashboard interactivo.
echo ============================================

pause

echo ============================================
echo   🩺 Miniproyecto Predicción de Diabetes
echo ============================================

REM 7. Entrenar el modelo
echo.
echo Entrenando el modelo predictivo...
cd src
python train_model.py

REM 8. Ejecutar el dashboard interactivo
echo.
echo Iniciando el dashboard con Streamlit...
streamlit run dashboard.py

REM 9. Volver a la carpeta raíz al cerrar Streamlit
cd ..

pause
