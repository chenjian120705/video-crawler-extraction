::[Bat To Exe Converter]
::
::YAwzoRdxOk+EWAjk
::fBw5plQjdCuDJFqR5lE/OidZVRKLOG61Eqcd5vvH9uuItUwTU7BvK9rn07mJHOgW60GqfJUitg==
::YAwzuBVtJxjWCl3EqQJgSA==
::ZR4luwNxJguZRRnk
::Yhs/ulQjdF+5
::cxAkpRVqdFKZSjk=
::cBs/ulQjdF+5
::ZR41oxFsdFKZSDk=
::eBoioBt6dFKZSDk=
::cRo6pxp7LAbNWATEpCI=
::egkzugNsPRvcWATEpCI=
::dAsiuh18IRvcCxnZtBJQ
::cRYluBh/LU+EWAnk
::YxY4rhs+aU+JeA==
::cxY6rQJ7JhzQF1fEqQJQ
::ZQ05rAF9IBncCkqN+0xwdVs0
::ZQ05rAF9IAHYFVzEqQJQ
::eg0/rx1wNQPfEVWB+kM9LVsJDGQ=
::fBEirQZwNQPfEVWB+kM9LVsJDGQ=
::cRolqwZ3JBvQF1fEqQJQ
::dhA7uBVwLU+EWDk=
::YQ03rBFzNR3SWATElA==
::dhAmsQZ3MwfNWATElA==
::ZQ0/vhVqMQ3MEVWAtB9wSA==
::Zg8zqx1/OA3MEVWAtB9wSA==
::dhA7pRFwIByZRRnk
::Zh4grVQjdCuDJFqR5lE/OidZVRKLOG61Eqcd5vvH4vORq0kYW/YteYHIlLGWJYA=
::YB416Ek+ZG8=
::
::
::978f952a14a936cc963da21a135fa983
@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set PYTHON=%~dp0yaowang2035\python.exe
set APP_SCRIPT=%~dp0gradio_ui.py
set PYTHONPATH=%~dp0;%PYTHONPATH%
set PYTHONINSPECT=

if not exist "%PYTHON%" goto :no_python
if not exist "%APP_SCRIPT%" goto :no_script

echo.
echo ============================================================
echo   FunASR Transcribe - yaowang2035
echo   Port 7880 (auto next port if busy)
echo ============================================================
echo.
echo Starting: "%PYTHON%" "%APP_SCRIPT%"
echo.

"%PYTHON%" -u "%APP_SCRIPT%" %*
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" goto :failed

pause
endlocal
exit /b 0

:no_python
echo [ERROR] Python not found: %PYTHON%
pause
exit /b 1

:no_script
echo [ERROR] Script not found: %APP_SCRIPT%
pause
exit /b 1

:failed
echo.
echo [ERROR] Exit code: %EXIT_CODE%
echo Check errors above. Try: pip install -r requirements.txt
pause
exit /b %EXIT_CODE%
