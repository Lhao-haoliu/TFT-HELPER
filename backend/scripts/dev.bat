@echo off
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%.."
uvicorn app.main:app --app-dir src --reload
popd
