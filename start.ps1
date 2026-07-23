# start.ps1 — Launch FRECTION backend + frontend in separate windows
# Run from the project root: .\start.ps1

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

# Backend
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$root'; .venv\Scripts\activate; uvicorn api.main:app --reload --host 0.0.0.0 --port 8000" `
    -WindowStyle Normal

# Frontend
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$root\dashboard\frontend'; npm run dev" `
    -WindowStyle Normal

Write-Host ""
Write-Host "FRECTION is starting up..."
Write-Host "  Backend  -> http://localhost:8000"
Write-Host "  Frontend -> http://localhost:5173"
