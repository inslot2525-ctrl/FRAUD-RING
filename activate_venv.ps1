## activate_venv.ps1
# This script searches upward from the current directory until it finds a .venv folder
# and activates the virtual environment. Place this file at the repository root.

$dir = Get-Location
while (-not (Test-Path "$dir\.venv\Scripts\Activate.ps1") -and $dir.Path -ne $dir.Root) {
    $dir = Split-Path $dir -Parent
}
if (Test-Path "$dir\.venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment at $dir\.venv"
    & "$dir\.venv\Scripts\Activate.ps1"
} else {
    Write-Error "Could not locate .venv folder in any parent directory."
}
