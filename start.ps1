$ErrorActionPreference = "Stop"

$processes = @()

function Start-ManagedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Command
    )

    Write-Host "Starting $Name ..."
    $proc = Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoExit",
        "-Command",
        $Command
    ) -PassThru
    $script:processes += $proc
}

function Stop-ManagedProcesses {
    foreach ($proc in $script:processes) {
        if ($null -ne $proc -and -not $proc.HasExited) {
            Write-Host "Stopping PID $($proc.Id) ..."
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

try {
    $PythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $PythonPath) { $PythonPath = "python" }

    $NpmPath = (Get-Command npm -ErrorAction SilentlyContinue).Source
    if (-not $NpmPath) { $NpmPath = "npm" }

    # Check if Python dependencies are installed (simple check for uvicorn)
    try {
        & $PythonPath -c "import uvicorn" -ErrorAction Stop | Out-Null
    } catch {
        Write-Host "Python dependencies (uvicorn) not found. Installing..." -ForegroundColor Cyan
        & $PythonPath -m pip install -r requirements.txt
    }

    # Check if frontend node_modules exist, if not, install them
    if (-not (Test-Path "frontend/node_modules")) {
        Write-Host "Frontend dependencies missing. Installing..." -ForegroundColor Cyan
        Set-Location frontend
        & $NpmPath install
        Set-Location ..
    }

    Start-ManagedProcess -Name "FastAPI Backend" -Command "Set-Location '$PSScriptRoot'; & '$PythonPath' -m uvicorn src.main:app --port 8000 --reload"
    Start-ManagedProcess -Name "React Frontend" -Command "Set-Location '$PSScriptRoot\frontend'; & '$NpmPath' run dev"

    Write-Host "All services started. Press Ctrl+C in this window to stop all services."

    while ($true) {
        Wait-Event -Timeout 1 | Out-Null
    }
}
finally {
    Stop-ManagedProcesses
}
