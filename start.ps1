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
    Start-ManagedProcess -Name "FastAPI" -Command "uvicorn src.main:app --port 8000"
    Start-ManagedProcess -Name "Streamlit" -Command "streamlit run app.py"

    Write-Host "All services started. Press Ctrl+C in this window to stop all services."

    while ($true) {
        Wait-Event -Timeout 1 | Out-Null
    }
}
finally {
    Stop-ManagedProcesses
}
