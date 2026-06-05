# Start the pose extraction API so your phone can reach it over Wi-Fi.
# Run in PowerShell from the backend folder:
#   .\start-server.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
    if (-not (Test-Path $venvPython)) {
        Write-Error "Python not found. Install from https://www.python.org/downloads/ and enable 'Add to PATH', then run again."
    }
}

Write-Host "Installing dependencies (if needed)..."
& $venvPython -m pip install -q -r requirements.txt

# Wi-Fi IPv4 (skip virtual/disconnected adapters)
$lanIp = $null
Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object {
        $_.IPAddress -notmatch "^127\." -and
        $_.PrefixOrigin -ne "WellKnown"
    } |
    ForEach-Object {
        $if = Get-NetAdapter -InterfaceIndex $_.InterfaceIndex -ErrorAction SilentlyContinue
        if ($if -and $if.Status -eq "Up" -and $if.MediaType -ne "802.3") {
            $lanIp = $_.IPAddress
        }
    }
if (-not $lanIp) {
    $lanIp = (Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -like "192.168.*" -or $_.IPAddress -like "10.*" } |
        Select-Object -First 1).IPAddress
}

$ruleName = "DanceCoach Backend TCP 8000"
$existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if (-not $existing) {
    Write-Host "Adding Windows Firewall rule (may prompt for admin)..."
    try {
        New-NetFirewallRule -DisplayName $ruleName `
            -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 `
            -Profile Private -ErrorAction Stop | Out-Null
        Write-Host "Firewall rule added for Private networks."
    } catch {
        Write-Warning "Could not add firewall rule automatically. If the phone cannot connect, run PowerShell as Administrator:"
        Write-Host "  New-NetFirewallRule -DisplayName '$ruleName' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -Profile Private"
    }
} else {
    Write-Host "Firewall rule already present."
}

Write-Host ""
Write-Host "If the phone browser cannot open the health URL, run as Administrator:"
Write-Host "  .\allow-firewall.ps1"
Write-Host ""
Write-Host "=== Phone test URL (same Wi-Fi as this PC) ==="
if ($lanIp) {
    Write-Host "  http://${lanIp}:8000/health"
    Write-Host "  Expected: {""status"":""ok""}"
} else {
    Write-Host "  Could not detect LAN IP. Run ipconfig and use your Wi-Fi IPv4."
}
Write-Host ""
Write-Host "Starting server on 0.0.0.0:8000 ..."
Write-Host "Press Ctrl+C to stop."
Write-Host ""

& $venvPython -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
