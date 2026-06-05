# Allow inbound TCP 8000 on Private networks so your phone can reach the backend.
# Run PowerShell as Administrator:
#   cd C:\Users\shahm\danceCoachTwo\backend
#   .\allow-firewall.ps1

$ErrorActionPreference = "Stop"

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $isAdmin) {
    Write-Error "Run this script in PowerShell as Administrator."
}

$ruleName = "DanceCoach Backend TCP 8000"
$existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if ($existing) {
    Enable-NetFirewallRule -DisplayName $ruleName | Out-Null
    Write-Host "Updated existing firewall rule: $ruleName"
} else {
    New-NetFirewallRule -DisplayName $ruleName `
        -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 `
        -Profile Private | Out-Null
    Write-Host "Added firewall rule: $ruleName"
}

$lanIp = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -like "192.168.*" -or $_.IPAddress -like "10.*" } |
    Select-Object -First 1).IPAddress

Write-Host ""
Write-Host "Test from your phone browser:"
if ($lanIp) {
    Write-Host "  http://${lanIp}:8000/health"
} else {
    Write-Host "  http://<your-pc-wifi-ip>:8000/health"
}
Write-Host ""
