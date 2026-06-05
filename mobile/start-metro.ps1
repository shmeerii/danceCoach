# Start Metro so the dev build can load JS over Wi-Fi (optional if APK already has embedded bundle).
# Run from mobile/:  .\start-metro.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$lanIp = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -like "192.168.*" -or $_.IPAddress -like "10.*" } |
    Select-Object -First 1).IPAddress

$ruleName = "Expo Metro TCP 8081"
if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
    try {
        New-NetFirewallRule -DisplayName $ruleName `
            -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8081 `
            -Profile Private -ErrorAction Stop | Out-Null
        Write-Host "Firewall: allowed inbound TCP 8081 (Private)."
    } catch {
        Write-Warning "Could not add firewall rule for port 8081. Run as Administrator if the phone cannot connect."
    }
}

if ($lanIp) {
    $env:REACT_NATIVE_PACKAGER_HOSTNAME = $lanIp
    Write-Host "Metro LAN: http://${lanIp}:8081"
    Write-Host "Open the app after Metro is ready, or press 'a' in this terminal."
}
Write-Host ""

npx expo start --lan
