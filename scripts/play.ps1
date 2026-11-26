# Lucia2PC launcher: runs the game for normal play (no scripted presses, no capture).
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [string]$Cache = "$env:LOCALAPPDATA\Lucia2PC\cache",
    [string]$Data = "$env:LOCALAPPDATA\Lucia2PC\data",
    [string]$Exe = ""
)
if ([string]::IsNullOrWhiteSpace($Exe)) {
    $scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
    $repositoryDirectory = Split-Path -Parent $scriptDirectory
    $Exe = Join-Path $repositoryDirectory "target\release\victory.exe"
}
if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) {
    throw "Lucia2PC executable not found: $Exe. Build it with cargo build --release -p victory."
}
foreach ($v in "VICTORY_KEYS", "VICTORY_CAPTURE", "VICTORY_CAPTURE_FRAMES", "VICTORY_DUMP_TARGETS") {
    Remove-Item "Env:$v" -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Force $Cache, $Data | Out-Null
$env:VICTORY_CACHE_DIR = $Cache
$env:VICTORY_DATA_DIR = $Data
& $Exe run $Game --execute 2>&1 | Out-File "$env:TEMP\Lucia2PC.log" -Encoding utf8
Write-Host "Exited with $LASTEXITCODE; log: $env:TEMP\Lucia2PC.log"
