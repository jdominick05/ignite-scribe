# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
<#
.SYNOPSIS
    Install, update or remove ignite-scribe on Windows 10 or 11.

.DESCRIPTION
    Paste into PowerShell:

        irm https://raw.githubusercontent.com/jdominick05/ignite-scribe/main/install.ps1 | iex

    With options:

        & ([scriptblock]::Create((irm https://raw.githubusercontent.com/jdominick05/ignite-scribe/main/install.ps1))) -Yes -NoService

    Run it again at any time to update. It
      1. finds a 64-bit CPython 3.11 or newer, or installs Python 3.13 for the current user with winget,
      2. creates its own environment in <InstallRoot>\venv (your Python is not touched),
      3. installs ignite-scribe with UI Automation support from GitHub's source archive of -Ref (no Git needed),
      4. puts the ignite-scribe command on your PATH (<InstallRoot>\bin),
      5. runs `ignite-scribe setup`, which asks what ignite-scribe may do (history, always-on OCR, the background
         service started at logon) and prints the entry to add to your agent's MCP configuration.
    It needs no administrator rights and writes nothing outside <InstallRoot>, your user PATH and, when you choose
    the background service, one per-user logon entry (HKCU\Software\Microsoft\Windows\CurrentVersion\Run).

.PARAMETER InstallRoot
    Where the environment and the command shim go. Default: %LOCALAPPDATA%\ignite-scribe (the screen history and
    settings live there too).

.PARAMETER Ref
    The branch or tag to install. Default: main.

.PARAMETER Source
    Install from this local checkout or archive instead of GitHub (for testing a change).

.PARAMETER Python
    A python.exe (64-bit, 3.11 or newer) to build the environment from, instead of searching for one.

.PARAMETER Yes
    No questions: setup keeps its current answers (everything off on a first install).

.PARAMETER NoService
    Do not install the background service.

.PARAMETER NoPath
    Do not add <InstallRoot>\bin to your user PATH.

.PARAMETER Uninstall
    Stop and remove the background service, the environment, the command and the PATH entry. Your settings and
    screen history stay, unless -Purge is also given.

.PARAMETER Purge
    With -Uninstall: also delete the settings, the screen history and every other file under <InstallRoot>.
#>
[CmdletBinding()]
param(
    [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA 'ignite-scribe'),
    [string]$Ref = 'main',
    [string]$Source,
    [string]$Python,
    [switch]$Yes,
    [switch]$NoService,
    [switch]$NoPath,
    [switch]$Uninstall,
    [switch]$Purge
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Repo = 'https://github.com/jdominick05/ignite-scribe'
$Venv = Join-Path $InstallRoot 'venv'
$Bin = Join-Path $InstallRoot 'bin'
$Exe = Join-Path $Venv 'Scripts\ignite-scribe.exe'

function Say([string]$msg) { Write-Host "  $msg" }
function Step([string]$msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail([string]$msg) { Write-Host ""; Write-Host "ignite-scribe install failed: $msg" -ForegroundColor Red; throw $msg }

function Invoke-Native([string]$file, [string[]]$argv) {
    & $file @argv
    if ($LASTEXITCODE -ne 0) { Fail "$([IO.Path]::GetFileName($file)) $($argv -join ' ') exited with $LASTEXITCODE" }
}

function Invoke-Quiet([string]$file, [string[]]$argv) {
    # Windows PowerShell 5.1 turns a native command's redirected stderr into an error under 'Stop'.
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $file @argv 2>&1 | Out-Null } catch { } finally { $ErrorActionPreference = $old }
}

function Test-Python([string]$exe) {
    if (-not $exe -or -not (Test-Path $exe)) { return $false }
    try {
        $out = & $exe -c "import sys, struct; print(sys.version_info[0], sys.version_info[1], struct.calcsize('P') * 8)" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $out) { return $false }
        $v = ($out | Select-Object -Last 1).Trim().Split(' ')
        return ([int]$v[0] -eq 3 -and [int]$v[1] -ge 11 -and [int]$v[2] -eq 64)
    } catch { return $false }
}

function Find-Python {
    if ($Python) {
        if (Test-Python $Python) { return $Python }
        Fail "-Python $Python is not a 64-bit CPython 3.11 or newer"
    }
    $candidates = @()
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        foreach ($v in '3.13', '3.12', '3.14', '3.11') {
            try {
                $p = & $py.Source "-$v" -c "import sys; print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $p) { $candidates += ($p | Select-Object -Last 1).Trim() }
            } catch { }
        }
    }
    foreach ($name in 'python.exe', 'python3.exe') {
        Get-Command $name -All -ErrorAction SilentlyContinue | ForEach-Object {
            if ($_.Source -notlike '*\WindowsApps\*') { $candidates += $_.Source }   # skip the Store stub
        }
    }
    foreach ($v in '313', '312', '314', '311') {
        $candidates += (Join-Path $env:LOCALAPPDATA "Programs\Python\Python$v\python.exe")
    }
    foreach ($c in $candidates) { if (Test-Python $c) { return $c } }
    return $null
}

function Get-UserPath { [Environment]::GetEnvironmentVariable('Path', 'User') }

function Set-UserPathEntry([string]$dir, [bool]$add) {
    $parts = @((Get-UserPath) -split ';' | Where-Object { $_ -and ($_.TrimEnd('\') -ne $dir.TrimEnd('\')) })
    if ($add) { $parts += $dir }
    [Environment]::SetEnvironmentVariable('Path', ($parts -join ';'), 'User')
    $session = @($env:Path -split ';' | Where-Object { $_ -and ($_.TrimEnd('\') -ne $dir.TrimEnd('\')) })
    if ($add) { $session += $dir }
    $env:Path = $session -join ';'
}

if ($env:OS -ne 'Windows_NT') { Fail 'ignite-scribe reads the Windows desktop, so it installs on Windows 10 or 11 only.' }

Write-Host ""
Write-Host "  ignite-scribe  -  your desktop, readable by agents, redacted first" -ForegroundColor White

# ---- uninstall ---------------------------------------------------------------------------------------------------
if ($Uninstall) {
    Step 'Removing ignite-scribe'
    if (Test-Path $Exe) {
        Invoke-Quiet $Exe @('service', 'uninstall')
        Say 'background service removed'
    }
    Set-UserPathEntry $Bin $false
    foreach ($d in $Venv, $Bin) { if (Test-Path $d) { Remove-Item -Recurse -Force $d; Say "removed $d" } }
    if ($Purge) {
        if (Test-Path $InstallRoot) { Remove-Item -Recurse -Force $InstallRoot; Say "removed $InstallRoot (settings and history)" }
    } elseif (Test-Path $InstallRoot) {
        Say "kept your settings and screen history in $InstallRoot (add -Purge to delete them)"
    }
    Write-Host ""
    Write-Host '  Done. Remove the ignite-scribe entry from your agent''s MCP configuration too.' -ForegroundColor Green
    return
}

# ---- 1. Python ---------------------------------------------------------------------------------------------------
Step 'Python'
$Base = Find-Python
if (-not $Base) {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) { Fail 'no 64-bit Python 3.11+ found and winget is unavailable: install Python from python.org, then run this again.' }
    Say 'installing Python 3.13 for this user with winget...'
    & $winget.Source install -e --id Python.Python.3.13 --scope user --silent --accept-package-agreements --accept-source-agreements | Out-Null
    $Base = Find-Python
    if (-not $Base) { Fail 'Python was installed but not found; open a new PowerShell window and run this again.' }
}
Say "using $Base"

# ---- 2. environment ----------------------------------------------------------------------------------------------
Step "Environment in $Venv"
if (Test-Path $Exe) {
    Invoke-Quiet $Exe @('service', 'stop')         # the service runs from this environment: stop it to update
}
$VenvPy = Join-Path $Venv 'Scripts\python.exe'
if (-not (Test-Python $VenvPy)) {
    if (Test-Path $Venv) { Remove-Item -Recurse -Force $Venv }
    Invoke-Native $Base @('-m', 'venv', $Venv)
    Say 'created'
} else {
    Say 'exists, updating'
}
Invoke-Native $VenvPy @('-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', '--upgrade', 'pip')

# ---- 3. ignite-scribe --------------------------------------------------------------------------------------------
if ($Source) {
    $Spec = "$((Resolve-Path $Source).Path)[uia]"
    Step "ignite-scribe from $Source"
} else {
    $Spec = "ignite-scribe[uia] @ $Repo/archive/$Ref.zip"
    Step "ignite-scribe ($Ref) from GitHub"
}
Invoke-Native $VenvPy @('-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', '--upgrade', '--force-reinstall', $Spec)
$Version = (& $VenvPy -c "import ignite_scribe; print(ignite_scribe.__version__)").Trim()
Say "installed $Version"

# ---- 4. command ----------------------------------------------------------------------------------------------------
Step 'Command'
New-Item -ItemType Directory -Force -Path $Bin | Out-Null
Set-Content -Path (Join-Path $Bin 'ignite-scribe.cmd') -Encoding ASCII -Value "@`"%~dp0..\venv\Scripts\ignite-scribe.exe`" %*"
if ($NoPath) {
    Say "run it as $Bin\ignite-scribe.cmd"
} else {
    Set-UserPathEntry $Bin $true
    Say "ignite-scribe is on your PATH ($Bin); open a new terminal for other windows to see it"
}

# ---- 5. setup ------------------------------------------------------------------------------------------------------
Step 'Setup'
$SetupArgs = @('setup')
if ($Yes) { $SetupArgs += '--yes' }
if ($NoService) { $SetupArgs += '--no-service' }
Invoke-Native $Exe $SetupArgs

Write-Host ""
Write-Host "  ignite-scribe $Version is installed. Restart your agents to load it; change your answers any time with: ignite-scribe setup" -ForegroundColor Green
