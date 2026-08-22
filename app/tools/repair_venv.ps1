$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

# app\tools\repair_venv.ps1 -> project root is two levels above $PSScriptRoot.
$ProjectDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$VenvDir = Join-Path $ProjectDir 'runtime\venv'
$ConfigPath = Join-Path $VenvDir 'pyvenv.cfg'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$ManagedPythonRoot = Join-Path $ProjectDir 'runtime\python'
$UvExe = Join-Path $ProjectDir 'runtime\uv\uv.exe'

function Test-NativeCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @()
    )
    $previous = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 may surface native stderr as a terminating
        # NativeCommandError when ErrorActionPreference is Stop. A stale uv
        # trampoline after moving the folder is an expected validation failure.
        $ErrorActionPreference = 'Continue'
        & $FilePath @Arguments *> $null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
    finally {
        $ErrorActionPreference = $previous
    }
}

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @()
    )
    $previous = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $FilePath @Arguments
        $code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) {
        throw "Command failed with exit code $code`: $FilePath $($Arguments -join ' ')"
    }
}

function Test-VenvPython {
    param([Parameter(Mandatory = $true)][string]$ExpectedBaseHome)
    $env:APP_EXPECTED_VENV = $VenvDir
    $env:APP_EXPECTED_BASE = $ExpectedBaseHome
    try {
        return Test-NativeCommand -FilePath $VenvPython -Arguments @(
            '-c',
            "import os,pathlib,struct,sys; expected=pathlib.Path(os.environ['APP_EXPECTED_VENV']).resolve(); base=pathlib.Path(os.environ['APP_EXPECTED_BASE']).resolve(); assert sys.version_info[:3]==(3,11,16); assert struct.calcsize('P')==8; assert pathlib.Path(sys.prefix).resolve()==expected; assert pathlib.Path(sys.base_prefix).resolve()==base"
        )
    }
    finally {
        Remove-Item Env:APP_EXPECTED_VENV -ErrorAction SilentlyContinue
        Remove-Item Env:APP_EXPECTED_BASE -ErrorAction SilentlyContinue
    }
}

if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf) -or
    -not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    Write-Host 'Local runtime\venv was not found.'
    exit 2
}

$Candidates = @(Get-ChildItem -LiteralPath $ManagedPythonRoot -Filter 'python.exe' -File -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Directory.Name -like 'cpython-3.11.16-windows-x86_64-none' })
if ($Candidates.Count -ne 1) {
    Write-Host 'Private CPython 3.11.16 was not found. Run install.bat once.'
    exit 3
}

$BasePython = $Candidates[0].FullName
$BaseHome = $Candidates[0].Directory.FullName
if (-not (Test-NativeCommand -FilePath $BasePython -Arguments @('-c', "import struct,sys; assert sys.version_info[:3]==(3,11,16); assert struct.calcsize('P')==8"))) {
    Write-Host 'Private CPython itself could not be started. Run install.bat once.'
    exit 4
}

$ConfigText = [System.IO.File]::ReadAllText($ConfigPath)
$IsRelocatable = $ConfigText -match '(?im)^relocatable\s*=\s*true\s*$'
$VenvWorks = Test-VenvPython -ExpectedBaseHome $BaseHome

# A relocatable uv venv may intentionally store relative metadata in
# pyvenv.cfg. Do not infer a folder move by comparing the raw `home` string.
# If the interpreter starts with the expected sys.prefix/sys.base_prefix, the
# environment is healthy regardless of how that path is represented in cfg.
if (-not $IsRelocatable -or -not $VenvWorks) {
    if (-not (Test-Path -LiteralPath $UvExe -PathType Leaf)) {
        Write-Host 'Private uv is missing, so runtime\venv cannot be repaired in place. Run install.bat once.'
        exit 5
    }

    if (-not $IsRelocatable) {
        Write-Host 'Upgrading the local environment for safe folder moves...'
    }
    else {
        Write-Host 'Local environment path changed or its launcher needs repair; rebuilding launcher files in place...'
    }

    $env:UV_PYTHON_INSTALL_DIR = $ManagedPythonRoot
    $env:UV_PYTHON_PREFERENCE = 'only-managed'
    $env:UV_PYTHON_INSTALL_BIN = '0'
    $env:UV_NO_CONFIG = '1'
    try {
        # Preserve all installed packages while uv rewrites pyvenv.cfg and the
        # Windows launcher trampolines. Editing pyvenv.cfg alone is not enough.
        Invoke-NativeCommand -FilePath $UvExe -Arguments @(
            'venv', $VenvDir,
            '--python', $BasePython,
            '--relocatable',
            '--allow-existing',
            '--no-project',
            '--no-python-downloads'
        )
    }
    catch {
        Write-Host ('Automatic local-environment repair failed: ' + $_.Exception.Message)
        exit 6
    }

    if (-not $IsRelocatable) {
        Write-Host 'Local environment upgraded to relocatable mode.'
    }
    else {
        Write-Host 'Local environment launcher repaired without reinstalling packages.'
    }
}

if (-not (Test-VenvPython -ExpectedBaseHome $BaseHome)) {
    Write-Host 'Local Python environment validation failed. Run install.bat once.'
    exit 7
}

exit 0
