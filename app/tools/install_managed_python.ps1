$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

# Always operate relative to this script. No path is passed from cmd.exe.
$ProjectDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $ProjectDir

if ([System.Environment]::Is64BitOperatingSystem -and -not [System.Environment]::Is64BitProcess) {
    $nativePowerShell = Join-Path $env:SystemRoot 'Sysnative\WindowsPowerShell\v1.0\powershell.exe'
    if (-not (Test-Path -LiteralPath $nativePowerShell -PathType Leaf)) {
        throw "64-bit Windows PowerShell was not found: $nativePowerShell"
    }
    Write-Host '32-bit launcher detected; restarting setup in 64-bit PowerShell...'
    & $nativePowerShell -NoLogo -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath
    exit $LASTEXITCODE
}

$UvVersion = '0.12.5'
$PythonVersion = '3.11.16'
$RuntimeDir = Join-Path $ProjectDir 'runtime'
$UvDir = Join-Path $RuntimeDir 'uv'
$UvExe = Join-Path $UvDir 'uv.exe'
$PythonInstallDir = Join-Path $RuntimeDir 'python'
$VenvDir = Join-Path $RuntimeDir 'venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$UvUrl = "https://releases.astral.sh/github/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
$UvSha256 = '4c4d49d8738847d9b71ba319e49a5688c93eac0fe6204b1df24e98528dddf39a'
$TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('managed_python_' + [guid]::NewGuid().ToString('N'))

try {
    [Net.ServicePointManager]::SecurityProtocol =
        [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch { }

$env:UV_PYTHON_INSTALL_DIR = $PythonInstallDir
$env:UV_PYTHON_PREFERENCE = 'only-managed'
$env:UV_PYTHON_INSTALL_BIN = '0'
$env:UV_NO_CONFIG = '1'
$env:UV_SYSTEM_CERTS = 'true'
$env:UV_HTTP_RETRIES = '5'
$env:UV_HTTP_TIMEOUT = '60'
$env:UV_CACHE_DIR = Join-Path $TempRoot 'uv-cache'

function Invoke-Native {
    param([Parameter(Mandatory=$true)][string]$FilePath, [string[]]$Arguments=@())
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
    }
}

function Test-Uv {
    if (-not (Test-Path -LiteralPath $UvExe -PathType Leaf)) { return $false }
    try {
        $text = (& $UvExe --version 2>$null) -join "`n"
        return $text -match ([regex]::Escape("uv $UvVersion"))
    } catch { return $false }
}

function Download-File {
    param([string]$Uri, [string]$OutFile, [int]$Attempts=4)
    $last = $null
    for ($i=1; $i -le $Attempts; $i++) {
        try {
            Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue
            Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $OutFile -TimeoutSec 120
            if (-not (Test-Path -LiteralPath $OutFile -PathType Leaf)) { throw 'download produced no file' }
            return
        } catch {
            $last = $_.Exception
            if ($i -lt $Attempts) {
                Write-Warning "Download failed: $($last.Message). Retrying..."
                Start-Sleep -Seconds ([Math]::Min(8, 2*$i))
            }
        }
    }
    throw "Download failed after $Attempts attempts: $($last.Message)"
}

function Find-ManagedPython {
    if (-not (Test-Path -LiteralPath $PythonInstallDir -PathType Container)) { return $null }
    $exact = @(Get-ChildItem -LiteralPath $PythonInstallDir -Filter 'python.exe' -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.Directory.Name -like 'cpython-3.11.16-windows-x86_64-none' })
    if ($exact.Count -eq 1) { return $exact[0].FullName }
    return $null
}

function Test-ManagedVenv {
    if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) { return $false }
    $base = Find-ManagedPython
    if (-not $base) { return $false }
    $cfg = Join-Path $VenvDir 'pyvenv.cfg'
    if (-not (Test-Path -LiteralPath $cfg -PathType Leaf)) { return $false }
    $text = [System.IO.File]::ReadAllText($cfg)
    if ($text -notmatch '(?im)^relocatable\s*=\s*true\s*$') { return $false }

    # Do not compare the raw `home` value with an absolute path. In a relocatable
    # uv environment the metadata can intentionally use relative paths. The
    # authoritative check is the interpreter that actually starts.
    $env:APP_EXPECTED_VENV = $VenvDir
    $env:APP_EXPECTED_BASE = Split-Path -Parent $base
    try {
        & $VenvPython -c "import os,pathlib,struct,sys; expected=pathlib.Path(os.environ['APP_EXPECTED_VENV']).resolve(); base=pathlib.Path(os.environ['APP_EXPECTED_BASE']).resolve(); assert sys.version_info[:3]==(3,11,16); assert struct.calcsize('P')==8; assert pathlib.Path(sys.prefix).resolve()==expected; assert pathlib.Path(sys.base_prefix).resolve()==base" *> $null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
    finally {
        Remove-Item Env:APP_EXPECTED_VENV -ErrorAction SilentlyContinue
        Remove-Item Env:APP_EXPECTED_BASE -ErrorAction SilentlyContinue
    }
}

$ok = $false
try {
    New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
    New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null

    Write-Host '[1/3] Preparing private uv...'
    if (-not (Test-Uv)) {
        if (Test-Path -LiteralPath $UvDir) { Remove-Item -LiteralPath $UvDir -Recurse -Force }
        New-Item -ItemType Directory -Path $UvDir -Force | Out-Null
        $zip = Join-Path $TempRoot 'uv.zip'
        Write-Host "Downloading uv $UvVersion..."
        Download-File -Uri $UvUrl -OutFile $zip
        $hash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hash -ne $UvSha256) { throw "uv archive SHA256 mismatch: $hash" }
        Expand-Archive -LiteralPath $zip -DestinationPath $UvDir -Force
        if (-not (Test-Uv)) { throw "uv $UvVersion was downloaded but could not be started." }
    }

    Write-Host '[2/3] Preparing private CPython 3.11.16...'
    Invoke-Native -FilePath $UvExe -Arguments @('python','install','--no-bin',$PythonVersion)
    $base = Find-ManagedPython
    if (-not $base) { throw "Managed CPython $PythonVersion was not found after installation." }

    Write-Host '[3/3] Checking local virtual environment...'
    if (-not (Test-ManagedVenv)) {
        if (Test-Path -LiteralPath $VenvDir) {
            Write-Host 'Rebuilding runtime\venv so it uses the private Python...'
            Remove-Item -LiteralPath $VenvDir -Recurse -Force
        }
        Invoke-Native -FilePath $UvExe -Arguments @('venv',$VenvDir,'--python',$base,'--relocatable')
    }
    if (-not (Test-ManagedVenv)) { throw 'The private Python environment did not pass validation.' }

    Write-Host "Private Python ready: $PythonVersion x64"
    Write-Host "Location: $PythonInstallDir"
    $ok = $true
}
finally {
    if (Test-Path -LiteralPath $TempRoot) {
        Remove-Item -LiteralPath $TempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if (-not $ok) { exit 1 }
exit 0
