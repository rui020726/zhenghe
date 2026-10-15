# ROUND75 S1E parent capture. PowerShell only: it launches the one authorized Python child.
$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Join-Path $Here "raw\s1e_probe"
New-Item -ItemType Directory -Force -Path $Root | Out-Null

$RequestId = "R75_S1E_FIXTURE_RECEIPT_REPAIR_20261015"
$Python = "D:\project_env\python.exe"
$ChildScript = Join-Path $Here "r75_s1e_fixture_receipt_repair.py"
$ChildArgv = @($ChildScript, "--run-b")
$StdoutPath = Join-Path $Root "phase_b_stdout.bin"
$StderrPath = Join-Path $Root "phase_b_stderr.bin"
$AttemptPath = Join-Path $Root "phase_b_parent_attempt.json"
$CapturePath = Join-Path $Root "phase_b_parent_capture.json"

foreach ($PathToCheck in @($StdoutPath, $StderrPath, $AttemptPath, $CapturePath)) {
    if (Test-Path -LiteralPath $PathToCheck) {
        throw "S1E parent refuses a second launch: $PathToCheck already exists"
    }
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "S1E parent did not find the authorized Python executable: $Python"
}

function Write-JsonCreateNew {
    param([string]$Path, [object]$Object)
    $Utf8 = New-Object System.Text.UTF8Encoding($false)
    $Bytes = $Utf8.GetBytes((ConvertTo-Json -InputObject $Object -Depth 12))
    $Stream = [System.IO.File]::Open($Path,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None)
    try {
        $Stream.Write($Bytes, 0, $Bytes.Length)
        $Stream.Flush($true)
    } finally {
        $Stream.Dispose()
    }
}

$Attempt = [ordered]@{
    kind = "round75-s1e-parent-attempt/1"
    request_id = $RequestId
    created_at = [DateTimeOffset]::UtcNow.ToString("o")
    parent_pid = $PID
    child_argv = $ChildArgv
    child_script = $ChildScript
    stdout_path = $StdoutPath
    stderr_path = $StderrPath
    launch_state = "issued"
}
Write-JsonCreateNew -Path $AttemptPath -Object $Attempt

$StartedAt = [DateTimeOffset]::UtcNow.ToString("o")
$Process = $null
$ChildPid = $null
$ChildStartTime = "missing"
$ChildExitTime = "missing"
$ExitCode = "missing"
$ActualChild = $false
$ParentError = $null

try {
    $Process = Start-Process -FilePath $Python `
        -ArgumentList $ChildArgv `
        -WorkingDirectory $Here `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath `
        -WindowStyle Hidden `
        -PassThru
    $ChildPid = $Process.Id
    $ChildStartTime = $Process.StartTime.ToString("o")
    $ActualChild = $true
    $null = $Process.WaitForExit()
    try { $ChildExitTime = $Process.ExitTime.ToString("o") } catch {
        $ChildExitTime = [DateTimeOffset]::UtcNow.ToString("o")
    }
    try { $ExitCode = $Process.ExitCode } catch { $ExitCode = "missing" }
} catch {
    $ParentError = $_.ToString()
}
$EndedAt = [DateTimeOffset]::UtcNow.ToString("o")

function BinaryArtifact {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return [ordered]@{path=$Path; exists=$false; bytes="missing"; sha256="missing"}
    }
    $Item = Get-Item -LiteralPath $Path
    $Hash = Get-FileHash -Algorithm SHA256 -LiteralPath $Path
    return [ordered]@{path=$Path; exists=$true; bytes=$Item.Length;
                      sha256=$Hash.Hash.ToLowerInvariant()}
}

$Capture = [ordered]@{
    kind = "round75-s1e-parent-capture/1"
    request_id = $RequestId
    parent_pid = $PID
    attempt_marker = $AttemptPath
    parent_launch_attempted = $true
    parent_launch_failed = [bool]$ParentError
    parent_error = $ParentError
    actual_python_child = $ActualChild
    python_started = $ActualChild
    child_pid = $ChildPid
    child_argv = $ChildArgv
    stdout_path = $StdoutPath
    stderr_path = $StderrPath
    stdout_artifact = BinaryArtifact -Path $StdoutPath
    stderr_artifact = BinaryArtifact -Path $StderrPath
    started_at = $StartedAt
    child_start_time = $ChildStartTime
    child_exit_time = $ChildExitTime
    ended_at = $EndedAt
    exit_code = $ExitCode
    retry_count = 0
    note = "One PowerShell Start-Process launch; stdout/stderr are read only from the redirected files as binary hashes."
}
Write-JsonCreateNew -Path $CapturePath -Object $Capture

if ($ParentError) { exit 2 }
if (-not $ActualChild) { exit 2 }
exit [int]$ExitCode
