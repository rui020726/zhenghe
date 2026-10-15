param([switch]$SelfTest = $false)

# ROUND75 S1F parent capture. PowerShell only: it launches the one authorized Python child.
$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Join-Path $Here "raw\s1f_probe"
New-Item -ItemType Directory -Force -Path $Root | Out-Null

$RequestId = "R75_S1F_GIT_IDENTITY_REPAIR_DRAFT_20261015"
$Python = "D:\project_env\python.exe"
$ChildScript = Join-Path $Here "r75_s1f_fixture_receipt_repair.py"
$ChildArgv = @($Python, $ChildScript, "--run-b")
$ChildArgumentList = @($ChildScript, "--run-b")
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

function BinaryArtifact {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return [ordered]@{path=$Path; exists=$false; bytes="missing"; sha256="missing"}
    }
    $Item = Get-Item -LiteralPath $Path
    $Sha256 = [System.Security.Cryptography.SHA256]::Create()
    $FileStream = [System.IO.File]::Open($Path,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::ReadWrite)
    try {
        $HashBytes = $Sha256.ComputeHash($FileStream)
    } finally {
        $FileStream.Dispose()
        $Sha256.Dispose()
    }
    $HashHex = [System.BitConverter]::ToString($HashBytes).Replace("-", "").ToLowerInvariant()
    return [ordered]@{path=$Path; exists=$true; bytes=$Item.Length;
                      sha256=$HashHex}
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

$PowerShellVersion = $PSVersionTable.PSVersion.ToString()
$OsDescription = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
$IsWindowsPlatform = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
    [System.Runtime.InteropServices.OSPlatform]::Windows)
$StdioImplementationBasis = if ($IsWindowsPlatform) {
    "Windows CreateProcess Start-Process branch supplies redirected files as process stdio file handles; no line reader is used."
} else {
    "Non-Windows Start-Process uses a line-oriented implementation; this parent must not be used for the S1E B capture."
}

if ($SelfTest) {
    $BinaryPath = Join-Path $Root ("parent_binary_selftest_" + [Guid]::NewGuid().ToString("N") + ".bin")
    $BinaryBytes = [byte[]]@(0x00, 0xFF, 0x00, 0xD8, 0x80, 0x00, 0x41, 0x00)
    $ExpectedSha = [System.BitConverter]::ToString(
        [System.Security.Cryptography.SHA256]::Create().ComputeHash($BinaryBytes)
    ).Replace("-", "").ToLowerInvariant()
    $BinaryStream = [System.IO.File]::Open($BinaryPath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None)
    try { $BinaryStream.Write($BinaryBytes, 0, $BinaryBytes.Length) } finally { $BinaryStream.Dispose() }
    $Observed = BinaryArtifact -Path $BinaryPath
    $Passed = ($Observed.bytes -eq $BinaryBytes.Length -and
               $Observed.sha256 -eq $ExpectedSha)
    $SelfTestResult = [ordered]@{
        kind = "round75-s1e-parent-binary-selftest/1"
        request_id = $RequestId
        powershell_version = $PowerShellVersion
        os_description = $OsDescription
        is_windows = $IsWindowsPlatform
        stdio_implementation_basis = $StdioImplementationBasis
        binary_path = $BinaryPath
        expected_bytes = $BinaryBytes.Length
        expected_sha256 = $ExpectedSha
        observed_artifact = $Observed
        passed = [bool]$Passed
        child_started = $false
    }
    Remove-Item -LiteralPath $BinaryPath -Force
    ConvertTo-Json -InputObject $SelfTestResult -Depth 8
    if ($Passed) { exit 0 } else { exit 1 }
}

$Head = (& git -C $Here rev-parse HEAD)
if ($LASTEXITCODE -ne 0) { throw "S1E parent could not read HEAD" }
$ParentHead = (& git -C $Here rev-parse HEAD^)
if ($LASTEXITCODE -ne 0) { throw "S1E parent could not read HEAD^" }
$Branch = (& git -C $Here rev-parse --abbrev-ref HEAD)
if ($LASTEXITCODE -ne 0) { throw "S1E parent could not read branch" }

$Attempt = [ordered]@{
    kind = "round75-s1e-parent-attempt/1"
    request_id = $RequestId
    created_at = [DateTimeOffset]::UtcNow.ToString("o")
    parent_pid = $PID
    head = $Head
    parent_head = $ParentHead
    branch = $Branch
    child_argv = $ChildArgv
    child_script = $ChildScript
    working_directory = $Here
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
$ChildMetadataError = $null
$WaitError = $null
$ExitReadError = $null

try {
    $Process = Start-Process -FilePath $Python `
        -ArgumentList $ChildArgumentList `
        -WorkingDirectory $Here `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath `
        -WindowStyle Hidden `
        -PassThru
    $ActualChild = $true
} catch {
    $ParentError = $_.ToString()
}

if ($ActualChild) {
    try { $ChildPid = $Process.Id } catch {
        $ChildMetadataError = $_.ToString()
    }
    try { $ChildStartTime = $Process.StartTime.ToString("o") } catch {
        $ChildMetadataError = $_.ToString()
    }
    try { $null = $Process.WaitForExit() } catch {
        $WaitError = $_.ToString()
        $ChildExitTime = "missing"
        $ExitCode = "missing"
    }
    if ($null -eq $WaitError) {
        try { $ChildExitTime = $Process.ExitTime.ToString("o") } catch {
            $ChildExitTime = "missing"
            $ExitReadError = $_.ToString()
        }
        try { $ExitCode = $Process.ExitCode } catch {
            $ExitCode = "missing"
            $ExitReadError = $_.ToString()
        }
    }
}
$EndedAt = [DateTimeOffset]::UtcNow.ToString("o")

$Capture = [ordered]@{
    kind = "round75-s1e-parent-capture/1"
    request_id = $RequestId
    parent_pid = $PID
    head = $Head
    parent_head = $ParentHead
    branch = $Branch
    working_directory = $Here
    attempt_marker = $AttemptPath
    parent_launch_attempted = $true
    parent_launch_failed = [bool]$ParentError
    parent_error = $ParentError
    actual_python_child = $ActualChild
    python_started = $ActualChild
    child_pid = $ChildPid
    child_argv = $ChildArgv
    child_metadata_error = $ChildMetadataError
    wait_error = $WaitError
    exit_read_error = $ExitReadError
    stdout_path = $StdoutPath
    stderr_path = $StderrPath
    stdout_artifact = BinaryArtifact -Path $StdoutPath
    stderr_artifact = BinaryArtifact -Path $StderrPath
    started_at = $StartedAt
    child_start_time = $ChildStartTime
    child_exit_time = $ChildExitTime
    ended_at = $EndedAt
    exit_code = $ExitCode
    powershell_version = $PowerShellVersion
    os_description = $OsDescription
    is_windows = $IsWindowsPlatform
    stdio_implementation_basis = $StdioImplementationBasis
    retry_count = 0
    note = "One PowerShell Start-Process launch; stdout/stderr are read only from the redirected files as binary hashes."
}
Write-JsonCreateNew -Path $CapturePath -Object $Capture

if ($ParentError) { exit 2 }
if (-not $ActualChild) { exit 2 }
exit [int]$ExitCode
