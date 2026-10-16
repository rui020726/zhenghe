param(
  [Parameter(Mandatory = $true)][string]$ContractPath,
  [Parameter(Mandatory = $true)][string]$PermitPath,
  [Parameter(Mandatory = $true)][string]$SourceHead
)

$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = 'D:/project_env/python.exe'
$StartedUtc = [DateTime]::UtcNow
$EvidenceRoot = Join-Path $Here 'raw\qualification\R76_B2_PARENT_CAPTURE_20261016'
$ChildPid = $null
$OsExitCode = $null
$TimedOut = $false

function Write-StageJson {
  param([string]$Path, [object]$Value)
  $Directory = Split-Path -Parent $Path
  New-Item -ItemType Directory -Force -Path $Directory | Out-Null
  $Temporary = Join-Path $Directory (".{0}.{1}.tmp" -f (Split-Path -Leaf $Path), $PID)
  $Json = $Value | ConvertTo-Json -Depth 24 -Compress
  $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Json)
  $Stream = [System.IO.File]::Open($Temporary,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::None)
  try {
    $Stream.Write($Bytes, 0, $Bytes.Length)
    $Stream.Flush($true)
  } finally {
    $Stream.Dispose()
  }
  if (Test-Path -LiteralPath $Path) { throw "stage JSON already exists: $Path" }
  Move-Item -LiteralPath $Temporary -Destination $Path
}

if (-not (Test-Path -LiteralPath $ContractPath)) { throw "missing contract: $ContractPath" }
if (-not (Test-Path -LiteralPath $PermitPath)) { throw "missing permit: $PermitPath" }
if (Test-Path -LiteralPath $EvidenceRoot) { throw "R76 B evidence root already exists: $EvidenceRoot" }

$Contract = Get-Content -Raw -LiteralPath $ContractPath | ConvertFrom-Json
$ContractSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ContractPath).Hash.ToLower()
$Permit = Get-Content -Raw -LiteralPath $PermitPath | ConvertFrom-Json
$PermitSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $PermitPath).Hash.ToLower()
if ($Contract.source_head -ne $SourceHead) { throw 'contract SourceHead does not match the real argument' }
if ($Permit.source_head -ne $SourceHead) { throw 'permit SourceHead does not match the real argument' }
New-Item -ItemType Directory -Path $EvidenceRoot | Out-Null

$ChildArguments = @(
  (Join-Path $Here 'r76_b_entry.py'),
  '--execute',
  '--contract-path', $ContractPath,
  '--permit-path', $PermitPath,
  '--source-head', $SourceHead
)
$Claim = [ordered]@{
  kind = 'round76-b-parent-prelaunch-claim/1'
  request_id = $Contract.request_id
  task_id = $Contract.task_id
  contract_sha256 = $ContractSha256
  permit_sha256 = $PermitSha256
  source_head = $SourceHead
  parent_pid = $PID
  parent_cwd = (Get-Location).Path
  child_executable = $Python
  child_arguments = $ChildArguments
  child_cwd = $Here
  evidence_root = $EvidenceRoot
  timeout_ms = 1800000
  started_utc = $StartedUtc.ToString('o')
}
$ClaimPath = Join-Path $EvidenceRoot 'prelaunch_claim.json'
Write-StageJson -Path $ClaimPath -Value $Claim

$StdoutPath = Join-Path $EvidenceRoot 'child.stdout.bin'
$StderrPath = Join-Path $EvidenceRoot 'child.stderr.bin'
$StdoutStream = $null
$StderrStream = $null
$Child = $null
$StartError = $null

try {
  $ProcessInfo = [System.Diagnostics.ProcessStartInfo]::new()
  $ProcessInfo.FileName = $Python
  foreach ($Argument in $ChildArguments) {
    $ProcessInfo.ArgumentList.Add($Argument)
  }
  $ProcessInfo.WorkingDirectory = $Here
  $ProcessInfo.UseShellExecute = $false
  $ProcessInfo.RedirectStandardOutput = $true
  $ProcessInfo.RedirectStandardError = $true
  $ProcessInfo.CreateNoWindow = $true

  $StdoutStream = [System.IO.File]::Open($StdoutPath,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::Read)
  $StderrStream = [System.IO.File]::Open($StderrPath,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::Read)
  $Child = [System.Diagnostics.Process]::Start($ProcessInfo)
  $ChildPid = $Child.Id
  $StdoutTask = $Child.StandardOutput.BaseStream.CopyToAsync($StdoutStream)
  $StderrTask = $Child.StandardError.BaseStream.CopyToAsync($StderrStream)
  $Completed = $Child.WaitForExit(1800000)
  if (-not $Completed) {
    $TimedOut = $true
    $Child.Kill()
    $Child.WaitForExit()
  }
  $StdoutTask.GetAwaiter().GetResult()
  $StderrTask.GetAwaiter().GetResult()
  $StdoutStream.Flush($true)
  $StderrStream.Flush($true)
  if ($Child.HasExited) { $OsExitCode = $Child.ExitCode }
} catch {
  $StartError = $_
  if ($Child -and -not $Child.HasExited) {
    try {
      $Child.Kill()
      $Child.WaitForExit()
    } catch {
    }
  }
  if ($Child -and $Child.HasExited) { $OsExitCode = $Child.ExitCode }
  throw
} finally {
  if ($StdoutStream) {
    try { $StdoutStream.Flush($true) } catch { }
    $StdoutStream.Dispose()
  }
  if ($StderrStream) {
    try { $StderrStream.Flush($true) } catch { }
    $StderrStream.Dispose()
  }
  $Receipt = [ordered]@{
    kind = 'round76-b-parent-receipt/1'
  request_id = $Contract.request_id
  task_id = $Contract.task_id
    contract_sha256 = $ContractSha256
    permit_sha256 = $PermitSha256
    source_head = $SourceHead
    started_utc = $StartedUtc.ToString('o')
    ended_utc = [DateTime]::UtcNow.ToString('o')
    child_pid = $ChildPid
    child_executable = $Python
    child_arguments = $ChildArguments
    child_cwd = $Here
    os_exit_code = $OsExitCode
    timed_out = $TimedOut
    start_error = if ($StartError) { $StartError.ToString() } else { $null }
    stdout_path = $StdoutPath
    stderr_path = $StderrPath
    prelaunch_claim_path = $ClaimPath
  }
  Write-StageJson -Path (Join-Path $EvidenceRoot 'parent_receipt.json') -Value $Receipt
}

if ($null -eq $OsExitCode) { exit 125 }
exit $OsExitCode
