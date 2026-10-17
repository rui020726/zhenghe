param(
  [Parameter(Mandatory = $true)][string]$ContractPath,
  [Parameter(Mandatory = $true)][string]$PermitPath,
  [Parameter(Mandatory = $true)][string]$SourceHead
)

$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = 'D:/project_env/python.exe'
$StartedUtc = [DateTime]::UtcNow
$EvidenceRoot = Join-Path $Here 'raw\qualification\R76_B2_R2_PARENT_CAPTURE_20261017'
$R2TaskStatePath = Join-Path $Here 'ROUND76_R2_TASK_STATE.json'
$EntryClaimPath = Join-Path $EvidenceRoot 'entry_attempt_claim.json'
$PrelaunchClaimPath = Join-Path $EvidenceRoot 'prelaunch_claim.json'
$StdoutPath = Join-Path $EvidenceRoot 'child.stdout.bin'
$StderrPath = Join-Path $EvidenceRoot 'child.stderr.bin'
$ReceiptPath = Join-Path $EvidenceRoot 'parent_receipt.json'
$ParentCommandLineArgs = [Environment]::GetCommandLineArgs()
$ChildArguments = @(
  (Join-Path $Here 'r76_b_entry.py'),
  '--execute',
  '--contract-path', $ContractPath,
  '--permit-path', $PermitPath,
  '--source-head', $SourceHead
)

$ChildPid = $null
$OsExitCode = $null
$TimedOut = $false
$ParentError = $null
$OwnEvidenceRootCreated = $false
$AttemptWritten = $false
$PrelaunchWritten = $false
$StdoutStream = $null
$StderrStream = $null
$Child = $null
$R2TaskStateSha256 = $null
$ContractSha256 = $null
$PermitSha256 = $null

function Write-StageJson {
  param([string]$Path, [object]$Value)
  $Directory = Split-Path -Parent $Path
  if (-not (Test-Path -LiteralPath $Directory)) { throw "stage JSON directory does not exist: $Directory" }
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

try {
  if (Test-Path -LiteralPath $EvidenceRoot) { throw "R2 evidence root already exists: $EvidenceRoot" }
  New-Item -ItemType Directory -Path $EvidenceRoot | Out-Null
  $OwnEvidenceRootCreated = $true

  $AttemptClaim = [ordered]@{
    kind = 'round76-r2-parent-entry-attempt/1'
    request_id = 'R76_B0_SOURCE_PREPARATION_20261016'
    task_id = 'R76_B_R71_NATIVE_RESTORE_QUALIFICATION_20261016'
    parameters = [ordered]@{
      contract_path = $ContractPath
      permit_path = $PermitPath
      source_head = $SourceHead
    }
    parent_pid = $PID
    parent_cwd = (Get-Location).Path
    parent_argv = $ParentCommandLineArgs
    started_utc = $StartedUtc.ToString('o')
    invocation_consumed = $true
    evidence_root = $EvidenceRoot
    timeout_ms = 1800000
  }
  Write-StageJson -Path $EntryClaimPath -Value $AttemptClaim
  $AttemptWritten = $true

  if (-not (Test-Path -LiteralPath $R2TaskStatePath)) { throw 'missing R2 task state' }
  if (-not (Test-Path -LiteralPath $ContractPath)) { throw "missing contract: $ContractPath" }
  if (-not (Test-Path -LiteralPath $PermitPath)) { throw "missing permit: $PermitPath" }

  $R2TaskStateSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $R2TaskStatePath).Hash.ToLower()
  $TaskState = Get-Content -Raw -LiteralPath $R2TaskStatePath | ConvertFrom-Json
  if ($TaskState.branch -ne 'exp/round76-constrained-early-native-mainline') { throw 'R2 task-state branch does not match' }
  if ($TaskState.source_head -ne $SourceHead) { throw 'R2 task-state SourceHead does not match the real argument' }
  if ($TaskState.current_stage -ne 'R2_B_CPU_PARENT_CAPTURE_RELEASED') { throw 'R2-B CPU parent capture is not stage-released' }
  foreach ($Gate in 'a0_source_review_passed',
                   'a1_pure_precheck_released',
                   'r2_b_cpu_capture_released') {
    if ($TaskState.gates.$Gate -ne $true) { throw "R2 gate is not true: $Gate" }
  }

  $ContractSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ContractPath).Hash.ToLower()
  $PermitSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $PermitPath).Hash.ToLower()
  $Contract = Get-Content -Raw -LiteralPath $ContractPath | ConvertFrom-Json
  $Permit = Get-Content -Raw -LiteralPath $PermitPath | ConvertFrom-Json
  if ($Contract.source_head -ne $SourceHead) { throw 'contract SourceHead does not match the real argument' }
  if ($Permit.source_head -ne $SourceHead) { throw 'permit SourceHead does not match the real argument' }
  if ($Contract.request_id -ne $TaskState.qualification_binding.request_id) { throw 'contract RequestId does not match R2 qualification identity' }
  if ($Permit.request_id -ne $TaskState.qualification_binding.request_id) { throw 'permit RequestId does not match R2 qualification identity' }
  if ($Contract.task_id -ne $TaskState.qualification_binding.task_id) { throw 'contract TaskId does not match R2 qualification identity' }
  if ($Permit.task_id -ne $TaskState.qualification_binding.task_id) { throw 'permit TaskId does not match R2 qualification identity' }

  $PrelaunchClaim = [ordered]@{
    kind = 'round76-r2-parent-prelaunch-claim/1'
    request_id = $TaskState.qualification_binding.request_id
    task_id = $TaskState.qualification_binding.task_id
    r2_task_state_path = $R2TaskStatePath
    r2_task_state_sha256 = $R2TaskStateSha256
    contract_sha256 = $ContractSha256
    permit_sha256 = $PermitSha256
    contract_request_id = $Contract.request_id
    contract_task_id = $Contract.task_id
    contract_source_head = $Contract.source_head
    permit_source_head = $Permit.source_head
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
  Write-StageJson -Path $PrelaunchClaimPath -Value $PrelaunchClaim
  $PrelaunchWritten = $true

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
  $ParentError = $_
  if ($Child -and -not $Child.HasExited) {
    try {
      $Child.Kill()
      $Child.WaitForExit()
    } catch {
    }
  }
  if ($Child -and $Child.HasExited) { $OsExitCode = $Child.ExitCode }
} finally {
  if ($OwnEvidenceRootCreated) {
    if ($StdoutStream) {
      try { $StdoutStream.Flush($true) } catch { }
      $StdoutStream.Dispose()
    }
    if ($StderrStream) {
      try { $StderrStream.Flush($true) } catch { }
      $StderrStream.Dispose()
    }
    $Receipt = [ordered]@{
      kind = 'round76-r2-parent-receipt/1'
      request_id = 'R76_B0_SOURCE_PREPARATION_20261016'
      task_id = 'R76_B_R71_NATIVE_RESTORE_QUALIFICATION_20261016'
      r2_task_state_path = $R2TaskStatePath
      r2_task_state_sha256 = $R2TaskStateSha256
      contract_sha256 = $ContractSha256
      permit_sha256 = $PermitSha256
      source_head = $SourceHead
      started_utc = $StartedUtc.ToString('o')
      ended_utc = [DateTime]::UtcNow.ToString('o')
      parent_error = if ($ParentError) { $ParentError.ToString() } else { $null }
      own_evidence_root_created = $OwnEvidenceRootCreated
      entry_attempt_claim_path = $EntryClaimPath
      attempt_written = $AttemptWritten
      prelaunch_claim_path = $PrelaunchClaimPath
      prelaunch_written = $PrelaunchWritten
      child_pid = $ChildPid
      child_executable = $Python
      child_arguments = $ChildArguments
      child_cwd = $Here
      os_exit_code = $OsExitCode
      timed_out = $TimedOut
      stdout_path = $StdoutPath
      stderr_path = $StderrPath
    }
    Write-StageJson -Path $ReceiptPath -Value $Receipt
  }
}

if ($ParentError -or $null -eq $OsExitCode) { exit 125 }
exit $OsExitCode
