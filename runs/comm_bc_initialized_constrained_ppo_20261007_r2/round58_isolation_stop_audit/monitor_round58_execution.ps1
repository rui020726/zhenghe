$dir = 'D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2\round58_isolation_stop_audit'
$raw = 'D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2\round58_gen3\raw'
Write-Output ('monitor started ' + (Get-Date -Format o))
while ($true) {
  $py = Get-Process python -ErrorAction SilentlyContinue
  if (-not $py) {
    $claims = Get-ChildItem ($raw + '\launch_claims') -Recurse -File -ErrorAction SilentlyContinue
    $complete = 0
    foreach ($c in $claims) {
      try { if ((Get-Content $c.FullName -Raw | ConvertFrom-Json).status -eq 'complete') { $complete++ } } catch {}
    }
    if ($complete -ge 1024) { Write-Output ('ALL 1024 CLAIMS COMPLETE ' + (Get-Date -Format o)); break }
    $n = (Get-ChildItem $dir -Filter 'ROUND58_GEN3_V20_EXECUTION_STDOUT_*.log' | Measure-Object).Count + 1
    $out = Join-Path $dir ('ROUND58_GEN3_V20_EXECUTION_STDOUT_{0}.log' -f $n)
    $err = Join-Path $dir ('ROUND58_GEN3_V20_EXECUTION_STDERR_{0}.log' -f $n)
    $args = @('-u', (Join-Path $dir 'r58_private_launcher_repaired.py'),
      '--authorization', (Join-Path $dir 'ROUND58_RESUME_EXECUTION_AUTHORIZATION_V20.json'),
      '--freeze', (Join-Path $dir 'ROUND58_RESUME_FREEZE_V20.json'),
      '--cumulative-quota', (Join-Path $dir 'ROUND58_GEN3_CUMULATIVE_QUOTA.json'),
      '--scene-audit', (Join-Path $dir 'ROUND58_GEN3_RESUME_SCENE_OCCUPANCY_RECHECK.json'),
      '--output-root', $raw, '--mode', 'recover_first_batch_gate')
    $p = Start-Process -FilePath 'D:\project_env\python.exe' -ArgumentList $args -WorkingDirectory $dir -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
    Write-Output ('launched ' + $p.Id + ' log=' + $n + ' complete_claims=' + $complete + ' at ' + (Get-Date -Format HH:mm:ss))
  }
  Start-Sleep -Seconds 40
}