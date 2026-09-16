@echo off
rem Replay-mode three-arm ablation. Usage: joint_replay.cmd <shard 0|1>
rem Per seed: ONE live PILOC recording run (this is the nocoord arm, 1000 steps,
rem writes streams/s<seed>.json), then two replays of that stream (rs0.25, rs0.5).
rem Windows 500/750/1000 are cut from the single 1000-step log by the metrics
rem module; the system is causal, so a prefix re-measures exactly a shorter run.
rem No Chinese literals; paths via %~dp0. Resume-safe: existing outputs skipped.
setlocal
set SHARD=%1
set OMP_NUM_THREADS=6
set MKL_NUM_THREADS=6
cd /d %~dp0..\..
set PY=D:\env\python.exe
if not exist "%PY%" set PY=python
set PILOC=runs\cur_v2rep05_s2\best\shared_mappo_model_21_3100.pth
set CTDE=runs\joint_eval_20260916\ctde_final_7_1.pth
set OUT=runs\joint_eval_20260916

if "%SHARD%"=="0" (set SEEDS=10000 10001 10002 10003 10004 10005 10006 10007 10008 10009 10010 10011 10012 10013 10014) else (set SEEDS=10015 10016 10017 10018 10019 10020 10021 10022 10023 10024 10025 10026 10027 10028 10029)

for %%s in (%SEEDS%) do (
  if not exist "%OUT%\rp_nocoord_s%%s.jsonl" (
    %PY% -m integrated_sar.joint_driver --piloc-checkpoint %PILOC% --task-checkpoint %CTDE% --actor-kv-mode tokens --steps 1000 --seed %%s --no-coordinator --quiet --record-stream %OUT%\streams\s%%s.json --output %OUT%\rp_nocoord_s%%s.jsonl
  )
  for %%r in (0.25 0.5) do (
    if not exist "%OUT%\rp_rs%%r_s%%s.jsonl" (
      %PY% -m integrated_sar.joint_driver --task-checkpoint %CTDE% --actor-kv-mode tokens --steps 1000 --seed %%s --replay-stream %OUT%\streams\s%%s.json --residual-scale %%r --quiet --output %OUT%\rp_rs%%r_s%%s.jsonl
    )
  )
)
echo DONE shard %SHARD%
endlocal
