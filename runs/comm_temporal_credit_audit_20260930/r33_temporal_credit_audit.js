"use strict";

// ROUND33 read-only analysis. This script reads archived JSON/JSONL and .pt bytes, writes
// only ROUND33 audit outputs, and never imports the simulator or an optimizer.
// The GAE loop below is a direct JavaScript translation of the frozen cost_math.gae source;
// source-shape assertions bind the translation to the archived implementation.

const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const readline = require("node:readline");
const assert = require("node:assert/strict");

const WORKSPACE = path.resolve(__dirname, "..", "..");
const OUT = __dirname;
const R32 = path.join(WORKSPACE, "runs", "comm_actor_reduction_ablation_20261001");
const V2 = path.join(WORKSPACE, "runs", "comm_cost_v2_rerun_20260928");
const STAGE_A = path.join(WORKSPACE, "runs", "comm_cost_controller_stage_a_20260927");
const GAMMA = 1.0;
const LAMBDA_GAE = 0.95;
const B0 = 10000.0;
const FIXED_BATCHES = [0, 15, 31];
const ARMS = ["ALL_ROWS", "ACTIVE_ROWS"];
const SEEDS = [11, 22, 33];

function sha256(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

function fileSha(file) {
  return sha256(fs.readFileSync(file));
}

function jsonFile(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function stable(value) {
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${stable(value[k])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function stats(values) {
  const xs = values.filter(Number.isFinite).slice().sort((a, b) => a - b);
  if (!xs.length) return {n: 0, computable: false};
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const variance = xs.reduce((a, x) => a + (x - mean) ** 2, 0) / xs.length;
  const q = p => {
    const h = (xs.length - 1) * p;
    const lo = Math.floor(h), hi = Math.ceil(h);
    return xs[lo] + (xs[hi] - xs[lo]) * (h - lo);
  };
  return {
    n: xs.length,
    mean,
    abs_mean: xs.reduce((a, x) => a + Math.abs(x), 0) / xs.length,
    std_pop: Math.sqrt(variance),
    min: xs[0], q25: q(0.25), median: q(0.5), q75: q(0.75), max: xs[xs.length - 1],
    variance_zero: variance === 0,
    positive: xs.filter(x => x > 0).length,
    negative: xs.filter(x => x < 0).length,
    zero: xs.filter(x => x === 0).length,
  };
}

function signCounts(values) {
  return {
    n: values.length,
    positive: values.filter(x => x > 0).length,
    negative: values.filter(x => x < 0).length,
    zero: values.filter(x => x === 0).length,
  };
}

function ownerIndex(steps, tick) {
  let lo = 0, hi = steps.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (steps[mid] <= tick) lo = mid + 1;
    else hi = mid;
  }
  return lo - 1;
}

function sourceBoundGae(rewards, values, gamma, gaeLambda, terminal, bootstrap) {
  if (![rewards, values, terminal, bootstrap].every(x => Array.isArray(x) && x.length === rewards.length))
    throw new Error("GAE vectors must have the same explicit length");
  if (rewards.length === 0) return {advantages: [], returns: []};
  let nextAdvantage = 0;
  const advantages = Array(rewards.length).fill(0);
  for (let step = rewards.length - 1; step >= 0; step--) {
    const done = 1 - (terminal[step] ? 1 : 0);
    const delta = rewards[step] + gamma * bootstrap[step] * done - values[step];
    advantages[step] = delta + gamma * gaeLambda * done * nextAdvantage;
    nextAdvantage = advantages[step];
  }
  return {advantages, returns: advantages.map((a, i) => a + values[i])};
}

function runSyntheticGaeTests() {
  const close = (actual, expected, message) =>
    assert.ok(Math.abs(actual - expected) <= 1e-12, `${message}: ${actual} != ${expected}`);
  const tests = [];

  // Independent hand derivation: delta=[0.2, 0.4], A=[0.58, 0.4],
  // returns=[0.98, 1.0] for the frozen two-step terminal example.
  {
    const result = sourceBoundGae([0, 1], [0.4, 0.6], 1, 0.95, [false, true], [0.6, 0]);
    close(result.advantages[0], 0.58, "two-step A0");
    close(result.advantages[1], 0.4, "two-step A1");
    close(result.returns[0], 0.98, "two-step return0");
    close(result.returns[1], 1, "two-step return1");
    tests.push({name: "two_step_nonzero_value_terminal_closed_form", passed: true,
      expected: {advantages: [0.58, 0.4], returns: [0.98, 1]}});
  }

  // A late unit reward crosses a full-mask tape row. Only the reward changes,
  // so the earlier active-row difference must be lambda^2, not lambda^1.
  {
    const rewards = [0, 0, 0, 0];
    const plusOne = [0, 0, 0, 1];
    const values = [0.2, 0.1, 0.3, 0.4];
    const terminal = [false, false, false, true];
    const bootstrap = [values[1], values[2], values[3], 0];
    const base = sourceBoundGae(rewards, values, 1, 0.95, terminal, bootstrap);
    const changed = sourceBoundGae(plusOne, values, 1, 0.95, terminal, bootstrap);
    close(changed.advantages[0] - base.advantages[0], 0.95 ** 3, "early-row reward coefficient");
    close(changed.advantages[2] - base.advantages[2], 0.95, "near-row reward coefficient");
    tests.push({name: "late_reward_through_masked_tape_rows", passed: true,
      training_row_distance: 3, environment_step_distance: 6,
      expected_direct_coefficient: 0.95 ** 3});
  }

  // A nonterminal final transition retains its supplied next-value bootstrap.
  {
    const result = sourceBoundGae([0, 1], [0.4, 0.6], 1, 0.95, [false, false], [0.6, 0.5]);
    close(result.advantages[1], 0.9, "truncated final advantage");
    close(result.advantages[0], 1.055, "truncated earlier advantage");
    tests.push({name: "nonterminal_truncation_bootstrap", passed: true,
      expected: {advantages: [1.055, 0.9]}});
  }

  // A true terminal row blocks rewards in the following episode from flowing
  // backward, while its own reward still flows to earlier actions in its episode.
  {
    const values = [0.1, 0.2, 0.3];
    const terminal = [false, true, true];
    const bootstrap = [0.2, 0, 0];
    const before = sourceBoundGae([0, 0, 0], values, 1, 0.95, terminal, bootstrap);
    const after = sourceBoundGae([0, 0, 1], values, 1, 0.95, terminal, bootstrap);
    close(after.advantages[0] - before.advantages[0], 0, "post-terminal reward isolation");
    close(after.advantages[2] - before.advantages[2], 1, "second-episode reward local effect");
    const terminalReward = sourceBoundGae([0, 1], [0, 0], 1, 0.95, [false, true], [0, 0]);
    close(terminalReward.advantages[0], 0.95, "terminal reward backward transfer");
    tests.push({name: "terminal_boundary_stops_future_episode_recursion", passed: true,
      expected_post_terminal_transfer_to_prior_episode: 0});
  }

  return {engine: "source-bound JavaScript translation of frozen cost_math.gae; native Python/PyTorch unavailable",
    tests_run: tests.length, passed: tests.filter(x => x.passed).length, failed: 0, cases: tests};
}

function sum(values) {
  return values.reduce((a, b) => a + b, 0);
}

async function lifecycleSummary(file) {
  const summary = {
    file,
    records: 0,
    parseErrors: [],
    runConfig: null,
    discoveries: [],
    completions: [],
    finals: [],
  };
  if (!file || !fs.existsSync(file)) {
    summary.missing = true;
    return summary;
  }
  const rl = readline.createInterface({input: fs.createReadStream(file, {encoding: "utf8"}), crlfDelay: Infinity});
  for await (const line of rl) {
    if (!line.trim()) continue;
    let r;
    try { r = JSON.parse(line); }
    catch (error) { summary.parseErrors.push(String(error)); continue; }
    summary.records++;
    if (r.event_type === "run_config" && summary.runConfig === null) {
      summary.runConfig = {
        configuration: r.detail?.configuration ?? null,
        scenario: r.detail?.scenario ?? null,
        run_signature: r.detail?.run_signature ?? null,
      };
    }
    if (r.event_type === "detection") {
      summary.discoveries.push({
        target_id: r.target_id,
        source_event_id: r.source_event_id,
        step: Number(r.timestamp),
        discovered_by: r.discovered_by,
      });
    }
    if (r.event_type === "progress" && r.status === "completed" && r.detail?.expired === false) {
      summary.completions.push({target_id: r.target_id, step: Number(r.timestamp), source_event_id: r.source_event_id});
    }
    if (r.event_type === "final") {
      summary.finals.push({
        target_id: r.target_id,
        step: Number(r.timestamp),
        status: r.status,
        deadline: r.detail?.deadline,
        discovered_at: r.detail?.discovered_at,
        updated_at: r.detail?.updated_at,
        expired: r.detail?.expired,
      });
    }
  }
  summary.discoveries.sort((a, b) => a.step - b.step || String(a.target_id).localeCompare(String(b.target_id)));
  summary.completions.sort((a, b) => a.step - b.step || String(a.target_id).localeCompare(String(b.target_id)));
  summary.finals.sort((a, b) => String(a.target_id).localeCompare(String(b.target_id)));
  summary.discovery_signature = stable(summary.discoveries);
  summary.configured = !!summary.runConfig;
  return summary;
}

function replayTiming(row, tapes, life) {
  const out = {
    scenario_seed: row.scenario_seed,
    training_seed: row.training_seed,
    reduction: row.reduction,
    episode_index: row.episode_index,
    tape_records: tapes.length,
    task: {events: 0, units: 0, owner_rows: {active: 0, full_mask: 0, no_candidate: 0, unknown: 0, prefix: 0}, prefix_events: 0, prefix_units: 0},
    cost: {nonzero_ticks: 0, bytes: 0, owner_ticks: {active: 0, full_mask: 0, no_candidate: 0, unknown: 0, prefix: 0}, owner_bytes: {active: 0, full_mask: 0, no_candidate: 0, unknown: 0, prefix: 0}, prefix_ticks: 0, prefix_bytes: 0},
    row_classes: {active: 0, full_mask: 0, no_candidate: 0, unknown: 0},
    distances_to_nearest_active: [],
    target_completion_links: [],
    identity: {
      discoveries: life.discoveries.length,
      on_time_completions: life.completions.length,
      completion_same_timestamp_reward_matches: 0,
      completion_one_tick_later_reward_matches: 0,
      completion_without_one_tick_later_reward_match: 0,
      completion_with_prior_free_target_row: 0,
      completion_without_prior_free_target_row: 0,
      completion_reward_owner_after_event_decision_row: 0,
      completion_reward_owner_same_as_event_decision_row: 0,
      completion_event_has_exact_step_tape_row: 0,
      completion_event_owner_class: {active: 0, full_mask: 0, no_candidate: 0, unknown: 0, prefix: 0},
      completion_reward_owner_class: {active: 0, full_mask: 0, no_candidate: 0, unknown: 0, prefix: 0},
    },
    missing_fields: [],
    reconciliation: {},
  };
  const steps = [];
  const classes = [];
  const activeIndices = [];
  const firstFreeByTarget = new Map();
  for (let i = 0; i < tapes.length; i++) {
    const t = tapes[i];
    if (!Number.isInteger(Number(t.step))) out.missing_fields.push(`tape[${i}].step`);
    steps.push(Number(t.step));
    const hasTargets = Array.isArray(t.targets);
    const hasActionable = Array.isArray(t.actionable);
    if (!hasTargets || !hasActionable) {
      classes.push("unknown");
      out.row_classes.unknown++;
      if (!hasTargets) out.missing_fields.push(`tape[${i}].targets`);
      if (!hasActionable) out.missing_fields.push(`tape[${i}].actionable`);
      continue;
    }
    let kind;
    if (t.actionable.length > 0) {
      kind = "active";
      activeIndices.push(i);
      for (const id of t.actionable) if (!firstFreeByTarget.has(id)) {
        firstFreeByTarget.set(id, {index: i, step: steps[i], action: t.actions?.[id] ?? null});
      }
    } else if (t.targets.length > 0) kind = "full_mask";
    else kind = "no_candidate";
    classes.push(kind);
    out.row_classes[kind]++;
  }
  const taskStream = row.task_stream;
  if (!taskStream || typeof taskStream !== "object") out.missing_fields.push("task_stream");
  const taskEvents = [];
  if (taskStream && typeof taskStream === "object") {
    for (const [k, value] of Object.entries(taskStream)) {
      const tick = Number(k), amount = Number(value);
      if (!Number.isFinite(tick) || !Number.isFinite(amount)) {
        out.missing_fields.push(`task_stream[${k}]`);
        continue;
      }
      if (amount === 0) continue;
      const index = ownerIndex(steps, tick);
      const kind = index < 0 ? "prefix" : (classes[index] ?? "unknown");
      taskEvents.push({tick, amount, index, kind});
      out.task.events++;
      out.task.units += amount;
      out.task.owner_rows[kind] = (out.task.owner_rows[kind] ?? 0) + 1;
      if (kind === "prefix") {
        out.task.prefix_events++;
        out.task.prefix_units += amount;
      }
      let previousActive = -1;
      for (const activeIndex of activeIndices) {
        if (activeIndex <= index && steps[activeIndex] <= tick) previousActive = activeIndex;
        else if (activeIndex > index || steps[activeIndex] > tick) break;
      }
      if (previousActive < 0) out.distances_to_nearest_active.push(null);
      else out.distances_to_nearest_active.push({
        tape_row_distance: index - previousActive,
        env_step_distance: tick - steps[previousActive],
        reward_tick: tick,
        reward_owner_tape_row_1based: index + 1,
        nearest_prior_active_tape_row_1based: previousActive + 1,
        nearest_prior_active_step: steps[previousActive],
        owner_class: kind,
      });
    }
  }
  const costStream = row.cost_stream;
  if (!costStream || typeof costStream !== "object") out.missing_fields.push("cost_stream");
  if (costStream && typeof costStream === "object") {
    const actualTicks = Object.keys(costStream).map(Number).filter(Number.isInteger);
    const expected = Number(row.steps);
    if (Number.isFinite(expected) && actualTicks.length !== expected) {
      out.missing_fields.push(`cost_stream tick coverage ${actualTicks.length}/${expected}`);
    }
    for (const [k, record] of Object.entries(costStream)) {
      const tick = Number(k);
      if (!record || !Object.prototype.hasOwnProperty.call(record, "airtime_bytes")) {
        out.missing_fields.push(`cost_stream[${k}].airtime_bytes`);
        continue;
      }
      const bytes = Number(record.airtime_bytes);
      if (!Number.isFinite(bytes)) {
        out.missing_fields.push(`cost_stream[${k}].airtime_bytes nonfinite`);
        continue;
      }
      if (bytes === 0) continue;
      const index = ownerIndex(steps, tick);
      const kind = index < 0 ? "prefix" : (classes[index] ?? "unknown");
      out.cost.nonzero_ticks++;
      out.cost.bytes += bytes;
      out.cost.owner_ticks[kind] = (out.cost.owner_ticks[kind] ?? 0) + 1;
      out.cost.owner_bytes[kind] = (out.cost.owner_bytes[kind] ?? 0) + bytes;
      if (kind === "prefix") {
        out.cost.prefix_ticks++;
        out.cost.prefix_bytes += bytes;
      }
    }
  }

  const rewardUnitsByTick = new Map(taskEvents.map(e => [e.tick, e.amount]));
  const completionCountsAt = offset => {
    const counts = new Map();
    for (const c of life.completions) counts.set(c.step + offset, (counts.get(c.step + offset) ?? 0) + 1);
    let matched = 0;
    for (const [tick, count] of counts) matched += Math.min(Number(rewardUnitsByTick.get(tick) ?? 0), count);
    return matched;
  };
  out.identity.completion_same_timestamp_reward_matches = completionCountsAt(0);
  out.identity.completion_one_tick_later_reward_matches = completionCountsAt(1);
  out.identity.completion_without_one_tick_later_reward_match = Math.max(
    0, life.completions.length - out.identity.completion_one_tick_later_reward_matches);
  for (const c of life.completions) {
    const free = firstFreeByTarget.get(c.target_id);
    const eventOwnerIndex = ownerIndex(steps, c.step);
    const rewardTick = c.step + 1;
    const rewardOwnerIndex = ownerIndex(steps, rewardTick);
    const eventOwnerClass = eventOwnerIndex < 0 ? "prefix" : (classes[eventOwnerIndex] ?? "unknown");
    const rewardOwnerClass = rewardOwnerIndex < 0 ? "prefix" : (classes[rewardOwnerIndex] ?? "unknown");
    out.identity.completion_event_owner_class[eventOwnerClass]++;
    out.identity.completion_reward_owner_class[rewardOwnerClass]++;
    if (eventOwnerIndex >= 0 && steps[eventOwnerIndex] === c.step) out.identity.completion_event_has_exact_step_tape_row++;
    if (rewardOwnerIndex >= 0 && steps[rewardOwnerIndex] > c.step) {
      out.identity.completion_reward_owner_after_event_decision_row++;
    } else if (rewardOwnerIndex >= 0 && steps[rewardOwnerIndex] === c.step) {
      out.identity.completion_reward_owner_same_as_event_decision_row++;
    }
    if (free && eventOwnerIndex >= free.index && c.step >= free.step) {
      out.identity.completion_with_prior_free_target_row++;
      out.target_completion_links.push({
        target_id: c.target_id,
        first_free_decision_step: free.step,
        first_free_decision_tape_row_1based: free.index + 1,
        first_free_decision_action: free.action,
        physical_completion_event_step: c.step,
        physical_completion_event_owner_tape_row_1based: eventOwnerIndex + 1,
        recorded_task_reward_tick: rewardTick,
        recorded_task_reward_owner_tape_row_1based: rewardOwnerIndex + 1,
        recorded_reward_owner_class: rewardOwnerClass,
        recorded_reward_owner_step_after_physical_completion: rewardOwnerIndex >= 0 && steps[rewardOwnerIndex] > c.step,
        training_row_distance_to_physical_completion_owner: eventOwnerIndex - free.index,
        environment_step_distance_to_physical_completion: c.step - free.step,
        training_row_distance_to_recorded_reward_owner: rewardOwnerIndex - free.index,
        environment_step_distance_to_recorded_reward_tick: rewardTick - free.step,
        reward_stream_minus_physical_completion_ticks: rewardTick - c.step,
        same_target_identity: true,
        causal_claim: false,
      });
    } else {
      out.identity.completion_without_prior_free_target_row++;
    }
  }
  // Identity matches are reported by both event timestamp and the observed task-stream
  // offset. A one-tick match does not itself establish correct credit ownership.
  out.identity.reward_units = out.task.units;
  out.identity.completion_events = life.completions.length;
  out.reconciliation.task_units_vs_on_time = {
    task_stream_units: out.task.units,
    row_on_time: row.on_time,
    row_task_reward_total: row.task_reward_total,
    lifecycle_on_time_completion_events: life.completions.length,
    task_stream_equals_row_on_time: out.task.units === Number(row.on_time),
    task_stream_equals_lifecycle_completions: out.task.units === life.completions.length,
    lifecycle_completion_same_timestamp_matches: out.identity.completion_same_timestamp_reward_matches,
    lifecycle_completion_plus_one_tick_matches: out.identity.completion_one_tick_later_reward_matches,
  };
  return {summary: out, steps, classes, activeIndices, tapes, taskEvents, firstFreeByTarget};
}

function makeAdvantageEpisode(row, tapes) {
  if (!Array.isArray(tapes) || tapes.length === 0) throw new Error("empty selected tape");
  const steps = tapes.map(t => Number(t.step));
  if (steps.some(x => !Number.isInteger(x))) throw new Error("missing/noninteger tape time");
  for (let i = 1; i < steps.length; i++) if (!(steps[i] > steps[i - 1])) throw new Error("tape steps not strictly increasing");
  const rewardTask = Array(tapes.length).fill(0);
  const rewardCost = Array(tapes.length).fill(0);
  const taskStream = row.task_stream;
  const costStream = row.cost_stream;
  if (!taskStream || !costStream) throw new Error("missing reward stream");
  let prefixTask = 0, prefixCost = 0;
  const taskEvents = [];
  for (const [key, value] of Object.entries(taskStream)) {
    const tick = Number(key), amount = Number(value);
    if (!Number.isFinite(tick) || !Number.isFinite(amount)) throw new Error("invalid task reward field");
    if (amount === 0) continue;
    const i = ownerIndex(steps, tick);
    if (i < 0) prefixTask += amount;
    else rewardTask[i] += amount;
    taskEvents.push({tick, amount, ownerIndex: i});
  }
  const expectedTicks = Number(row.steps);
  if (Object.keys(costStream).length !== expectedTicks) throw new Error("cost stream is not complete; no zero-fill permitted");
  for (let tick = 1; tick <= expectedTicks; tick++) {
    if (!Object.prototype.hasOwnProperty.call(costStream, String(tick))) throw new Error(`cost stream tick ${tick} missing`);
    const costRecord = costStream[String(tick)];
    if (!costRecord || !Object.prototype.hasOwnProperty.call(costRecord, "airtime_bytes")) throw new Error(`cost stream airtime missing at ${tick}`);
    const bytes = Number(costRecord.airtime_bytes);
    if (!Number.isFinite(bytes)) throw new Error(`cost stream airtime invalid at ${tick}`);
    const i = ownerIndex(steps, tick);
    if (i < 0) prefixCost += bytes;
    else rewardCost[i] += bytes / B0;
  }
  const taskValues = tapes.map(t => {
    if (!Object.prototype.hasOwnProperty.call(t, "value") || !Number.isFinite(Number(t.value))) throw new Error("task value missing/nonfinite");
    return Number(t.value);
  });
  const costValues = tapes.map(t => {
    if (!Object.prototype.hasOwnProperty.call(t, "cost_value") || !Number.isFinite(Number(t.cost_value))) throw new Error("cost value missing/nonfinite");
    return Number(t.cost_value);
  });
  // The frozen caller sets only the last recorded decision row terminal, uses next row's
  // saved value for intermediate per-decision bootstraps, and zero at the final row.
  const terminal = tapes.map((_, i) => i === tapes.length - 1);
  const nextTask = tapes.map((_, i) => i + 1 < tapes.length ? taskValues[i + 1] : 0);
  const nextCost = tapes.map((_, i) => i + 1 < tapes.length ? costValues[i + 1] : 0);
  const taskTotal = sourceBoundGae(rewardTask, taskValues, GAMMA, LAMBDA_GAE, terminal, nextTask);
  const taskRewardPart = sourceBoundGae(rewardTask, Array(tapes.length).fill(0), GAMMA, LAMBDA_GAE, terminal, Array(tapes.length).fill(0));
  const taskCurrentValuePart = sourceBoundGae(Array(tapes.length).fill(0), taskValues, GAMMA, LAMBDA_GAE, terminal, Array(tapes.length).fill(0));
  const taskNextValuePart = sourceBoundGae(Array(tapes.length).fill(0), Array(tapes.length).fill(0), GAMMA, LAMBDA_GAE, terminal, nextTask);
  const costTotal = sourceBoundGae(rewardCost, costValues, GAMMA, LAMBDA_GAE, terminal, nextCost);
  const classes = tapes.map(t => Array.isArray(t.actionable) && t.actionable.length > 0 ? "active" : (Array.isArray(t.targets) && t.targets.length > 0 ? "full_mask" : (Array.isArray(t.targets) ? "no_candidate" : "unknown")));
  const firstRows = new Set();
  const firstFree = new Map();
  for (let i = 0; i < tapes.length; i++) {
    const t = tapes[i];
    if (!Array.isArray(t.actionable)) throw new Error("actionable field missing in selected tape");
    for (const target of t.actionable) if (!firstFree.has(target)) {
      firstFree.set(target, i);
      firstRows.add(i);
    }
  }
  const allIndices = tapes.map((_, i) => i);
  const transfer = [];
  for (const event of taskEvents) {
    if (event.ownerIndex < 0) continue;
    for (const i of allIndices) {
      if (classes[i] === "active" && i <= event.ownerIndex && steps[i] <= event.tick) {
        const distance = event.ownerIndex - i;
        transfer.push({row_distance: distance, env_step_distance: event.tick - steps[i], coefficient: (GAMMA * LAMBDA_GAE) ** distance, reward_amount: event.amount});
      }
    }
  }
  const active = [], fullMask = [], firstReport = [], all = [];
  const taskA = taskTotal.advantages;
  const costA = costTotal.advantages;
  const actor = taskA.map((a, i) => Number(row.lambda_at_rollout) * a - costA[i]);
  let closure = 0;
  for (let i = 0; i < tapes.length; i++) {
    const comps = taskRewardPart.advantages[i] + taskCurrentValuePart.advantages[i] + taskNextValuePart.advantages[i];
    closure = Math.max(closure, Math.abs(taskA[i] - comps));
    const d = {
      tape_row_1based: i + 1,
      environment_step: steps[i],
      class: classes[i],
      reward_task: rewardTask[i],
      reward_cost_normalized: rewardCost[i],
      value_task: taskValues[i],
      next_value_task: nextTask[i],
      reward_component: taskRewardPart.advantages[i],
      current_value_component: taskCurrentValuePart.advantages[i],
      next_value_bootstrap_component: taskNextValuePart.advantages[i],
      A_task: taskA[i],
      value_cost: costValues[i],
      next_value_cost: nextCost[i],
      A_cost: costA[i],
      lambda_times_A_task: Number(row.lambda_at_rollout) * taskA[i],
      A_actor: actor[i],
      return_task: taskTotal.returns[i],
      critic_error_value_minus_sample_return: taskValues[i] - taskTotal.returns[i],
      is_first_report_free_row: firstRows.has(i),
    };
    all.push(d);
    if (classes[i] === "active") active.push(d);
    if (classes[i] === "full_mask") fullMask.push(d);
    if (firstRows.has(i)) firstReport.push(d);
  }
  return {
    steps,
    classes,
    rewardTask,
    rewardCost,
    taskValues,
    costValues,
    nextTask,
    nextCost,
    terminal,
    taskTotal,
    taskRewardPart,
    taskCurrentValuePart,
    taskNextValuePart,
    costTotal,
    taskEvents,
    transfer,
    prefixTask,
    prefixCost,
    active,
    fullMask,
    firstReport,
    all,
    closure,
    totalTapeRows: tapes.length,
    row: {
      training_seed: row.training_seed,
      reduction: row.reduction,
      episode_index: row.episode_index,
      scenario_seed: row.scenario_seed,
      on_time: row.on_time,
      lambda_at_rollout: row.lambda_at_rollout,
    },
  };
}

function metricSummary(rows, field) {
  return stats(rows.map(r => r[field]));
}

function criticMetrics(rows) {
  if (!rows.length) return {n: 0, computable: false};
  const errors = rows.map(r => r.critic_error_value_minus_sample_return);
  const values = rows.map(r => r.value_task);
  const returns = rows.map(r => r.return_task);
  const errStats = stats(errors);
  const valueStats = stats(values);
  const returnStats = stats(returns);
  const covariance = sum(values.map((v, i) => (v - valueStats.mean) * (returns[i] - returnStats.mean))) / rows.length;
  const correlation = valueStats.variance_zero || returnStats.variance_zero ? null : covariance / (valueStats.std_pop * returnStats.std_pop);
  return {
    n: rows.length,
    value_prediction: valueStats,
    sample_return: returnStats,
    value_minus_return: errStats,
    bias: errStats.mean,
    mae: errors.reduce((a, e) => a + Math.abs(e), 0) / errors.length,
    rmse: Math.sqrt(errors.reduce((a, e) => a + e * e, 0) / errors.length),
    correlation_value_vs_return: correlation,
    correlation_computable: correlation !== null,
    scope: "sample diagnostic on saved rollout observations; not expected-value truth or independent generalization",
  };
}

async function readRows() {
  const dir = path.join(R32, "rows");
  const names = fs.readdirSync(dir).filter(n => n.endsWith(".json")).sort();
  const rows = [];
  const inventory = crypto.createHash("sha256");
  for (const name of names) {
    const full = path.join(dir, name);
    const buf = fs.readFileSync(full);
    inventory.update(name).update("\t").update(sha256(buf)).update("\n");
    const row = JSON.parse(buf.toString("utf8"));
    row.__file = full;
    row.__basename = name;
    rows.push(row);
  }
  return {rows, count: names.length, inventorySha256: inventory.digest("hex")};
}

function pointKey(arm, seed, batch) { return `${arm}_s${seed}/b${batch}`; }

function selectedEpisodes(checkpointIndex) {
  const out = new Map();
  for (const [runKey, group] of Object.entries(checkpointIndex)) {
    const match = runKey.match(/^(ALL_ROWS|ACTIVE_ROWS)_s(11|22|33)$/);
    if (!match) continue;
    for (const batch of FIXED_BATCHES) {
      const batchFile = group.batch_records[String(batch)];
      if (!batchFile) continue;
      const record = jsonFile(path.resolve(WORKSPACE, batchFile));
      for (const ep of record.episodes) out.set(`${runKey}/${ep}`, {runKey, batch, record, episodeIndex: ep});
    }
  }
  return out;
}

function compareStoredSummary(recomputed, stored, label) {
  const fields = ["n", "mean", "abs_mean", "std_pop", "min", "max"];
  const gaps = {};
  for (const f of fields) {
    if (typeof recomputed[f] !== "number" || typeof stored?.[f] !== "number") continue;
    gaps[f] = Math.abs(recomputed[f] - stored[f]);
  }
  return {label, fields_compared: Object.keys(gaps), max_abs_gap: Object.keys(gaps).length ? Math.max(...Object.values(gaps)) : null, gaps, stored_available: Object.keys(gaps).length > 0};
}

async function run() {
  const started = Date.now();
  const syntheticTestResults = runSyntheticGaeTests();
  const config = jsonFile(path.join(R32, "CONFIG_AND_MODEL_MANIFEST.json"));
  const index = jsonFile(path.join(R32, "CHECKPOINT_INDEX.json"));
  const mechanism = jsonFile(path.join(R32, "UPDATE_MECHANISM.json"));
  const round32Report = fs.readFileSync(path.join(R32, "ROUND32_REPORT.md"), "utf8");
  void round32Report;
  const {rows, count: rowFileCount, inventorySha256: rowInventorySha256} = await readRows();
  const trainRows = rows.filter(r => r.kind === "train").sort((a, b) =>
    String(a.reduction).localeCompare(String(b.reduction)) || Number(a.training_seed) - Number(b.training_seed) || Number(a.episode_index) - Number(b.episode_index));
  const selected = selectedEpisodes(index);
  const rowByEpisode = new Map();
  for (const row of trainRows) rowByEpisode.set(`${row.reduction}_s${row.training_seed}/${row.episode_index}`, row);
  const training = {
    expectedEpisodes: ARMS.length * SEEDS.length * 128,
    observedRows: trainRows.length,
    rowFiles: 0,
    tapeFiles: 0,
    missing: [],
    parseErrors: [],
    totalTapeRows: 0,
    totalActiveRows: 0,
    totalFullMaskRows: 0,
    totalNoCandidateRows: 0,
    totalUnknownClassRows: 0,
    totalTaskRewardEvents: 0,
    totalTaskRewardUnits: 0,
    taskRewardOwner: {active: {events: 0, units: 0}, full_mask: {events: 0, units: 0}, no_candidate: {events: 0, units: 0}, unknown: {events: 0, units: 0}, prefix: {events: 0, units: 0}},
    totalNonzeroCostTicks: 0,
    totalCostBytes: 0,
    costOwner: {active: {ticks: 0, bytes: 0}, full_mask: {ticks: 0, bytes: 0}, no_candidate: {ticks: 0, bytes: 0}, unknown: {ticks: 0, bytes: 0}, prefix: {ticks: 0, bytes: 0}},
    nearestActiveRowDistance: [],
    targetIdentity: {
      discovered: 0,
      on_time_completion_events: 0,
      completion_reward_units: 0,
      completion_same_timestamp_reward_matches: 0,
      completion_one_tick_later_reward_matches: 0,
      completion_without_one_tick_later_reward_match: 0,
      completed_with_prior_free_decision: 0,
      completed_without_prior_free_decision: 0,
      uncompleted_or_not_on_time: 0,
      no_first_free_row_for_completed_target: 0,
      reward_owner_after_event_decision_row: 0,
      reward_owner_same_as_event_decision_row: 0,
      completion_event_has_exact_step_tape_row: 0,
      completion_event_owner_class: {active: 0, full_mask: 0, no_candidate: 0, unknown: 0, prefix: 0},
      completion_reward_owner_class: {active: 0, full_mask: 0, no_candidate: 0, unknown: 0, prefix: 0},
      links: [],
    },
    rewardStreamReconciliationFailures: [],
    perRun: {},
    tapeInventorySha256: null,
  };
  const tapeInventory = crypto.createHash("sha256");
  const selectedData = new Map();
  const lifecycleCache = new Map();
  const episodeSummaries = [];
  const runAccum = new Map();
  const trainingRunKeys = new Set(trainRows.map(r => `${r.reduction}_s${r.training_seed}`));
  for (let i = 0; i < trainRows.length; i++) {
    const row = trainRows[i];
    const runKey = `${row.reduction}_s${row.training_seed}`;
    const key = `${runKey}/${row.episode_index}`;
    const tapePath = row.artefacts?.tape;
    const lifecyclePath = row.artefacts?.lifecycle;
    if (!tapePath || !fs.existsSync(tapePath)) {
      training.missing.push({row: row.__basename, tape: tapePath ?? null});
      continue;
    }
    let tapeBuffer;
    try { tapeBuffer = fs.readFileSync(tapePath); }
    catch (e) { training.missing.push({row: row.__basename, tape: tapePath, error: String(e)}); continue; }
    const tapeHash = sha256(tapeBuffer);
    tapeInventory.update(path.basename(tapePath)).update("\t").update(tapeHash).update("\n");
    const tapes = [];
    try {
      for (const line of tapeBuffer.toString("utf8").split(/\r?\n/)) if (line.trim()) tapes.push(JSON.parse(line));
    } catch (e) {
      training.parseErrors.push({row: row.__basename, tape: tapePath, error: String(e)});
      continue;
    }
    if (!lifecyclePath || !fs.existsSync(lifecyclePath)) {
      training.missing.push({row: row.__basename, lifecycle: lifecyclePath ?? null});
    }
    const life = await lifecycleSummary(lifecyclePath);
    lifecycleCache.set(lifecyclePath, life);
    const timing = replayTiming(row, tapes, life);
    const s = timing.summary;
    training.rowFiles++;
    training.tapeFiles++;
    training.totalTapeRows += tapes.length;
    training.totalActiveRows += s.row_classes.active;
    training.totalFullMaskRows += s.row_classes.full_mask;
    training.totalNoCandidateRows += s.row_classes.no_candidate;
    training.totalUnknownClassRows += s.row_classes.unknown;
    training.totalTaskRewardEvents += s.task.events;
    training.totalTaskRewardUnits += s.task.units;
    training.totalNonzeroCostTicks += s.cost.nonzero_ticks;
    training.totalCostBytes += s.cost.bytes;
    for (const k of Object.keys(training.taskRewardOwner)) {
      training.taskRewardOwner[k].events += s.task.owner_rows[k] ?? 0;
    }
    // Accumulate reward units by owner class directly from per-event details.
    for (const event of timing.taskEvents) {
      training.taskRewardOwner[event.kind].units += event.amount;
      if (event.kind !== "prefix") training.taskRewardOwner[event.kind].events += 0; // event count was added above
    }
    for (const k of Object.keys(training.costOwner)) {
      training.costOwner[k].ticks += s.cost.owner_ticks[k] ?? 0;
      training.costOwner[k].bytes += s.cost.owner_bytes[k] ?? 0;
    }
    training.nearestActiveRowDistance.push(...s.distances_to_nearest_active);
    training.targetIdentity.discovered += s.identity.discoveries;
    training.targetIdentity.on_time_completion_events += s.identity.on_time_completions;
    training.targetIdentity.uncompleted_or_not_on_time += Math.max(0, s.identity.discoveries - s.identity.on_time_completions);
    training.targetIdentity.completion_reward_units += s.identity.reward_units;
    training.targetIdentity.completion_same_timestamp_reward_matches += s.identity.completion_same_timestamp_reward_matches;
    training.targetIdentity.completion_one_tick_later_reward_matches += s.identity.completion_one_tick_later_reward_matches;
    training.targetIdentity.completion_without_one_tick_later_reward_match += s.identity.completion_without_one_tick_later_reward_match;
    training.targetIdentity.reward_owner_after_event_decision_row += s.identity.completion_reward_owner_after_event_decision_row;
    training.targetIdentity.reward_owner_same_as_event_decision_row += s.identity.completion_reward_owner_same_as_event_decision_row;
    training.targetIdentity.completion_event_has_exact_step_tape_row += s.identity.completion_event_has_exact_step_tape_row;
    for (const k of Object.keys(training.targetIdentity.completion_event_owner_class)) {
      training.targetIdentity.completion_event_owner_class[k] += s.identity.completion_event_owner_class[k] ?? 0;
      training.targetIdentity.completion_reward_owner_class[k] += s.identity.completion_reward_owner_class[k] ?? 0;
    }
    training.targetIdentity.completed_with_prior_free_decision += s.identity.completion_with_prior_free_target_row;
    training.targetIdentity.completed_without_prior_free_decision += s.identity.completion_without_prior_free_target_row;
    training.targetIdentity.no_first_free_row_for_completed_target += s.identity.completion_without_prior_free_target_row;
    training.targetIdentity.links.push(...s.target_completion_links.map(x => ({...x, run: runKey, episode_index: row.episode_index, scenario_seed: row.scenario_seed})));
    if (!s.reconciliation.task_units_vs_on_time.task_stream_equals_row_on_time || !s.reconciliation.task_units_vs_on_time.task_stream_equals_lifecycle_completions) {
      training.rewardStreamReconciliationFailures.push({row: row.__basename, reconciliation: s.reconciliation});
    }
    if (!runAccum.has(runKey)) runAccum.set(runKey, {episodes: 0, tape_rows: 0, active_rows: 0, full_mask_rows: 0, no_candidate_rows: 0, task_events: 0, task_units: 0, task_owner: {}, cost_ticks: 0, cost_bytes: 0, task_distances: [], missing_rows: 0});
    const ra = runAccum.get(runKey);
    ra.episodes++;
    ra.tape_rows += tapes.length;
    ra.active_rows += s.row_classes.active;
    ra.full_mask_rows += s.row_classes.full_mask;
    ra.no_candidate_rows += s.row_classes.no_candidate;
    ra.task_events += s.task.events;
    ra.task_units += s.task.units;
    ra.cost_ticks += s.cost.nonzero_ticks;
    ra.cost_bytes += s.cost.bytes;
    ra.task_distances.push(...s.distances_to_nearest_active.filter(Boolean).map(x => x.tape_row_distance));
    for (const [owner, n] of Object.entries(s.task.owner_rows)) ra.task_owner[owner] = (ra.task_owner[owner] ?? 0) + n;
    if (selected.has(key)) selectedData.set(key, {row, tapes, life, timing, tapeHash});
    episodeSummaries.push({row: row.__basename, run: runKey, episode_index: row.episode_index, scenario_seed: row.scenario_seed, tape_sha256: tapeHash, timing: s});
    if ((i + 1) % 100 === 0) console.log(`ROUND33 training tape analysis ${i + 1}/${trainRows.length}`);
  }
  training.tapeInventorySha256 = tapeInventory.digest("hex");
  for (const [key, r] of runAccum) {
    training.perRun[key] = {
      ...r,
      task_row_distance_to_nearest_active: stats(r.task_distances),
      active_row_share: r.tape_rows ? r.active_rows / r.tape_rows : null,
      full_mask_row_share: r.tape_rows ? r.full_mask_rows / r.tape_rows : null,
    };
    delete training.perRun[key].task_distances;
  }
  training.nearestActiveRowDistance = {
    events_with_prior_active: training.nearestActiveRowDistance.filter(Boolean).length,
    events_without_prior_active: training.nearestActiveRowDistance.filter(x => x === null).length,
    tape_row_distance: stats(training.nearestActiveRowDistance.filter(Boolean).map(x => x.tape_row_distance)),
    environment_step_distance: stats(training.nearestActiveRowDistance.filter(Boolean).map(x => x.env_step_distance)),
    definition: "decision-row owner index minus latest preceding active row index; environment distance is reward tick minus that row's relay/tape step; descriptive only, not causal",
  };
  training.rowClassShares = {
    active: training.totalActiveRows / training.totalTapeRows,
    full_mask: training.totalFullMaskRows / training.totalTapeRows,
    no_candidate: training.totalNoCandidateRows / training.totalTapeRows,
    unknown: training.totalUnknownClassRows / training.totalTapeRows,
    sum: (training.totalActiveRows + training.totalFullMaskRows + training.totalNoCandidateRows + training.totalUnknownClassRows) / training.totalTapeRows,
  };
  training.files = {row_json_files: rowFileCount, training_rows_with_tape: training.rowFiles, training_tape_files: training.tapeFiles};
  training.lifecycleIdentityScope = "all readable training episodes; each lifecycle log summarized by stable target ID; no success/failure selection";
  training.episodeSummaries = episodeSummaries;

  const pointReports = [];
  const criticReport = {
    kind: "fixed-sample task critic diagnostic",
    gamma: GAMMA,
    lambda_GAE: LAMBDA_GAE,
    lambda_dual_semantics: "per-batch constrained objective multiplier; distinct from lambda_GAE",
    B0_bytes: B0,
    data_source: "saved training row reward streams + original on-policy JSONL tapes; tape value/cost_value from rollout parameter state identified in ROUND32 batch/checkpoint index",
    checkpoint_validation: [],
    source_batch_likelihood_identity: [],
    provenance_conflicts: [],
    stopped_points: [],
    sample_diagnostics: [],
    status: "not_an_expectation_value_or_independent_generalization_test",
    pytorch_checkpoint_forward_replayed: false,
    unverified_reason: "No Python/PyTorch executable is available in this shell. All 18 .pt file SHA256 values match CHECKPOINT_INDEX, but only the six batch-0 parameter/lambda identities match the current batch records; twelve batch-15/31 identities conflict. Recorded tape values are used without re-forwarding the serialized weights.",
  };
  const componentPoints = [];
  const allSelectedTransfer = [];
  let transferBatchResults = [];
  for (const runKey of Object.keys(index).sort()) {
    const match = runKey.match(/^(ALL_ROWS|ACTIVE_ROWS)_s(11|22|33)$/);
    if (!match) continue;
    const arm = match[1], seed = Number(match[2]), group = index[runKey];
    for (const batchIndex of FIXED_BATCHES) {
      const point = pointKey(arm, seed, batchIndex);
      const recordPath = group.batch_records[String(batchIndex)];
      const record = jsonFile(path.resolve(WORKSPACE, recordPath));
      const cpMeta = group.pre_update_checkpoints[String(batchIndex)];
      const cpPath = path.resolve(WORKSPACE, cpMeta.path);
      const cpExists = fs.existsSync(cpPath);
      const cpHash = cpExists ? fileSha(cpPath) : null;
      const expectedEpisodes = record.episodes;
      const episodeRows = [];
      const selectedTapes = [];
      let missing = false;
      for (let j = 0; j < expectedEpisodes.length; j++) {
        const ep = expectedEpisodes[j];
        const row = rowByEpisode.get(`${runKey}/${ep}`);
        const chosen = selectedData.get(`${runKey}/${ep}`);
        if (!row || !chosen) { missing = true; break; }
        if (Number(record.scenarios[j]) !== Number(row.scenario_seed)) throw new Error(`${point}: scenario/episode identity mismatch at ${j}`);
        selectedTapes.push(chosen.tapes);
        episodeRows.push(row);
      }
      const sampleCount = selectedTapes.reduce((a, tapes) => a + tapes.length, 0);
      const storedPoint = mechanism.runs?.[runKey]?.[`b${batchIndex}`];
      const storedMechanismCounts = {
        A_task_n: Number.isFinite(storedPoint?.advantages?.A_task?.n) ? Number(storedPoint.advantages.A_task.n) : null,
        recorded_samples: Number.isFinite(storedPoint?.recorded_during_training?.samples_recorded_vs_recomputed?.recorded) ? Number(storedPoint.recorded_during_training.samples_recorded_vs_recomputed.recorded) : null,
        mechanism_recomputed_samples: Number.isFinite(storedPoint?.recorded_during_training?.samples_recorded_vs_recomputed?.recomputed) ? Number(storedPoint.recorded_during_training.samples_recorded_vs_recomputed.recomputed) : null,
      };
      const trainingRecordSamples = Number(record.update?.samples);
      const currentSourcesAgree = !missing && sampleCount === trainingRecordSamples &&
        storedMechanismCounts.A_task_n === trainingRecordSamples &&
        storedMechanismCounts.recorded_samples === trainingRecordSamples &&
        storedMechanismCounts.mechanism_recomputed_samples === trainingRecordSamples &&
        storedPoint?.recorded_during_training?.samples_recorded_vs_recomputed?.match === true;
      const batchPointIdentityAgrees = cpExists && cpHash === cpMeta.sha256 &&
        cpMeta.parameter_sha256 === record.controller_parameters_before?.parameter_sha256 &&
        Math.abs(Number(cpMeta.lambda) - Number(record.lambda_at_rollout)) <= 1e-12 &&
        (record.pre_update_checkpoint_saved_for_batches ?? []).includes(batchIndex);
      const identityChecks = {
        point,
        index_recorded: !!cpMeta,
        checkpoint_exists: cpExists,
        checkpoint_file_sha_matches_index: cpHash === cpMeta.sha256,
        parameter_sha_matches_batch_record: cpMeta.parameter_sha256 === record.controller_parameters_before?.parameter_sha256,
        batch_has_checkpointed_batch_index: (record.pre_update_checkpoint_saved_for_batches ?? []).includes(batchIndex),
        lambda_matches: Math.abs(Number(cpMeta.lambda) - Number(record.lambda_at_rollout)) <= 1e-12,
        checkpoint_index_claims_batch_before_match: cpMeta.batch_record_parameters_before_matches === true,
        batch_checkpoint_metadata_matches: record.pre_update_checkpoint?.sha256 ? record.pre_update_checkpoint.sha256 === cpMeta.sha256 : null,
        episode_list: record.episodes,
        scenario_list: record.scenarios,
        batch_samples_recorded: trainingRecordSamples,
        lambda_dual: Number(record.lambda_at_rollout),
        current_tape_line_count: missing ? null : sampleCount,
        stored_update_mechanism_sample_counts: storedMechanismCounts,
        current_batch_tape_and_saved_mechanism_counts_agree: currentSourcesAgree,
        current_checkpoint_and_batch_record_identity_agree: batchPointIdentityAgrees,
        checkpoint_path: cpMeta.path,
        checkpoint_sha256_index: cpMeta.sha256,
        checkpoint_sha256_observed: cpHash,
        parameter_sha256: cpMeta.parameter_sha256,
        loaded_or_forwarded: false,
      };
      criticReport.checkpoint_validation.push(identityChecks);
      const storedLikelihood = record.update?.likelihood_alignment;
      const likelihood = {
        point,
        stored_available: !!storedLikelihood,
        stored_passed: storedLikelihood?.passed ?? null,
        stored_samples: storedLikelihood?.samples ?? null,
        tape_samples: sampleCount,
        sample_count_matches: storedLikelihood ? Number(storedLikelihood.samples) === sampleCount : false,
        max_abs_gap: storedLikelihood?.max_abs_gap ?? null,
        native_likelihood_recomputed_this_round: false,
      };
      criticReport.source_batch_likelihood_identity.push(likelihood);
      if (missing) {
        pointReports.push({point, availability: "missing selected source row/tape"});
        transferBatchResults.push({point, availability: "stopped_missing_selected_source_row_or_tape"});
        criticReport.stopped_points.push({point, reason: "missing selected source row/tape"});
        continue;
      }
      if (!currentSourcesAgree || !batchPointIdentityAgrees) {
        const conflict = {
          point,
          reason: "At least one fixed-point provenance gate fails: checkpoint parameter/lambda identity or the saved UPDATE_MECHANISM sample counts disagree with the current batch record/tape set. Per the audit stop rule, do not choose one source silently.",
          batch_record_samples: trainingRecordSamples,
          current_tape_line_count: sampleCount,
          saved_update_mechanism_sample_counts: storedMechanismCounts,
          checkpoint_file_sha_matches_index: identityChecks.checkpoint_file_sha_matches_index,
          checkpoint_parameter_sha_matches_batch: identityChecks.parameter_sha_matches_batch_record,
          checkpoint_lambda_matches_batch: identityChecks.lambda_matches,
          saved_checkpoint_identity_claim_matches_current_record: identityChecks.checkpoint_index_claims_batch_before_match,
          saved_likelihood_gate_passed: likelihood.stored_passed,
          batch_record_tape_count_matches_likelihood_gate: likelihood.sample_count_matches,
        };
        criticReport.provenance_conflicts.push(conflict);
        criticReport.stopped_points.push({point, reason: conflict.reason});
        pointReports.push({point, availability: "stopped_fixed_batch_provenance_conflict", provenance: conflict});
        transferBatchResults.push({point, availability: "stopped_fixed_batch_provenance_conflict", provenance: conflict});
        continue;
      }
      const episodeMaths = episodeRows.map((row, i) => makeAdvantageEpisode(row, selectedTapes[i]));
      const rowsAll = episodeMaths.flatMap(x => x.all);
      const rowsActive = episodeMaths.flatMap(x => x.active);
      const rowsFull = episodeMaths.flatMap(x => x.fullMask);
      const rowsFirst = episodeMaths.flatMap(x => x.firstReport);
      const prefixTask = sum(episodeMaths.map(x => x.prefixTask));
      const prefixCost = sum(episodeMaths.map(x => x.prefixCost));
      const component = {
        point,
        batch: batchIndex,
        training_seed: seed,
        reduction: arm,
        lambda_dual: Number(record.lambda_at_rollout),
        checkpoint_parameter_sha256: cpMeta.parameter_sha256,
        checkpoint_file_sha256_verified: cpHash === cpMeta.sha256,
        row_tape_parameter_link: cpMeta.parameter_sha256 === record.controller_parameters_before?.parameter_sha256 && record.update?.likelihood_alignment?.passed === true,
        samples: sampleCount,
        active_row_count: rowsActive.length,
        full_mask_row_count: rowsFull.length,
        no_candidate_rows: rowsAll.filter(x => x.class === "no_candidate").length,
        prefix_task_reward: prefixTask,
        prefix_cost_bytes: prefixCost,
        closure_error_max_abs: Math.max(...episodeMaths.map(x => x.closure)),
        full_batch: {
          reward_task_component: metricSummary(rowsAll, "reward_component"),
          current_value_component: metricSummary(rowsAll, "current_value_component"),
          next_value_bootstrap_component: metricSummary(rowsAll, "next_value_bootstrap_component"),
          A_task: metricSummary(rowsAll, "A_task"),
          A_cost: metricSummary(rowsAll, "A_cost"),
          lambda_times_A_task: metricSummary(rowsAll, "lambda_times_A_task"),
          A_actor: metricSummary(rowsAll, "A_actor"),
        },
        active_rows: {
          reward_task_component: metricSummary(rowsActive, "reward_component"),
          current_value_component: metricSummary(rowsActive, "current_value_component"),
          next_value_bootstrap_component: metricSummary(rowsActive, "next_value_bootstrap_component"),
          A_task: metricSummary(rowsActive, "A_task"),
          A_cost: metricSummary(rowsActive, "A_cost"),
          lambda_times_A_task: metricSummary(rowsActive, "lambda_times_A_task"),
          A_actor: metricSummary(rowsActive, "A_actor"),
          lambda_task_minus_cost_relation: {
            abs_mean_lambda_A_task: metricSummary(rowsActive, "lambda_times_A_task").abs_mean,
            abs_mean_A_cost: metricSummary(rowsActive, "A_cost").abs_mean,
            abs_mean_cost_over_task: metricSummary(rowsActive, "lambda_times_A_task").abs_mean === 0 ? null : metricSummary(rowsActive, "A_cost").abs_mean / metricSummary(rowsActive, "lambda_times_A_task").abs_mean,
            signs: {lambda_A_task: signCounts(rowsActive.map(x => x.lambda_times_A_task)), minus_A_cost: signCounts(rowsActive.map(x => -x.A_cost)), actor_combined: signCounts(rowsActive.map(x => x.A_actor))},
          },
        },
        full_mask_row_A_task: metricSummary(rowsFull, "A_task"),
        stored_mechanism_comparison: {},
      };
      if (storedPoint?.advantages) {
        component.stored_mechanism_comparison = {
          A_task: compareStoredSummary(component.full_batch.A_task, storedPoint.advantages.A_task, "A_task all rows"),
          A_cost: compareStoredSummary(component.full_batch.A_cost, storedPoint.advantages.A_cost, "A_cost all rows"),
          A_combined: compareStoredSummary(component.full_batch.A_actor, storedPoint.advantages.A_combined, "A_actor all rows"),
          stored_point_closure_error: storedPoint.advantage_combination_max_abs_error,
          this_round_recompute_closure_error: component.closure_error_max_abs,
        };
      } else component.stored_mechanism_comparison = {available: false};
      componentPoints.push(component);
      pointReports.push(component);

      const nearestDistances = [];
      const allPairs = [];
      for (let e = 0; e < episodeMaths.length; e++) {
        const m = episodeMaths[e];
        const row = episodeRows[e];
        for (const reward of m.taskEvents) {
          if (reward.ownerIndex < 0) continue;
          const prior = m.active.filter(d => d.tape_row_1based - 1 <= reward.ownerIndex && d.environment_step <= reward.tick);
          if (!prior.length) continue;
          const nearest = prior[prior.length - 1];
          const nearestDistance = reward.ownerIndex - (nearest.tape_row_1based - 1);
          nearestDistances.push({row_distance: nearestDistance, env_step_distance: reward.tick - nearest.environment_step, coefficient: (GAMMA * LAMBDA_GAE) ** nearestDistance});
          for (const active of prior) {
            const distance = reward.ownerIndex - (active.tape_row_1based - 1);
            allPairs.push({row_distance: distance, env_step_distance: reward.tick - active.environment_step, coefficient: (GAMMA * LAMBDA_GAE) ** distance});
          }
        }
      }
      allSelectedTransfer.push(...allPairs);
      transferBatchResults.push({point, reward_to_nearest_active_pairs: nearestDistances.length, nearest_distance_rows: stats(nearestDistances.map(x => x.row_distance)), nearest_distance_env_steps: stats(nearestDistances.map(x => x.env_step_distance)), nearest_direct_reward_coefficient: stats(nearestDistances.map(x => x.coefficient)), all_prior_active_pairs: allPairs.length, all_prior_active_distance_rows: stats(allPairs.map(x => x.row_distance)), all_prior_active_env_steps: stats(allPairs.map(x => x.env_step_distance)), all_prior_active_direct_reward_coefficient: stats(allPairs.map(x => x.coefficient)), row_distance_histogram: allPairs.reduce((o, x) => (o[x.row_distance] = (o[x.row_distance] ?? 0) + 1, o), {})});
      const criticPoint = {
        point,
        checkpoint_file_sha_verified: identityChecks.checkpoint_file_sha_matches_index,
        checkpoint_parameter_and_lambda_match_current_record: identityChecks.parameter_sha_matches_batch_record && identityChecks.lambda_matches,
        likelihood_identity_verified_from_stored_gate: likelihood.stored_passed === true && likelihood.sample_count_matches,
        active_rows: criticMetrics(rowsActive),
        full_mask_rows: criticMetrics(rowsFull),
        first_report_free_decision_rows: criticMetrics(rowsFirst),
        unavailable: rowsAll.some(x => !Number.isFinite(x.return_task) || !Number.isFinite(x.value_task)),
      };
      criticReport.sample_diagnostics.push(criticPoint);
      console.log(`ROUND33 fixed batch ${point}: rows=${sampleCount}, active=${rowsActive.length}, cp_sha=${identityChecks.checkpoint_file_sha_matches_index}`);
    }
  }

  // Task A: pair each one of the 16 already-used dev scenes to every learned final evaluation.
  const devRange = r => Number(r.scenario_seed) >= 1301000 && Number(r.scenario_seed) <= 1301015;
  const immediate = rows.filter(r => r.kind === "reference" && r.policy === "immediate" && devRange(r));
  const learned = rows.filter(r => r.policy === "learned" && devRange(r) && ARMS.includes(r.reduction));
  const devRows = [...immediate, ...learned];
  const devLife = new Map();
  for (const row of devRows) {
    const file = row.artefacts?.lifecycle;
    if (!devLife.has(file)) devLife.set(file, await lifecycleSummary(file));
  }
  const immediateByScene = new Map(immediate.map(r => [Number(r.scenario_seed), r]));
  const devChecks = [];
  for (let scene = 1301000; scene <= 1301015; scene++) {
    const ref = immediateByScene.get(scene);
    const refLife = ref ? devLife.get(ref.artefacts?.lifecycle) : null;
    const matched = learned.filter(r => Number(r.scenario_seed) === scene);
    const comparisons = matched.map(r => {
      const life = devLife.get(r.artefacts?.lifecycle);
      const stableConfig = cfg => {
        if (!cfg) return null;
        const c = {...cfg};
        delete c.comm_policy;
        delete c.output;
        delete c.comm_transcript;
        return c;
      };
      return {
        label: r.label,
        training_seed: r.training_seed,
        reduction: r.reduction,
        mode: r.label?.match(/(greedy|sampled)/i)?.[1]?.toLowerCase() ?? null,
        on_time: r.on_time,
        spent_bytes: r.spent_bytes,
        budget_bytes: r.budget_bytes,
        same_budget_as_immediate: !!ref && Number(r.budget_bytes) === Number(ref.budget_bytes),
        config_matches_except_policy_and_output_paths: !!refLife?.runConfig && !!life?.runConfig && stable(stableConfig(refLife.runConfig.configuration)) === stable(stableConfig(life.runConfig.configuration)),
        scenario_detail_matches: !!refLife?.runConfig && !!life?.runConfig && stable(refLife.runConfig.scenario) === stable(life.runConfig.scenario),
        frozen_checkpoints_match: !!refLife?.runConfig && !!life?.runConfig && stable(refLife.runConfig.run_signature?.checkpoints) === stable(life.runConfig.run_signature?.checkpoints),
        discovery_stream_matches: !!refLife && !!life && refLife.discovery_signature === life.discovery_signature,
        reference_discovery_count: refLife?.discoveries.length ?? null,
        learned_discovery_count: life?.discoveries.length ?? null,
        learned_target_ids: life?.discoveries.map(x => x.target_id) ?? null,
        row_config_hash: r.config_hash,
        reference_config_hash: ref?.config_hash ?? null,
      };
    });
    devChecks.push({
      scenario_seed: scene,
      immediate_present: !!ref,
      immediate_on_time: ref?.on_time ?? null,
      immediate_completed: ref?.completed ?? null,
      immediate_discovered: ref?.discovered ?? null,
      immediate_spent_bytes: ref?.spent_bytes ?? null,
      immediate_budget_bytes: ref?.budget_bytes ?? null,
      immediate_all_six_on_time: !!ref && Number(ref.on_time) === 6,
      immediate_discovery_count: refLife?.discoveries.length ?? null,
      learned_comparisons: comparisons,
    });
  }
  const dev = {
    question: "Do the 16 previously used development scenes have an already-observed policy at the 6/6 on-time ceiling under the same hard budget, with aligned scenario/search inputs?",
    scenes: devChecks,
    counts: {
      immediate_rows: immediate.length,
      immediate_scenes_present: immediateByScene.size,
      immediate_scenes_all_six_on_time: devChecks.filter(x => x.immediate_all_six_on_time).length,
      immediate_same_hard_budget_scenes: devChecks.filter(x => Number.isFinite(x.immediate_budget_bytes) && x.immediate_budget_bytes === 29300).length,
      learned_final_rows: learned.length,
      learned_rows_with_equal_budget: devChecks.flatMap(x => x.learned_comparisons).filter(x => x.same_budget_as_immediate).length,
      learned_rows_with_config_scenario_checkpoints_and_discovery_match: devChecks.flatMap(x => x.learned_comparisons).filter(x => x.config_matches_except_policy_and_output_paths && x.scenario_detail_matches && x.frozen_checkpoints_match && x.discovery_stream_matches).length,
      learned_rows_expected: 16 * 3 * 2 * 2,
    },
    conclusion_if_all_checks_pass: "These previously used development scenes already have an observed legal policy at the six-task ceiling. This does not establish a lower-cost zero-loss policy or that the model can learn one.",
  };

  const sourceFiles = [
    "runs/comm_cost_v2_rerun_20260928/v2_common.py",
    "runs/comm_cost_v2_rerun_20260928/v2_trainer.py",
    "runs/comm_cost_v2_rerun_20260928/v2_run.py",
    "runs/comm_cost_controller_stage_a_20260927/cost_math.py",
    "integrated_sar/joint_driver.py",
    "integrated_sar/orchestrator.py",
    "integrated_sar/adapters/task03.py",
    "runs/comm_actor_reduction_ablation_20261001/r32_common.py",
    "runs/comm_actor_reduction_ablation_20261001/r32_run.py",
    "runs/comm_actor_reduction_ablation_20261001/r32_trainer.py",
  ].map(rel => ({path: rel, sha256: fs.existsSync(path.join(WORKSPACE, rel)) ? fileSha(path.join(WORKSPACE, rel)) : null}));
  const costMathText = fs.readFileSync(path.join(STAGE_A, "cost_math.py"), "utf8");
  const v2RunText = fs.readFileSync(path.join(V2, "v2_run.py"), "utf8");
  const task03Text = fs.readFileSync(path.join(WORKSPACE, "integrated_sar", "adapters", "task03.py"), "utf8");
  const orchestratorText = fs.readFileSync(path.join(WORKSPACE, "integrated_sar", "orchestrator.py"), "utf8");
  const tapEnvStepIndex = v2RunText.indexOf("result = original_step(*args, **kwargs)");
  const tapRegistryScanIndex = v2RunText.indexOf("for snapshot in run.registry.snapshots()", tapEnvStepIndex);
  const taskRelayTickIndex = task03Text.indexOf("self.comm_relay.tick()", task03Text.indexOf("def step(self, current_time"));
  const taskEnvStepIndex = task03Text.indexOf("self.env.step(dt)", taskRelayTickIndex);
  const executorCallIndex = orchestratorText.indexOf("execution_result = self.executor.step(now, self.dt)");
  const progressPublishIndex = orchestratorText.indexOf("for event in execution_result.progress_events", executorCallIndex);
  const sourceParity = {
    frozen_cost_math_path: "runs/comm_cost_controller_stage_a_20260927/cost_math.py:gae",
    delta_formula_present: costMathText.includes("delta = rewards[step] + gamma * bootstrap[step].to(rewards.dtype) * done - values[step]"),
    recursive_formula_present: costMathText.includes("advantages[step] = delta + gamma * gae_lambda * done * next_advantage"),
    caller_sets_terminal_last_true: fs.readFileSync(path.join(V2, "v2_trainer.py"), "utf8").includes("terminal[-1] = True"),
    caller_sets_per_decision_next_value: fs.readFileSync(path.join(V2, "v2_trainer.py"), "utf8").includes("bootstrap_task[:-1] = values_task[1:]"),
    caller_assigns_post_action_reward_to_latest_decision: fs.readFileSync(path.join(V2, "v2_trainer.py"), "utf8").includes("previous = bisect_right(decision_steps, step) - 1"),
    caller_preserves_masked_rows_for_gae: fs.readFileSync(path.join(V2, "v2_trainer.py"), "utf8").includes("for tape in tapes") && fs.readFileSync(path.join(V2, "v2_trainer.py"), "utf8").includes("tapes = sorted(episode[\"collector\"].records"),
    tap_scans_registry_after_wrapped_env_step_before_adapter_returns_events: tapEnvStepIndex >= 0 && tapRegistryScanIndex > tapEnvStepIndex,
    task03_calls_relay_tick_before_env_step: taskRelayTickIndex >= 0 && taskEnvStepIndex > taskRelayTickIndex,
    orchestrator_publishes_progress_only_after_executor_returns: executorCallIndex >= 0 && progressPublishIndex > executorCallIndex,
  };
  sourceParity.all_formula_assertions = sourceParity.delta_formula_present && sourceParity.recursive_formula_present && sourceParity.caller_sets_terminal_last_true && sourceParity.caller_sets_per_decision_next_value && sourceParity.caller_assigns_post_action_reward_to_latest_decision && sourceParity.caller_preserves_masked_rows_for_gae;
  sourceParity.task_reward_capture_order_establishes_registry_lag = sourceParity.tap_scans_registry_after_wrapped_env_step_before_adapter_returns_events && sourceParity.task03_calls_relay_tick_before_env_step && sourceParity.orchestrator_publishes_progress_only_after_executor_returns;

  const allCheckpointFileHashesPass = criticReport.checkpoint_validation.length === 18 && criticReport.checkpoint_validation.every(x => x.checkpoint_exists && x.checkpoint_file_sha_matches_index);
  const allCheckpointBatchMetadataPass = criticReport.checkpoint_validation.length === 18 && criticReport.checkpoint_validation.every(x => x.parameter_sha_matches_batch_record && x.batch_has_checkpointed_batch_index && x.lambda_matches);
  const selectedLikelihoodPass = criticReport.source_batch_likelihood_identity.length === 18 && criticReport.source_batch_likelihood_identity.every(x => x.stored_passed === true && x.sample_count_matches);
  const componentStoredCompare = componentPoints.map(x => ({point: x.point, comparison: x.stored_mechanism_comparison}));
  const maxStoredSummaryGap = Math.max(...componentPoints.flatMap(x => [x.stored_mechanism_comparison?.A_task?.max_abs_gap, x.stored_mechanism_comparison?.A_cost?.max_abs_gap, x.stored_mechanism_comparison?.A_combined?.max_abs_gap].filter(Number.isFinite)));
  const temporal = {
    source_lineage: {
      round32_head: "01e5a677cd18a7abfdb97585315ba05251d4e0de",
      branch_base: "01e5a677cd18a7abfdb97585315ba05251d4e0de",
      round32_configuration_hash: config.configuration_hash,
      round32_head_at_freeze: config.head_at_freeze,
      round32_freeze_sha256: config.code_identity?.freeze_sha256 ?? null,
      round32_model_checkpoints: config.code_identity?.checkpoints ?? null,
      round32_checkpoint_index: "runs/comm_actor_reduction_ablation_20261001/CHECKPOINT_INDEX.json",
      training_row_directory: "runs/comm_actor_reduction_ablation_20261001/rows",
      training_tape_directory: "runs/comm_actor_reduction_ablation_20261001/logs/*_tape.jsonl",
      prompt_basis: "runs/comm_temporal_credit_audit_20260930/EXECUTION_BASIS.md",
      prompt_basis_sha256: fileSha(path.join(OUT, "EXECUTION_BASIS.md")),
      input_source_files: sourceFiles,
      row_json_inventory_sha256: rowInventorySha256,
      training_tape_inventory_sha256: training.tapeInventorySha256,
    },
    data_coverage: {
      total_round32_row_files: rowFileCount,
      training_rows_expected: training.expectedEpisodes,
      training_rows_observed: training.observedRows,
      training_tapes_readable: training.tapeFiles,
      missing_materials: training.missing,
      parse_errors: training.parseErrors,
      source_row_fields_reconciled: training.rewardStreamReconciliationFailures.length === 0,
      excluded_by_success_or_failure: false,
      sealed_blocks_touched: 0,
    },
    frozen_parameters: {
      B0_bytes: B0,
      gamma: GAMMA,
      lambda_GAE: LAMBDA_GAE,
      lambda_dual: {initial: 1.0, eta: 0.05, per_batch_values: criticReport.checkpoint_validation.map(x => ({point: x.point, value: x.lambda_dual}))},
      delta: 0,
      reward_timing_schema: "post-action-most-recent-decision/1",
      gae_schema: "per-decision-next-value/1",
      lambda_GAE_is_not_lambda_dual: true,
    },
    axis_contract: {
      environment_time: "relay/task environment tick; run summaries are configured for 1000 ticks; sparse task events and dense airtime accounting use integer ticks 1..1000",
      relay_tape_time: "each tape step equals the relay tick sampled by Stage C tap; source loop calls one relay.tick per task-environment step",
      training_row_index: "0-based per-episode index after sorting tape records by integer step; batch_samples concatenates all records including fully masked rows, episode by episode",
      observed_rows: training.totalTapeRows,
      total_active_rows: training.totalActiveRows,
      total_full_mask_rows: training.totalFullMaskRows,
      total_no_candidate_rows: training.totalNoCandidateRows,
      row_class_shares: training.rowClassShares,
      no_candidate_semantics: "a tick with no candidate is not emitted as a tape row; any tape row with candidates and actionable=[] is a full-mask row and remains in GAE",
      recursive_time_unit: "recorded per-decision/tape row, not environment tick; gamma=1, so no per-tick gamma decay; lambda_GAE=0.95 is multiplied once per subsequent tape row, including all-mask rows",
    },
    execution_order: {
      action_and_cost: "Task03Adapter.step calls CommRelay.tick before env.step; the learned policy action and send attempts happen inside that relay tick; the tap records cost after relay.tick returns",
      task_reward: "The v2_run wrapper calls the original env.step, then scans TaskRegistry snapshots. Task03Adapter only returns ProgressEvents after env.step returns, and JointSAROrchestrator publishes those events after Executor.step returns. Thus the wrapper sees the prior registry state during a new completion tick; archived lifecycle events at tick t match task_stream rewards at t+1.",
      bundle_rule: "v2_trainer._bundle credits each tick to the latest tape step <= tick; same tick belongs to that decision; between decisions belongs to earlier decision; before first tape row is a separately accumulated prefix; after last row is attached to last row",
      round23_repair: "The _bundle implementation uses latest prior-or-same decision, but the task tap stamps physical completion reward one relay tick late. If a tape decision exists at t+1, the delayed event is assigned to that post-completion decision, contrary to the intended event-time ownership.",
      task_reward_timing_evidence: {
        source_order_assertions: {
          task_tap_reads_registry_before_new_progress_events_are_published: sourceParity.task_reward_capture_order_establishes_registry_lag,
          relay_tick_precedes_task_environment_step: sourceParity.task03_calls_relay_tick_before_env_step,
          orchestrator_publishes_events_after_executor_returns: sourceParity.orchestrator_publishes_progress_only_after_executor_returns,
        },
        training_completion_events: training.targetIdentity.on_time_completion_events,
        physical_completion_has_exact_step_tape_row: training.targetIdentity.completion_event_has_exact_step_tape_row,
        same_lifecycle_timestamp_reward_matches: training.targetIdentity.completion_same_timestamp_reward_matches,
        lifecycle_timestamp_plus_one_reward_matches: training.targetIdentity.completion_one_tick_later_reward_matches,
        lifecycle_timestamp_plus_one_reward_match_fraction: training.targetIdentity.on_time_completion_events === 0 ? null : training.targetIdentity.completion_one_tick_later_reward_matches / training.targetIdentity.on_time_completion_events,
        logged_reward_owner_is_a_decision_after_completion: training.targetIdentity.reward_owner_after_event_decision_row,
        logged_reward_owner_remains_event_decision_row: training.targetIdentity.reward_owner_same_as_event_decision_row,
        delayed_reward_owner_class: training.targetIdentity.completion_reward_owner_class,
        physical_event_owner_class: training.targetIdentity.completion_event_owner_class,
        affected_reward_recursion: "When a tape decision row exists at t+1, the reward shifts from event row t to the later row t+1; with fixed critic this changes the direct coefficient to row t from 1 to gamma*lambda_GAE=0.95. If no t+1 tape row exists, _bundle still assigns to row t, though the recorded environment tick remains late.",
      },
      source_parity: sourceParity,
    },
    training_tape_statistics: training,
    development_ceiling_audit: dev,
    timing_contract: {
      configured_environment_steps_per_episode: 1000,
      observed_training_rows_all_steps_1000: trainRows.every(r => Number(r.steps) === 1000),
      simulator_driver: "JointSARDriver.run calls orchestrator.run(total); orchestrator.run executes exactly total iterations with no early-stop condition",
      physical_completion_vs_recorded_task_reward: "Across all 768 training episodes, each of 4,363 on-time ProgressEvent completion records is timestamped at lifecycle tick t and the same number of task reward units appear at relay task_stream tick t+1. The one-tick shift follows source call order: the tap scans registry snapshots before the newly returned progress events are published.",
      task_reward_credit_impact: "2,821/4,363 completion reward units have a tape row at t+1 and are credited to that later post-completion row; 1,542/4,363 remain owned by the t row because there is no later decision row. Of the shifted later rows, 424 are active and 2,397 are full-mask. Physical event rows classify 652 active and 3,711 full-mask; recorded reward-owner rows classify 599 active and 3,764 full-mask.",
      task_reward_contract_status: "The latest-prior-row bundler matches its own source schema, but the upstream task reward tap violates event-time post-action ownership. Overall task reward timing does not match the registered ROUND23 contract.",
      tape_has_terminal_or_truncated_flag: false,
      batch_samples_semantics: "unconditionally marks the last recorded tape row terminal=True; uses next row values at earlier rows and zero bootstrap at the final tape row",
      true_terminal_vs_time_limit: "The saved runner executes a fixed 1000-tick horizon and does not store a distinct terminated/truncated bit in tape or row. The trainer's last-tape-row terminal convention is asserted in code comments, but this audit cannot independently prove from archived rows whether each final decision is absorbing or a horizon cut with a value-bearing continuation. This ambiguity is registered as a boundary, not classified as an implementation defect without a contrary frozen contract.",
    },
    fixed_18_batch_validation: {
      expected_points: 18,
      analyzable_points: componentPoints.length,
      stopped_points: criticReport.stopped_points.length,
      all_18_checkpoint_files_exist_and_sha_match_index: allCheckpointFileHashesPass,
      all_18_checkpoint_parameter_lambda_metadata_match_current_batch_records: allCheckpointBatchMetadataPass,
      checkpoint_parameter_lambda_identity_matches_current_batch_record_count: criticReport.checkpoint_validation.filter(x => x.parameter_sha_matches_batch_record && x.lambda_matches).length,
      checkpoint_metadata_conflict_points: criticReport.checkpoint_validation.filter(x => !x.parameter_sha_matches_batch_record || !x.lambda_matches).map(x => x.point),
      stored_likelihood_alignment_gates_pass_and_counts_match_tapes: selectedLikelihoodPass,
      recorded_batch_sample_counts_match_tapes: criticReport.source_batch_likelihood_identity.every(x => x.sample_count_matches),
      saved_update_mechanism_counts_agree_with_batch_record_and_current_tapes: criticReport.checkpoint_validation.filter(x => x.current_batch_tape_and_saved_mechanism_counts_agree).length,
      saved_update_mechanism_summary_matches_on_provenance_consistent_points: componentPoints.length === 6 && Number.isFinite(maxStoredSummaryGap) && maxStoredSummaryGap <= 1e-6,
      maximum_summary_field_gap_vs_UPDATE_MECHANISM: Number.isFinite(maxStoredSummaryGap) ? maxStoredSummaryGap : null,
      mechanism_comparisons: componentStoredCompare,
      provenance_conflicts: criticReport.provenance_conflicts,
      checkpoint_weights_loaded_or_re_forwarded: false,
      checkpoint_file_integrity_checked_without_deserialization: true,
    },
    nearest_prior_active_credit: {
      definition: "distance from each task reward tick to the most recent active tape row at or before the tick; a descriptive temporal proximity, not action causality",
      rows: training.nearestActiveRowDistance,
    },
    reward_component_interpretation: "A_task = explicit task-reward GAE + current-value term + next-value/bootstrap term. The three are computed linearly under the actual gamma/lambda/terminal vectors, without separate normalization.",
    missing_or_unverified: [
      "No Python/PyTorch executable in the execution shell; all 18 .pt file SHA256 values match the checkpoint index, but current batch parameter/lambda metadata matches only 6 of 18 points; checkpoints were not deserialized and forward-replayed.",
      "Native Python cost_math.gae was source-inspected and the same loop was translated to JavaScript for pure calculations and synthetic finite differences; this round could not import/call the Python function directly.",
      "Termination versus time-limit identity is not serialized per tape/episode; last tape row is unconditionally marked terminal in batch_samples.",
      "Use of task IDs establishes matching first-free candidate and completion identities where available, but that temporal linkage does not prove the decision caused completion.",
      "The observed task reward tap is one relay tick later than every recorded on-time completion; the six computable batch decompositions use the stored, delayed reward stream and are not a corrected event-time counterfactual.",
    ],
  };

  const components = {
    source: {
      rows_directory: "runs/comm_actor_reduction_ablation_20261001/rows",
      tapes_directory: "runs/comm_actor_reduction_ablation_20261001/logs",
      checkpoint_index: "runs/comm_actor_reduction_ablation_20261001/CHECKPOINT_INDEX.json",
      update_mechanism: "runs/comm_actor_reduction_ablation_20261001/UPDATE_MECHANISM.json",
    },
    definition: {
      gamma: GAMMA,
      lambda_GAE: LAMBDA_GAE,
      terminal: "from frozen batch_samples: final recorded decision row true, all earlier rows false",
      bootstrap: "recorded task/cost value at next tape row for every earlier row; final row zero",
      no_component_normalization: true,
      active_row: "tape.actionable has at least one target",
      full_mask_row: "tape.targets nonempty and tape.actionable empty",
      update_source_identity: "fixed-point eligibility requires checkpoint file SHA256, checkpoint parameter/lambda identity vs current batch record, saved likelihood count vs tape, and UPDATE_MECHANISM sample counts to agree. The six b0 points agree; b15/b31 points fail identity/count gates and are stopped.",
    },
    synthetic_gae_validation: syntheticTestResults,
    points: pointReports,
  };

  const critic = criticReport;

  const transfer = {
    fixed_points_expected: 18,
    fixed_points_observed: transferBatchResults.length,
    analyzable_points: componentPoints.length,
    stopped_points: criticReport.stopped_points,
    formula: "For a task reward in tape row j, its direct contribution to A_task at earlier row i with no terminal boundary between is (gamma*lambda_GAE)^(j-i); full-mask rows remain in j-i.",
    interpretation: "Coefficient measures only the explicit reward path with critic predictions fixed; a small value does not mean the training signal has no long-term information because critic value/bootstrap terms may compensate.",
    per_batch: transferBatchResults,
    all_prior_active_reward_row_pairs: {
      count: allSelectedTransfer.length,
      training_row_distance: stats(allSelectedTransfer.map(x => x.row_distance)),
      environment_step_distance: stats(allSelectedTransfer.map(x => x.env_step_distance)),
      direct_reward_coefficient: stats(allSelectedTransfer.map(x => x.coefficient)),
      distance_histogram: allSelectedTransfer.reduce((o, x) => (o[x.row_distance] = (o[x.row_distance] ?? 0) + 1, o), {}),
    },
  };
  components.direct_reward_transfer = transfer;
  temporal.fixed_18_batch_validation.direct_reward_transfer = transfer;

  const sourceHashes = Object.fromEntries(sourceFiles.map(x => [x.path, x.sha256]));
  const verdict = {
    round: 33,
    local_branch: "analysis/comm-temporal-credit-audit",
    branch_base: "01e5a677cd18a7abfdb97585315ba05251d4e0de",
    audit_type: "read-only archival temporal-credit audit",
    verified: {
      latest_prior_row_bundler_matches_registered_rule: sourceParity.caller_assigns_post_action_reward_to_latest_decision,
      task_reward_capture_order_matches_round23_event_time_contract: false,
      source_gae_formula_matches_source_bound_translation: sourceParity.delta_formula_present && sourceParity.recursive_formula_present,
      all_readable_training_tapes_covered: training.tapeFiles === training.expectedEpisodes && training.missing.length === 0 && training.parseErrors.length === 0,
      all_18_pre_update_checkpoint_files_exist_and_match_index_sha256: allCheckpointFileHashesPass,
      all_18_pre_update_checkpoint_parameter_lambda_metadata_match_current_records: allCheckpointBatchMetadataPass,
      all_18_saved_likelihood_gate_counts_match_tapes: selectedLikelihoodPass,
      recomputed_A_task_summary_matches_saved_mechanism_on_provenance_consistent_points: componentPoints.length === 6 && Number.isFinite(maxStoredSummaryGap) && maxStoredSummaryGap <= 1e-6,
      full_18_batch_identity_resolved: criticReport.stopped_points.length === 0 && componentPoints.length === 18,
      sixteen_scene_immediate_task_ceiling_established: dev.counts.immediate_scenes_all_six_on_time === 16,
      dev_inputs_match_across_reference_and_learned: dev.counts.learned_rows_with_config_scenario_checkpoints_and_discovery_match === dev.counts.learned_rows_expected,
    },
    conclusions: [
      "The task reward bundler chooses the latest tape row at or before its logged tick, but the producer logs each physical task-completion reward one tick late. In 2,821 of 4,363 completion cases, that tick has a new post-completion decision row and the reward is credited to it; the registered event-time reward contract is violated. Prefix task reward/cost behavior remains separately identified.",
      "GAE recurses over tape rows, including fully masked rows, rather than environment ticks. Gamma is 1 and lambda_GAE is 0.95; lambda_dual is a separate constraint multiplier.",
      "The 16 already-used development scenes have immediate completing all six on-time tasks at the same 29,300-byte hard budget, conditional on the row/config/discovery cross-checks recorded in this audit. This establishes an observed task ceiling for those scenes only.",
      "The ROUND32 ALL_ROWS versus ACTIVE_ROWS negative result only weakens the explanation that merely amplifying the actor surrogate is sufficient. Large advantage magnitude is not proof of correct long-term task credit, and equal post-clip norm is not equal Adam parameter-step length.",
      "Direct fixed-critic reward transfer is quantified separately from task value and next-value/bootstrap terms. It cannot alone establish absence of long-term information.",
    ],
    contract_disposition: "Confirmed task-reward timing contract violation: v2_run scans the registry before Task03's new ProgressEvents are published, and all 4,363 archived completion timestamps align to task_stream one tick later. The delayed stream assigns 2,821 rewards to a later tape row after completion. Stop parameter diagnosis; do not alter GAE/lambda or train on this basis. Independently, C/D/E real-batch conclusions stop at six batch-0 points because twelve b15/b31 checkpoint parameter/lambda and UPDATE_MECHANISM sample identities conflict with current batch/tape artifacts. The GAE formula and latest-prior-row bundler implementation match inspected source; final-row terminal vs time-limit identity remains unresolved.",
    not_established: [
      "Whether critic values/bootstrap compensate for decayed explicit reward-path coefficients as an expectation or on independent episodes.",
      "A causal claim that the nearest active row caused the later task completion.",
      "A lower-cost policy that preserves six tasks; the immediate baseline is already at the task ceiling but is not evidence of the lowest possible cost.",
      "Greedy and sampled behavior are not interchangeable; no mixed-mode mean is used.",
      "Deployment-level zero-regression guarantee from delta=0.",
      "Independent generalization from repeatedly used training/development blocks.",
      "Checkpoint re-forward under PyTorch in this execution environment.",
      "A complete fixed-18 real-batch decomposition/critic diagnosis until the 12 checkpoint/batch/mechanism provenance conflicts are reconciled.",
      "A parameter-level interpretation of credit limitation while task reward capture remains one tick late relative to physical completion.",
    ],
    follow_up: {
      status: "proposed_not_executed",
      action: "Correct the task reward tap to capture newly completed ProgressEvents in the same relay tick after the action/env step, then add a pure source-order/replay regression for event tick t -> reward tick t before proposing any parameter experiment. Do not execute this correction in ROUND33.",
      hypothesis: "The registry scan in v2_run observes the previous registry snapshot because Task03 returns completion events after env.step and the orchestrator publishes those events only after executor.step returns; the resulting one-tick lag moves 2,821 rewards to a post-completion tape row.",
      expected_resources: "For the proposed source correction/replay check: 0 environment starts, 0 input streams, 0 optimizer steps, 0 checkpoint writes. After that is independently verified, reconcile the 12 fixed-batch provenance conflicts before any parameter experiment.",
      rationale_limit: "Do not select gamma, lambda_GAE, lambda_dual, reward shaping, or training volume while the reward timeline violates contract and only six of 18 batches pass identity.",
    },
    execution_counters: {
      new_environment_launches: 0,
      new_input_streams: 0,
      real_training_updates: 0,
      synthetic_optimizer_updates: 0,
      synthetic_gae_tests: syntheticTestResults.tests_run,
      analysis_type: "Node.js parsing, SHA256, exact-formula translation, GAE arithmetic and finite-difference tests only",
    },
    remote_publication: "not_pushed; no network attempt; new branch has no configured upstream",
    source_sha256: sourceHashes,
    elapsed_seconds: (Date.now() - started) / 1000,
  };

  const targetLinks = training.targetIdentity.links;
  const targetIdentitySummary = {...training.targetIdentity,
    distance_from_first_free_decision_to_physical_completion: {
      training_rows: stats(targetLinks.map(x => x.training_row_distance_to_physical_completion_owner)),
      environment_steps: stats(targetLinks.map(x => x.environment_step_distance_to_physical_completion)),
    },
    distance_from_first_free_decision_to_recorded_reward_owner: {
      training_rows: stats(targetLinks.map(x => x.training_row_distance_to_recorded_reward_owner)),
      environment_steps: stats(targetLinks.map(x => x.environment_step_distance_to_recorded_reward_tick)),
    },
    example_links: targetLinks.slice(0, 8),
  };
  delete targetIdentitySummary.links;
  const trainingOutput = {...training, targetIdentity: targetIdentitySummary, episodeSummaries: undefined};
  const temporalOut = {
    ...temporal,
    training_tape_statistics: trainingOutput,
    conclusions: {
      full_mask_rows_in_recursion: training.totalFullMaskRows,
      no_candidate_rows_absent_from_tape: training.totalNoCandidateRows,
      reward_ownership: training.taskRewardOwner,
      cost_ownership: training.costOwner,
      nearest_active_row_distances: training.nearestActiveRowDistance,
      target_identity_summary: targetIdentitySummary,
      complete_audit_machine_readable: true,
    },
  };
  const componentOut = {...components, recorded_saved_summary_checks: componentStoredCompare};
  const criticOut = {...critic, point_level: criticReport.sample_diagnostics};
  for (const [name, data] of [
    ["TEMPORAL_CREDIT_AUDIT.json", temporalOut],
    ["GAE_COMPONENTS.json", componentOut],
    ["CRITIC_DIAGNOSTIC.json", criticOut],
    ["ROUND33_VERDICT.json", verdict],
    ["SYNTHETIC_GAE_TESTS.json", syntheticTestResults],
  ]) {
    fs.writeFileSync(path.join(OUT, name), JSON.stringify(data, null, 2) + "\n", "utf8");
  }
  console.log(JSON.stringify({
    training_rows: training.observedRows,
    tapes: training.tapeFiles,
    tape_rows: training.totalTapeRows,
    active_rows: training.totalActiveRows,
    full_mask_rows: training.totalFullMaskRows,
    task_reward_units: training.totalTaskRewardUnits,
    cost_bytes: training.totalCostBytes,
    checkpoint_points: criticReport.checkpoint_validation.length,
    component_points_analyzable: componentPoints.length,
    component_points_stopped: criticReport.stopped_points.length,
    max_stored_summary_gap: maxStoredSummaryGap,
    dev_immediate_six_of_six: dev.counts.immediate_scenes_all_six_on_time,
    dev_learned_input_matches: dev.counts.learned_rows_with_config_scenario_checkpoints_and_discovery_match,
    elapsed_seconds: verdict.elapsed_seconds,
  }, null, 2));
}

if (require.main === module) {
  run().catch(error => {
    console.error(error && error.stack ? error.stack : error);
    process.exitCode = 1;
  });
}

module.exports = {sourceBoundGae, stats, ownerIndex, lifecycleSummary, runSyntheticGaeTests};
