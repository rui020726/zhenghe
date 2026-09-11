"""Generate the non-data-dependent paper sections as a Word document."""

from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
import datetime

doc = Document()

# ── Page setup ──
for section in doc.sections:
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

# ── Style helpers ──
style = doc.styles["Normal"]
font = style.font
font.name = "Times New Roman"
font.size = Pt(11)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.15

for level in range(1, 4):
    h = doc.styles[f"Heading {level}"]
    h.font.name = "Times New Roman"
    h.font.color.rgb = RGBColor(0, 0, 0)
    if level == 1:
        h.font.size = Pt(14)
        h.font.bold = True
    elif level == 2:
        h.font.size = Pt(12)
        h.font.bold = True
    else:
        h.font.size = Pt(11)
        h.font.bold = True
        h.font.italic = True


def add_para(text, bold=False, italic=False, size=None, alignment=None):
    """Add a paragraph with optional formatting."""
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    if alignment is not None:
        p.alignment = alignment
    return p


def add_placeholder(text):
    """Add a placeholder paragraph in gray."""
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.color.rgb = RGBColor(150, 150, 150)
    run.italic = True
    return p


# ══════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════
add_para(
    "Deadline-Aware Heterogeneous Multi-Robot Task Allocation "
    "via Centralized Training with Decentralized Execution",
    bold=True, size=16, alignment=WD_ALIGN_PARAGRAPH.CENTER
)

add_para("", size=6)

add_para(
    "[Author names], [Affiliations], [Corresponding author email]",
    italic=True, size=10, alignment=WD_ALIGN_PARAGRAPH.CENTER
)

add_para(
    f"Draft — {datetime.date.today().strftime('%d %B %Y')}",
    italic=True, size=9, alignment=WD_ALIGN_PARAGRAPH.CENTER
)

doc.add_paragraph()

# ── Title candidates box ──
doc.add_heading("Alternative Title Candidates", level=3)
candidates = [
    "1. Deadline-Aware Multi-Robot Task Allocation via Centralized Training with Decentralized Execution",
    "2. Gossip-Mediated CTDE for Heterogeneous Multi-Robot Coordination Under Tidal Task Arrival",
    "3. Soft Deadline Constraints Enable Scalable Multi-Robot Task Allocation with CTDE",
    "4. Task-Attends-to-Agent: A CTDE Framework for Deadline-Constrained Heterogeneous MRTA",
]
for c in candidates:
    add_para(c, italic=True)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════
# ABSTRACT
# ══════════════════════════════════════════════════════════════════════
doc.add_heading("Abstract", level=1)

add_para(
    "Multi-robot task allocation (MRTA) under partial observability and deadline "
    "constraints remains a fundamental challenge for real-world deployment. Existing "
    "methods typically assume full information, batch task arrival, or hard deadline "
    "termination—conditions rarely met in logistics or disaster-response settings "
    "where tasks arrive continuously and expired tasks may still require completion. "
    "Here we propose a centralized-training-decentralized-execution (CTDE) framework "
    "that combines (1) a task-attends-to-agent cross-attention architecture for "
    "heterogeneous capability matching, (2) a gossip-based communication protocol with "
    "controlled information asymmetry, and (3) a three-tier reward design spanning "
    "step-level shaping, stage-level feedback, and episode-level terminal reward with "
    "soft deadline constraints."
)
add_placeholder(
    "[Across three agent scales (16–22 agents) and five task-load regimes, "
    "our method achieves a deadline success rate of X% under sustained tidal task "
    "injection, outperforming classical baselines (Hungarian, CBBA, Greedy-*) by "
    "Y percentage points. A 5-stage curriculum with out-of-distribution interleaving "
    "improves generalization by Z%.]"
)
add_para(
    "These results demonstrate that CTDE with soft constraints enables scalable, "
    "deadline-aware coordination under realistic information asymmetry, with "
    "implications for autonomous warehouse logistics, disaster response, and "
    "elastic manufacturing."
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════
# 1. INTRODUCTION
# ══════════════════════════════════════════════════════════════════════
doc.add_heading("1. Introduction", level=1)

# --- 1.1 Establish importance ---
doc.add_heading("1.1 The rise of multi-robot systems", level=2)

add_para(
    "Multi-robot systems are increasingly deployed across a range of economically "
    "significant domains, including warehouse logistics, manufacturing, precision "
    "agriculture, and disaster response (ref). The core coordination problem in these "
    "systems is multi-robot task allocation (MRTA): determining which robot should "
    "perform which task, at what time, to maximize overall system performance. The "
    "scale of real-world deployments underscores the practical importance of this "
    "problem—Amazon alone operates over 750,000 mobile robots across its fulfilment "
    "centres, and the warehouse automation market is projected to exceed USD 40 billion "
    "by 2030 (ref)."
)

add_para(
    "Despite decades of research on MRTA, three features of real-world deployments "
    "remain inadequately addressed by existing methods. First, tasks carry deadlines: "
    "a pallet must be picked before a truck departs, a damaged structure must be "
    "surveyed before a storm intensifies. Late completion loses value, but unlike the "
    "hard constraints assumed in most formal models, an expired task may still require "
    "completion rather than abandonment. Second, information is asymmetric: robots "
    "operate under range-limited communication and cannot maintain a globally consistent "
    "task list. Third, task arrival is sustained rather than batched—new orders, "
    "discoveries, or incidents arrive continuously, creating a regime of perpetual "
    "saturation in which the system never clears its queue. These three factors—"
    "deadlines, partial observability, and sustained arrival—together define what "
    "we term the tidal saturation regime."
)

# --- 1.2 Summarize what is known ---
doc.add_heading("1.2 Prior approaches to multi-robot task allocation", level=2)

add_para(
    "Classical approaches to MRTA can be broadly divided into optimization-based and "
    "heuristic methods. The Hungarian algorithm (Kuhn, 1955) provides an optimal "
    "assignment for the static, single-task, single-robot case but requires full "
    "information and batch formulation. Auction-based methods and the Consensus-Based "
    "Bundle Algorithm (CBBA; Choi, Brunet & How, 2009) introduce decentralisation "
    "through iterative bidding and consensus, yet they assume reliable communication "
    "and struggle under sustained task arrival where recomputation is expensive. "
    "Greedy heuristics—nearest-task, priority-ordered, coalition-forming—offer "
    "computational simplicity but sacrifice optimality and typically ignore deadline "
    "constraints. Across all of these methods, the combination of partial observability, "
    "soft deadlines, and sustained arrival has not been systematically addressed."
)

add_para(
    "Multi-agent reinforcement learning (MARL) has recently emerged as a promising "
    "alternative. The centralized-training-decentralized-execution (CTDE) paradigm, "
    "in which a centralized critic accesses global state during training while actors "
    "rely only on local observations at execution time, has been particularly "
    "influential. Algorithms such as MADDPG (Lowe et al., 2017), QMIX (Rashid et al., "
    "2018), and MAPPO (Yu et al., 2022) have demonstrated strong performance in "
    "cooperative multi-agent settings. However, the application of CTDE to MRTA "
    "remains limited. Existing work has largely focused on episodic task completion "
    "with homogeneous agents and batch task arrival (ref). Heterogeneous capability "
    "matching—where agents possess distinct ability profiles and tasks require "
    "specific combinations of capabilities—introduces a combinatorial assignment "
    "problem that standard MARL architectures do not natively address. Furthermore, "
    "multi-stage tasks that transition between requirement profiles, soft deadline "
    "constraints that preserve credit assignment, and sustained tidal arrival have not, "
    "to our knowledge, been combined within a single CTDE framework."
)

# --- 1.3 Identify the gap ---
doc.add_heading("1.3 The three-way gap", level=2)

add_para(
    "No prior work simultaneously addresses the three challenges that define the tidal "
    "saturation regime: (a) soft deadline constraints, under which expired tasks remain "
    "actionable and contribute partial reward rather than being terminated; "
    "(b) information asymmetry produced by a realistic range-limited communication "
    "model, where only a fraction of agents are directly notified of new tasks and "
    "the remainder must acquire knowledge through peer-to-peer gossip; and "
    "(c) sustained tidal task injection, which creates a steady-state congestion "
    "regime fundamentally different from batch-clearance formulations. The closest "
    "related work, TARGETNET (ref), employs hard penalty signals for deadline "
    "violation, which can destabilise credit assignment in policy-gradient training "
    "when tasks are abruptly terminated."
)

# --- 1.4 State the study aim ---
doc.add_heading("1.4 Study aim", level=2)

add_para(
    "We investigate whether a CTDE framework combining proximal policy optimisation "
    "(PPO) with a task-attends-to-agent attention architecture, gossip-based "
    "communication under controlled information asymmetry, and three-tier reward "
    "shaping can coordinate heterogeneous multi-robot teams under sustained tidal "
    "task arrival with soft deadline constraints. Specifically, we ask: can such a "
    "framework learn deadline-aware prioritisation, propagate task information "
    "efficiently through a partially connected communication graph, and scale to "
    "teams of 16–22 agents without performance collapse?"
)

# --- 1.5 Indicate approach ---
doc.add_heading("1.5 Contributions", level=2)

add_para(
    "This work makes four contributions:"
)

contributions = [
    "Reversed cross-attention architecture. We introduce a task-attends-to-agent "
    "attention mechanism in which task tokens query agent tokens (rather than the "
    "conventional agent-attends-to-task direction). This inductive bias allows each "
    "task to independently identify suitable agents from a heterogeneous pool, "
    "naturally handling the combinatorial matching problem at the core of "
    "heterogeneous MRTA.",

    "Gossip protocol with controlled information asymmetry. We implement a "
    "timestamp-synchronised, range-limited gossip protocol in which only 20% of "
    "agents receive direct notification of new tidal tasks. The remaining 80% acquire "
    "task knowledge through incremental peer-to-peer exchange. We demonstrate that "
    "this protocol achieves near-optimal information propagation at substantially "
    "reduced communication cost.",

    "Three-tier reward shaping for multi-timescale credit assignment. We design a "
    "reward structure operating at three timescales: step-level shaping (congestion, "
    "loneliness, urgency-modulated), stage-level intermediate feedback (backfilled "
    "at stage transitions within multi-stage tasks), and episode-level terminal "
    "reward with a 10× gradient between on-time and expired completion. This "
    "three-tier design resolves the credit assignment problem across sub-second "
    "action selection and hour-long episodes.",

    "Five-stage curriculum with out-of-distribution interleaving. We develop a "
    "training curriculum that progressively increases difficulty across five "
    "dimensions—task count, duration, deadline tightness, communication range, "
    "and discovery probability—while interleaving out-of-distribution samples "
    "to prevent catastrophic forgetting and improve generalisation to unseen "
    "conditions.",
]
for i, c in enumerate(contributions):
    p = doc.add_paragraph()
    run = p.add_run(f"({i + 1}) ")
    run.bold = True
    p.add_run(c)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════
# 2. MATERIALS AND METHODS
# ══════════════════════════════════════════════════════════════════════
doc.add_heading("2. Materials and Methods", level=1)

# --- 2.1 Problem formulation ---
doc.add_heading("2.1 Problem formulation", level=2)

add_para(
    "We formalise heterogeneous MRTA under tidal arrival as a Dec-POMDP defined by "
    "the tuple ⟨N, S, {Oᵢ}, {Aᵢ}, P, R, γ⟩, where:"
)

items = [
    "N agents. Each agent i belongs to a species that determines its binary ability "
    "vector bᵢ ∈ {0,1}⁵, where each dimension represents a distinct "
    "capability (e.g., lifting, sensing, communication, manipulation, computation). "
    "An agent can contribute at most one capability dimension to one task at a time.",

    "State S. The full state includes continuous 2D agent positions, discrete agent "
    "states (idle, moving, working), the task set with per-task requirements, current "
    "stage progress, absolute deadlines, current member assignments, and task origin "
    "labels (initial, tidal, discovered).",

    "Observations Oᵢ. Agent i receives an ego-centric local observation consisting "
    "of: self-state (10 dimensions: abilities, position, current task, travel distance, "
    "waiting time); neighbour features (up to 20 neighbours × 10 dimensions: "
    "abilities, position, current task, assignment status, travel distance); and task "
    "features (variable count × 20 dimensions: current status, future requirements, "
    "relative location, deadline urgency, complexity, member count, information "
    "freshness indicators). A depot option with zero task features is always available.",

    "Actions Aᵢ. At each decision step, agent i selects a (task, capability "
    "dimension) pair or remains idle. Illegal actions—tasks for which the agent "
    "lacks the required capability, tasks already completed, or agents in cooldown—"
    "are masked during both training and execution.",

    "Transition P. Agent movement and work progress are deterministic. Stochastic "
    "elements arise from tidal task injection, dynamic task discovery, and "
    "communication graph structure.",

    "Reward R. Three-tier design detailed in Section 2.4.",

    "γ = 0.99. Discount factor.",
]
for item in items:
    p = doc.add_paragraph(item, style="List Bullet")

# --- 2.2 Environment design ---
doc.add_heading("2.2 Environment design", level=2)

doc.add_heading("Task model", level=3)
add_para(
    "Each task consists of 1–4 sequential stages (uniform random). Every stage "
    "is defined by: (i) a binary requirement vector r ∈ {0,1}⁵ with 1–3 "
    "active dimensions drawn uniformly, specifying which capabilities are needed for "
    "that stage; and (ii) a work duration d = (base_time + complexity × "
    "duration_scale × 2 + noise) / n_stages, with a minimum of 2.0 time units "
    "per stage. Stage completion requires all active requirement dimensions to be "
    "covered by present agents and the work duration to elapse. On stage transition, "
    "agents whose abilities do not match the next stage’s requirements are "
    "automatically released for reallocation; compatible agents continue to the next "
    "stage. The next stage’s requirements are broadcast to current members."
)

doc.add_heading("Deadline assignment", level=3)
add_para(
    "Each task receives an absolute deadline at creation: deadline = current_time + "
    "max(estimated × tightness, estimated × 0.95), where estimated = "
    "min_travel_time + total_work + 10.0. The tightness parameter varies across "
    "curriculum stages (see Section 2.4). Critically, deadlines are soft constraints: "
    "an expired task is neither deleted nor forcibly terminated. It remains visible, "
    "selectable, and completable, but expired completion earns only 10% of the on-time "
    "completion reward. A decaying shaping incentive (decay = max(0.05, 1.0 − "
    "overdue/100)) continues to encourage work on expired but still-valuable tasks. "
    "This design avoids the credit assignment instability that arises when tasks are "
    "hard-terminated at their deadline."
)

doc.add_heading("Tidal task injection", level=3)
add_para(
    "Every 200.0 time units, a tidal burst of 8–15 new tasks (uniform random) "
    "is injected into the environment. Each task receives an independent deadline "
    "computed from the current time. Only 20% of agents (randomly selected) are "
    "directly notified of new tidal tasks; the remaining 80% must acquire this "
    "knowledge through the gossip protocol. This creates a controlled information "
    "asymmetry that mirrors real-world conditions—for instance, only robots near "
    "a receiving dock may be directly aware of newly arrived orders. With a 3,000-step "
    "horizon, approximately 15 tidal waves occur per episode, generating roughly "
    "180 total tasks alongside 8–40 initial tasks. This sustained injection "
    "creates a perpetual saturation regime in which the task queue is never fully "
    "cleared."
)

doc.add_heading("Dynamic task discovery", level=3)
add_para(
    "At each environment step, a new task may be discovered at a random location with "
    "probability that increases across curriculum stages (0.01 → 0.08). Discovered "
    "tasks are known only to agents within communication range of the discovery "
    "location, and subsequently propagate through gossip."
)

doc.add_heading("Agent model", level=3)
add_para(
    "Each agent belongs to one of 3–5 species (depending on scale configuration; "
    "see Section 2.5). Species determines the agent’s 5-dimensional binary ability "
    "vector. The agent follows a finite state machine: idle → moving (toward "
    "selected task) → working (contributing one capability dimension) → "
    "idle. At most two agents can simultaneously work on the same task. A working "
    "timeout mechanism prevents deadlock: if an agent works continuously beyond a "
    "threshold without stage completion, it enters a cooldown period and is released "
    "for reallocation."
)

doc.add_heading("Gossip communication protocol", level=3)
add_para(
    "Communication proceeds in three stages at each gossip step. (1) Graph "
    "construction: an N × N adjacency matrix is computed via vectorised Euclidean "
    "distance with a threshold comm_range (curriculum-varying: 0.50 → 0.25). "
    "(2) Self-sensing: every agent independently detects all tasks within its "
    "communication range through broadcasting. (3) One-hop exchange: for each "
    "connected neighbour pair, agents bidirectionally exchange (i) novel task IDs "
    "(set difference of known tasks), (ii) novel completion IDs, and (iii) status "
    "and timestamp synchronisation for mutually known tasks, in which the agent with "
    "the more recent task_last_update_time overwrites the other’s cached status, "
    "requirements, and member information. To reduce computational overhead, gossip "
    "runs every gossip_interval = 2 environment steps, yielding approximately 40% "
    "communication reduction with negligible performance degradation."
)

# --- 2.3 Network architecture ---
doc.add_heading("2.3 Network architecture", level=2)

doc.add_heading("Actor (decentralised policy)", level=3)
add_para(
    "The actor processes local observations through a seven-stage pipeline:"
)
steps = [
    "Self encoder. A linear projection (10 → 128) with ReLU activation encodes "
    "the agent’s self-state into a hidden representation.",

    "Neighbour attention. A 4-head multi-head attention (MHA) module uses the self "
    "representation as query over neighbour tokens (up to 20 neighbours, 10 "
    "dimensions each). A padding mask handles variable neighbourhood sizes.",

    "Agent context formation. The self representation and neighbour attention output "
    "are aggregated to form a unified agent context vector.",

    "Task count modulation. A learned embedding of valid_task_count / 100.0, scaled "
    "by 0.1, is added to the agent context (FiLM-style conditioning; Perez et al., "
    "2018). This makes the policy aware of how many valid task options exist.",

    "Task encoding. All task observations are projected through a shared linear layer "
    "(20 → 128) with ReLU. Sinusoidal positional encoding is added with weight 0.1.",

    "Cross-attention (CTDE core). Task tokens serve as queries; the agent context "
    "(self + neighbours) serves as keys and values in a 4-head MHA module. This "
    "reversed attention direction—task attends to agent—is the central "
    "architectural innovation. Each task independently evaluates the suitability of "
    "the agent pool, learning to route itself to the most appropriate agent for its "
    "currently unmet capability requirements.",

    "Logits head and action selection. A residual connection combines task embeddings "
    "with attention output. A single linear layer (128 → 1) produces a scalar "
    "logit per task. Illegal actions are masked with −10⁹. A softmax over "
    "valid actions produces the final action distribution.",
]
for i, s in enumerate(steps):
    p = doc.add_paragraph()
    run = p.add_run(f"({i + 1}) ")
    run.bold = True
    p.add_run(s)

doc.add_heading("Critic (centralised value function)", level=3)
add_para(
    "The critic receives the full global state during training only. Agent features "
    "(13 dimensions) and task features (23 dimensions) are projected through separate "
    "linear layers to 128-dimensional embeddings. All embeddings are concatenated into "
    "a single sequence [h_agents; h_tasks]. A 4-head self-attention module operates "
    "over the full sequence, enabling the critic to model interactions between all "
    "entities. Masked mean pooling aggregates valid (non-padding) positions into a "
    "fixed-length representation. A two-layer MLP (128 → 64 → 1) produces "
    "the scalar state value."
)

doc.add_heading("Key architectural hyperparameters", level=3)
add_para(
    "Hidden dimension = 128 throughout. Attention heads = 4. Dropout probability = "
    "0.15 applied to attention and MLP layers. Actor KV mode = tokens (self + "
    "neighbours as multi-token key-value context), rather than a summary vector, to "
    "preserve individual agent identity during cross-attention."
)

# --- 2.4 Training ---
doc.add_heading("2.4 Training procedure", level=2)

doc.add_heading("PPO configuration", level=3)
add_para(
    "We train with Proximal Policy Optimisation (PPO; Schulman et al., 2017) using "
    "eight parallel environments with ThreadPoolExecutor-based vectorisation. The "
    "update step size is 8,192 environment steps, with mini-batches of 1,024 and "
    "K = 3 epochs per update. Generalised Advantage Estimation (GAE; Schulman et al., "
    "2016) uses λ = 0.95 and γ = 0.99, computed with per-episode segmentation "
    "to prevent advantage leakage across episode boundaries. The PPO clip range is "
    "ε = 0.2. Gradient norms are clipped at 1.0. The Adam optimiser is used "
    "with a learning rate that decays linearly from 3 × 10⁻⁴ (episode "
    "0–10,000) to 5 × 10⁻⁵ (episode 15,000). The entropy "
    "coefficient is linearly annealed from 0.02 (episode 0) to 0.003 (episode 8,000) "
    "to encourage exploration in early training."
)

doc.add_heading("Three-tier reward design", level=3)
add_para(
    "The reward function operates at three distinct timescales to address the credit "
    "assignment problem across sub-second decisions and hour-long episodes.",
    bold=False
)

add_para("Step-level shaping (applied at each action selection, globally scaled "
         "×0.25):", bold=True)
shaping_items = [
    "Congestion penalty (−0.15): applied per agent when the number of agents "
    "assigned to a task exceeds the remaining requirement dimensions.",
    "Loneliness bonus (+0.10 × urgency): rewarded when the agent fills a "
    "requirement dimension that would otherwise remain unmet.",
    "Distance penalty (−0.05 × dist²): quadratic in normalised "
    "distance, significant only for distant tasks to avoid penalising local decisions.",
    "Capability match (+0.05 × urgency): small bonus when the agent possesses "
    "at least one ability matching the task’s current requirements.",
    "Redundancy penalty (−0.10 × 0.33): reduced penalty for contributing "
    "a capability already covered by other agents.",
    "Expired incentive (+0.05 × decay): a decaying bonus (decay = max(0.05, "
    "1.0 − overdue/100)) for continuing work on expired but still-valuable tasks.",
    "Urgency modulation: all shaping terms are multiplied by (1.5 − urgency), "
    "where urgency = tanh(time_until_deadline / 50). This progressively amplifies "
    "shaping signals as deadlines approach.",
]
for item in shaping_items:
    doc.add_paragraph(item, style="List Bullet")

add_para("Stage-level intermediate reward:", bold=True)
add_para(
    "When a multi-stage task completes its current stage but the task is not yet "
    "fully finished, each current member receives +5.0 × urgency. This reward "
    "is backfilled to the agent’s most recent decision step in the PPO buffer, "
    "providing intermediate credit assignment that bridges the gap between step-level "
    "shaping and episode-level terminal reward."
)

add_para("Episode-level terminal reward (spread to all steps via weighted "
         "distribution):", bold=True)
add_para(
    "The terminal reward is distributed to all steps of the episode using a linearly "
    "increasing weight vector (weights = arange(1, ep_len + 1), normalised to sum to "
    "1), reflecting that later decisions have proportionally greater influence on "
    "terminal outcomes. The seven reward components are:"
)
terminal_items = [
    "Completion reward (coefficient A = 600): (n_on_time + 0.1 × n_late) / "
    "n_total, providing a 10× gradient between on-time and expired completion.",
    "Speed reward (λ = 120): bonus for finishing tasks early relative to the "
    "episode horizon.",
    "Slack reward (λ = 120): sum of (deadline − finish_time) for on-time "
    "completions, rewarding the accumulation of temporal headroom.",
    "Unfinished penalty (η = 40): −n_unfinished / n_total.",
    "Expired penalty (ω = 400): −n_expired / n_total. This is the largest "
    "coefficient in the terminal reward, reflecting the importance of deadline "
    "compliance while maintaining the soft-constraint property.",
    "Stage completion (γ = 20): proportion of total stages completed across all "
    "multi-stage tasks.",
    "Makespan penalty (μ = 15): −final_completion_time.",
    "Failed attempts (φ = 1): −n_failed per agent.",
]
for item in terminal_items:
    doc.add_paragraph(item, style="List Bullet")

doc.add_heading("Five-stage curriculum", level=3)
add_para(
    "Training proceeds through five curriculum stages that progressively increase "
    "difficulty across five dimensions. Domain randomisation adds ±20% uniform "
    "jitter to communication range, task duration, and task count per episode. "
    "Out-of-distribution (OOD) sprinkling applies with 20% probability per episode "
    "during stages 4–5, sampling tasks from the 35–50 range to encourage "
    "generalisation to unseen high-load conditions. Uniform task-pool sampling "
    "(full-range interleaving) within each stage prevents catastrophic forgetting "
    "of earlier conditions. The curriculum parameters are summarised in Table 1."
)

# ── Curriculum table ──
add_para("Table 1. Five-stage curriculum parameters.", bold=True, italic=True)
table = doc.add_table(rows=7, cols=6)
table.style = "Light Grid Accent 1"
headers = ["Stage", "Episodes", "Task range", "Duration ×",
           "Deadline tightness", "Comm. range"]
for i, h in enumerate(headers):
    cell = table.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        for run in p.runs:
            run.bold = True
            run.font.size = Pt(9)

data = [
    ["1 (warm-up)", "0–500", "8–15", "3", "1.5–2.5", "0.50"],
    ["2 (ramp)", "500–5,000", "15–25", "5", "1.2–2.0", "0.40"],
    ["3 (advance)", "5,000–11,000", "20–30", "7", "1.0–1.5", "0.35"],
    ["4 (main)", "11,000–15,000", "25–35", "9", "1.0–1.5", "0.28"],
    ["5 (high-pressure)", "15,000+", "30–40", "10", "1.0–1.3", "0.25"],
    ["OOD (sprinkling)", "20% prob.", "35–50", "—", "—", "—"],
]
for r, row_data in enumerate(data):
    for c, val in enumerate(row_data):
        cell = table.rows[r + 1].cells[c]
        cell.text = val
        for p in cell.paragraphs:
            for run in p.runs:
                run.font.size = Pt(9)

doc.add_paragraph()  # spacer

doc.add_heading("Hardware and acceleration", level=3)
add_para(
    "All experiments were conducted on a single workstation with an Intel Core "
    "i5-14600KF processor (14 cores, 12 threads allocated to environment "
    "parallelisation) and an NVIDIA RTX 50-series GPU with 12 GB VRAM. Automatic "
    "mixed precision (AMP) training uses the bfloat16 data type. TF32 is enabled "
    "on supported hardware. torch.inference_mode() is active during rollout "
    "sampling. Training throughput is approximately 190–250 environment steps "
    "per second, with the environment stepping (∼54–80% of wall-clock "
    "time) dominating over network inference during later curriculum stages when "
    "task counts are high."
)

# --- 2.5 Evaluation protocol ---
doc.add_heading("2.5 Evaluation protocol", level=2)

doc.add_heading("Scale configurations", level=3)
add_para(
    "We evaluate across three agent scales, designed to maintain an agent-to-task "
    "ratio appropriate for the tidal saturation regime (Table 2)."
)

add_para("Table 2. Evaluation scale configurations.", bold=True, italic=True)
table2 = doc.add_table(rows=4, cols=4)
table2.style = "Light Grid Accent 1"
for i, h in enumerate(["Scale", "Species count", "Agents per species", "Mean total agents"]):
    cell = table2.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        for run in p.runs:
            run.bold = True
            run.font.size = Pt(9)
scale_data = [
    ["Small", "3–4", "4–5", "16"],
    ["Medium", "3–4", "5–7", "21"],
    ["Large", "4–5", "5–7", "22"],
]
for r, row_data in enumerate(scale_data):
    for c, val in enumerate(row_data):
        cell = table2.rows[r + 1].cells[c]
        cell.text = val
        for p in cell.paragraphs:
            for run in p.runs:
                run.font.size = Pt(9)

doc.add_paragraph()

add_para(
    "Five task-load levels are evaluated at each scale: [8–15], [15–25], "
    "[20–30], [30–40], and [40–50]. Each of the resulting 15 "
    "configurations (3 scales × 5 loads) is tested over 100 episodes with "
    "fixed random seeds, yielding 1,500 evaluation episodes in total. Testing uses "
    "the trained policy in deterministic mode (argmax action selection, no exploration "
    "noise)."
)

doc.add_heading("Metrics", level=3)
add_para("We report six primary metrics per episode:")
metric_items = [
    "Deadline Success Rate (DL-SR). Proportion of tasks completed before their "
    "absolute deadline. This is the primary performance indicator, as it jointly "
    "measures coordination quality and temporal awareness.",

    "Task Completion Rate. Proportion of tasks completed at any time, including "
    "post-deadline. This measures the policy’s ability to avoid task "
    "abandonment.",

    "Expired Rate. Proportion of tasks that exceeded their deadline (the complement "
    "of the set of on-time + unfinished tasks).",

    "Average Slack. Mean of (deadline − finish_time) across on-time completions, "
    "measured in environment time units. Higher slack indicates more temporal headroom.",

    "Resource Utilisation. Proportion of agent-steps spent in the working state, "
    "aggregated across all agents.",

    "Idle Rate. Proportion of agent-steps spent idle.",

    "By-source disaggregation. Each of the above metrics is computed separately for "
    "initial tasks, tidal tasks, and discovered tasks, enabling analysis of how "
    "information asymmetry differentially affects performance by task origin.",
]
for item in metric_items:
    doc.add_paragraph(item, style="List Bullet")

doc.add_heading("Baselines", level=3)
add_para(
    "We compare against five classical MRTA methods, each receiving the full task set "
    "(no information asymmetry), providing an upper-bound reference for coordination "
    "quality under perfect information:"
)
baseline_items = [
    "Hungarian algorithm (Kuhn, 1955). Optimal minimum-cost assignment for the "
    "single-task, single-robot case, solved on the static task set.",
    "Consensus-Based Bundle Algorithm (CBBA; Choi, Brunet & How, 2009). Decentralised "
    "auction with consensus-based conflict resolution.",
    "Greedy-Nearest. Each idle agent selects the nearest task for which it can "
    "contribute at least one capability.",
    "Greedy-Coalition. Agents are assigned to tasks as a coalition; greedy selection "
    "by distance.",
    "Greedy-Priority. Tasks ranked by deadline urgency; agents assigned greedily by "
    "capability match within priority order.",
]
for item in baseline_items:
    doc.add_paragraph(item, style="List Bullet")

# --- 2.6 Ablation studies ---
doc.add_heading("2.6 Ablation studies", level=2)

add_para(
    "To isolate the contribution of each component, we conduct the following "
    "controlled ablation experiments (Table 3). Each ablation varies one factor "
    "while holding all other parameters at their stage-4 curriculum defaults."
)

add_para("Table 3. Ablation study design.", bold=True, italic=True)
table3 = doc.add_table(rows=9, cols=3)
table3.style = "Light Grid Accent 1"
for i, h in enumerate(["Ablation", "Variable", "Range"]):
    cell = table3.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        for run in p.runs:
            run.bold = True
            run.font.size = Pt(9)
ablation_data = [
    ["Information asymmetry", "Tidal notification ratio", "10%, 20%, 50%, 100%"],
    ["Deadline constraint type", "Soft (ours) vs. hard termination", "Binary"],
    ["Deadline tightness", "Tightness multiplier", "0.8, 1.0, 1.2, 1.5, 2.0, 2.5"],
    ["Gossip interval", "Steps between gossip rounds", "1, 2, 4, 8"],
    ["Timestamp synchronisation", "On/off within gossip", "Binary"],
    ["Curriculum type", "None / sequential / mixed + OOD", "3 variants"],
    ["Attention direction", "Task → Agent (ours) vs. Agent → Task", "Binary"],
    ["Agent scale", "Number of agents", "8, 12, 16, 20, 24, 28, 32"],
]
for r, row_data in enumerate(ablation_data):
    for c, val in enumerate(row_data):
        cell = table3.rows[r + 1].cells[c]
        cell.text = val
        for p in cell.paragraphs:
            for run in p.runs:
                run.font.size = Pt(9)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════
# 3. DATA AND CODE AVAILABILITY
# ══════════════════════════════════════════════════════════════════════
doc.add_heading("3. Data and Code Availability", level=1)

items_avail = [
    "Code. The full implementation, including environment, network architectures, "
    "training driver, and evaluation scripts, is available at [repository URL] under "
    "the MIT License.",
    "Trained checkpoints. Model weights for all reported configurations are archived "
    "at [URL] for exact reproducibility of reported results.",
    "Test sets. The RALTestSet-1 and RALTestSet-2 evaluation environments are "
    "provided in the repository as serialised pickle files.",
    "Training logs. TensorBoard event files containing all reported training curves "
    "are available in the runs/ directory of the repository.",
    "Figure source data. Aggregated metric values underlying all figures and tables "
    "are provided as CSV files in the paper/ directory.",
]
for item in items_avail:
    doc.add_paragraph(item, style="List Bullet")

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════
# REFERENCES
# ══════════════════════════════════════════════════════════════════════
doc.add_heading("References", level=1)

references = [
    "Gerkey, B. P. & Matarić, M. J. A formal analysis and taxonomy of task "
    "allocation in multi-robot systems. Int. J. Rob. Res. 23, 939–954 (2004).",

    "Korsah, G. A., Stentz, A. & Dias, M. B. A comprehensive taxonomy for "
    "multi-robot task allocation. Int. J. Rob. Res. 32, 1495–1512 (2013).",

    "Choi, H.-L., Brunet, L. & How, J. P. Consensus-based decentralized auctions "
    "for robust task allocation. IEEE Trans. Robot. 25, 912–926 (2009).",

    "Kuhn, H. W. The Hungarian method for the assignment problem. Nav. Res. Logist. "
    "Q. 2, 83–97 (1955).",

    "Lowe, R. et al. Multi-agent actor-critic for mixed cooperative-competitive "
    "environments. Adv. Neural Inf. Process. Syst. 30 (2017).",

    "Rashid, T. et al. QMIX: Monotonic value function factorisation for deep "
    "multi-agent reinforcement learning. Proc. 35th Int. Conf. Mach. Learn. (2018).",

    "Yu, C. et al. The surprising effectiveness of PPO in cooperative multi-agent "
    "games. Adv. Neural Inf. Process. Syst. 35 (2022).",

    "Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. Proximal "
    "policy optimization algorithms. arXiv:1707.06347 (2017).",

    "Schulman, J., Moritz, P., Levine, S., Jordan, M. & Abbeel, P. High-dimensional "
    "continuous control using generalized advantage estimation. arXiv:1506.02438 (2016).",

    "Perez, E., Strub, F., de Vries, H., Dumoulin, V. & Courville, A. FiLM: Visual "
    "reasoning with a general conditioning layer. Proc. AAAI Conf. Artif. Intell. "
    "32 (2018).",

    "[Additional references to be added for: deadline-constrained MRTA, TARGETNET, "
    "gossip protocols, multi-stage task allocation, heterogeneous MRTA with MARL]",
]
for ref in references:
    p = doc.add_paragraph()
    run = p.add_run(ref)
    run.font.size = Pt(10)
    p.paragraph_format.left_indent = Cm(1.27)
    p.paragraph_format.first_line_indent = Cm(-1.27)

# ══════════════════════════════════════════════════════════════════════
# DATA PLACEHOLDER SECTIONS (to be filled after experiments)
# ══════════════════════════════════════════════════════════════════════
doc.add_page_break()
doc.add_heading("Sections Requiring Experimental Data", level=1)

add_para(
    "The following sections depend on experimental results and will be written after "
    "data collection is complete. The structure is provided below as a writing guide.",
    italic=True
)

doc.add_heading("4. Results [placeholder]", level=1)

result_sections = [
    ("4.1 Overall performance across scales and task loads",
     "3×5 grid (scales × loads), grouped bar chart, Table of all metrics "
     "vs. baselines. Primary claim: CTDE outperforms classical baselines on DL-SR "
     "under tidal regime."),
    ("4.2 Information asymmetry and gossip compensation",
     "Line plot: DL-SR vs. notification ratio, with/without gossip. Primary claim: "
     "gossip compensates for partial notification; performance saturates at moderate "
     "ratios."),
    ("4.3 Deadline sensitivity: soft vs. hard constraints",
     "Dual-axis plot: DL-SR + completion rate vs. tightness. Primary claim: soft "
     "constraint maintains non-zero completion even under tightest deadlines."),
    ("4.4 Multi-stage task efficiency",
     "Histogram of stage gap times, with/without release mechanism. Primary claim: "
     "stage-switching reduces idle gaps."),
    ("4.5 Communication efficiency",
     "Table: gossip interval sweep with message count, staleness, and DL-SR. Primary "
     "claim: interval = 2 achieves near-optimal trade-off."),
    ("4.6 Curriculum effectiveness",
     "Learning curves: DL-SR over episodes, three curriculum variants. Primary claim: "
     "mixed curriculum + OOD prevents catastrophic forgetting."),
    ("4.7 Scalability analysis",
     "Dual-panel: DL-SR + utilisation vs. agent count (8–32). Primary claim: "
     "DL-SR plateaus; utilisation decreases with agent count beyond saturation."),
]
for title, desc in result_sections:
    doc.add_heading(title, level=2)
    add_placeholder(desc)

doc.add_heading("5. Discussion [placeholder]", level=1)

disc_sections = [
    ("5.1 Main finding",
     "CTDE + PPO with soft constraints achieves deadline-aware coordination under "
     "tidal saturation."),
    ("5.2 Mechanism interpretation",
     "Why task-attends-to-agent attention works for heterogeneous matching. Why "
     "three-tier reward resolves multi-timescale credit assignment. Why soft deadline "
     "avoids training instability."),
    ("5.3 Comparison with prior work",
     "vs. CBBA/auctions, vs. MAPPO/QMIX, vs. TARGETNET."),
    ("5.4 Limitations",
     "High agent idle rates (70–80%). Species capability coverage. Single shared "
     "policy. Fixed communication range."),
    ("5.5 Implications",
     "Practical for warehouse/logistics, extensible to energy-constrained and "
     "adversarial settings."),
    ("5.6 Future work",
     "Dynamic coalition formation, learned communication, role specialisation, "
     "energy-constrained agents."),
]
for title, desc in disc_sections:
    doc.add_heading(title, level=2)
    add_placeholder(desc)

# ── Save ──
output_path = "D:/code/task-03-修改为CTDE范式/paper_draft.docx"
doc.save(output_path)
print(f"Document saved to: {output_path}")
