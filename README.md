# EvalForge

Continuous evaluation infrastructure for coding agents.

Running a benchmark once and reading off an accuracy number does not tell you whether an
agent is good. Two agents can reach the same final answer while behaving completely
differently — one reads three files and makes a minimal fix, the other rewrites fourteen
files, breaks the build, and stumbles into a passing test suite. EvalForge evaluates both
the **outcome** and the **trajectory**, and it takes seriously the question most eval
harnesses skip: *are these measurements actually telling you anything?*

## What makes this different

Most agent-eval projects stop at "run the tasks, print the score". The work here goes into
evaluation quality:

- **Validated judges.** An LLM judge is itself a model that can be wrong. EvalForge scores
  the judge against human labels (Cohen's κ, per-class precision/recall) and probes it for
  position bias and self-preference. A judge whose agreement drops below threshold fails
  the build.
- **Statistically honest regressions.** 68% versus 72% on 50 tasks is noise. Comparisons
  use paired bootstrap confidence intervals and McNemar's test on per-case outcomes, with
  a power calculation telling you how many tasks you would need to detect the effect you
  care about.
- **Variance as a first-class signal.** k samples per case, `pass@k` for capability and
  `pass^k` for stability, keeping agent nondeterminism separate from flaky environments.
- **Actionable failure analysis.** Deterministic clustering over structured failure
  signals, so a regression reads *"concentrated in cluster: abandons after first failed
  test"* rather than *"−4%"*.
- **Cost-aware evaluation.** Budget-constrained subset selection picks the most
  discriminative cases per dollar, because a full suite on every commit is not affordable.

The distributed-execution semantics that matter — bounded concurrency, timeouts, retries,
cancellation, partial results — are implemented behind an `ExecutorBackend` protocol, so a
Celery or Kubernetes backend is a swap rather than a rewrite. It ships as an asyncio
scheduler rather than an ops stack, on purpose.

## Design notes

**The trajectory is the spine.** Every agent action becomes a typed event —
`ModelCall`, `ToolCall`, `FileEdit`, `CommandRun`, `SafetyViolation`, `Submission`. Every
evaluator reads that trace. Nothing re-derives facts by scraping logs.

**Safety is observed, not asserted.** The workspace resolves every path an agent touches
and checks containment. Escape attempts are *recorded* as `SafetyViolation` events rather
than raised, so a containment breach produces a measurable safety metric instead of an
error that looks like a harness bug.

**Reproducibility by content hash.** Datasets, evaluator suites and agent configs each
hash their canonical JSON, and every run records all three. "Agent v1.4 on dataset v3 with
suite v2" is a checkable claim, not a hopeful one.

**Everything runs free by default.** The default agent is deterministic and makes no API
calls, so the full test suite and the CI eval gate cost nothing. Real model runs are
opt-in, default to the cheapest capable model, and sit behind an on-disk response cache
and a hard per-run spend ceiling.

## Status

Under active construction, one verified milestone at a time. Each milestone has to pass
lint, strict types, the full test suite and its own acceptance run before the next starts.

- **M1 — foundations** ✅ schema with content hashing, sandboxed workspace, deterministic
  agent, asyncio scheduler, outcome/patch/trajectory evaluators, SQLite store, seeded-bug
  dataset generator, CLI.
- **M2 — real agents** ✅ provider-agnostic model layer, tool-use loop, response cache,
  resource budgets, rate limiting, replay, efficiency and safety evaluators.
- M3 — statistics, k-sampling, regression detection
- M4 — LLM judge and judge validation
- M5 — API and dashboard with trajectory viewer
- M6 — failure clustering, cost-aware selection, CI eval gate

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

evalforge dataset build --out datasets/synth --cases 30 --seed 7
evalforge dataset verify datasets/synth          # re-hash; drift is an error

evalforge run --dataset synth@v1 --agent scripted:baseline
evalforge runs                                   # what has been measured
evalforge show <run-id>                          # reload a run in full
evalforge trajectory <run-id> <case-id>          # every recorded event
evalforge doctor                                 # what this machine can enforce
```

Five deterministic agents ship, all free to run: `scripted:oracle` (solves everything —
the upper bound), `scripted:baseline`, `scripted:regressed` (a planted regression in both
outcome and behaviour), `scripted:idle` (the lower bound), and `scripted:malicious`, which
solves the task while attempting a workspace escape — the control for the safety evaluator.

### Running a real model

The model layer speaks the chat-completions wire protocol directly, so it reaches Groq,
OpenRouter, GitHub Models or a local Ollama by changing a URL rather than a code path.

```bash
cp .env.example .env        # then paste your key in; .env is git-ignored
evalforge run --dataset synth@v1 --agent model --max-model-tokens 200000
evalforge run --dataset synth@v1 --agent model:llama-3.3-70b-versatile
```

An exported `GROQ_API_KEY` always beats the file, so overriding it for a single
run works as expected.

Three things make this safe to point at a free tier. Requests are **paced proactively**
against a per-minute allowance rather than firing concurrently and absorbing 429s, with
`Retry-After` honoured when one arrives anyway. Responses are **cached on disk**, so a
repeated run makes no requests, spends no quota, and is reproducible in a way that
sampling never is. And **budgets refuse the call that would cross a ceiling** rather than
reporting the overrun afterwards — denominated in steps and tokens, because on a free tier
those are what run out, not dollars.

```bash
evalforge run --agent scripted:malicious --suite strict   # safety gates, not just tests
```

## Development

```bash
ruff check .      # lint
mypy src          # types (strict)
pytest -q         # tests — offline, no API key needed
```

## Licence

MIT
