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
- **M3 — statistics** ✅ Wilson intervals, paired bootstrap, McNemar's exact test,
  power analysis, pass@k vs pass^k, and `evalforge compare` with a verdict rather than
  a delta.
- **M4 — judge validation** ✅ Cohen's κ against human labels, per-class precision and
  recall, position-bias and self-preference probes, judge fingerprinting, and
  `evalforge judge validate` as a build gate.
- **M5 — API and dashboard** ✅ read-only FastAPI over the store, plus a single-page
  dashboard with run detail, a comparison view and a trajectory viewer. No build step.
- **M6 — clustering, selection, CI gate** ✅ deterministic failure clustering,
  budget-constrained case selection, and an `evalforge gate` command wired into a
  GitHub Actions workflow that costs nothing to run.

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

### Comparing two runs

```bash
evalforge run --agent scripted:oracle    # baseline
evalforge run --agent scripted:baseline  # candidate
evalforge compare <run-a> <run-b> --fail-on-regression
```

A comparison reports a verdict and its evidence, never a bare delta:

```
verdict: regression
before        100.0%
after          76.7%
difference    -23.3% [-40.0, -10.0]
p-value       0.0156
paired outcomes: 0 fixed, 7 broken, 23 stable pass, 0 stable fail
broke: missing_tiebreak-007, shared_mutable_state-002, ...
```

When nothing reaches significance it says so *and* says what it would have taken:

```
verdict: no significant change
difference    -10.0% [-23.3, 0.0]
p-value       0.2500
underpowered: detecting a 10.0% difference needs about 234 cases; this run compared 30
```

That distinction matters. "No significant change" and "this dataset is too small to
tell" are different findings, and conflating them is how teams end up shipping on noise.

### Validating the judge

An LLM judge is itself a model that can be wrong, so it is measured before it is
trusted. Raw accuracy is the trap: if most attempts fail, a judge that answers "fail"
every time scores well and has learned nothing. Cohen's κ corrects for exactly that.

```bash
evalforge judge validate --gold datasets/gold/judgments.jsonl \
                         --verdicts verdicts.txt --threshold 0.6
```

```
verdict: not usable: kappa 0.000 (none or worse than chance) against 12 labels
accuracy                                        58.3%
cohen's kappa                                   0.000
```

That judge agrees with the humans 58% of the time and is worthless; the command exits
non-zero so it cannot reach a build. Validation runs offline from recorded verdicts —
re-running the judge is not needed to re-measure agreement — and the judge's model,
prompt hash and temperature are fingerprinted, so comparing κ across a changed prompt
reports *why* the two numbers are not comparable instead of quietly contrasting them.

### The dashboard

```bash
evalforge serve --db .evalforge/runs.db     # http://127.0.0.1:8000
```

Run list → run detail with metric tiles → click any case to see the agent's trajectory as
an expandable timeline, and a comparison view that shows the verdict, the confidence
interval and which cases moved. One static page, no bundler, no framework; it binds to
localhost because the API has no authentication and exposing it should be a decision
rather than an accident.

### Making a regression actionable

```bash
evalforge cluster <run-id>     # group failures by shape
evalforge select --budget 20   # the most informative cases per unit cost
evalforge gate <run-id> --min-success 0.70 --max-unsafe 0
```

Clustering is deterministic and feature-based rather than embedding-based, so a cluster
that grows between runs grew because behaviour changed — and because the dataset seeds
known faults, cluster membership is checkable rather than eyeballed:

```
cluster                                   cases  dominant fault
changed nothing                              14  integer_division
edited the right file, tests still failing    8  missing_empty_case
```

`gate` judges a run it did not produce and exits non-zero on a breach, so CI can enforce
a threshold without being able to change the numbers it is judging. The whole workflow
uses the deterministic agents and never contacts a provider — it cannot be blocked by a
rate limit or spend a token budget.

## Development

```bash
ruff check .      # lint
mypy src          # types (strict)
pytest -q         # tests — offline, no API key needed
```

## Licence

MIT
