# ForecastIQ
Professional load-forecast operations dashboard with deterministic analytics, revision intelligence, and safe LLM-assisted explanations.

## What This Project Does
ForecastIQ helps grid/utility operators answer one question quickly: **what load risk is coming in the next 90 hours, and how stable is that forecast across updates?**

The project combines:
- deterministic forecasting (LightGBM pipeline)
- cache-first artifact generation per run
- revision/stability analytics across overlapping forecasts
- weather-load operational coupling analysis
- deterministic decision briefing
- safe, tool-grounded agentic chat over saved artifacts

## Key Design Principles
- Deterministic analytics first; LLM is only for explanation.
- No model retraining if cached run outputs already exist.
- Every major phase writes auditable artifacts to disk.
- LLM answers are constrained by evidence and numeric-token validation.

## High-Level Flow
1. User selects forecast date/hour in the Streamlit app.
2. Phase 1 either reuses cached forecast CSV or runs the pipeline.
3. Later phases read saved artifacts and compute additional analytics.
4. Dashboard renders KPIs, charts, briefings, and chat answers.

## Project Layout
```text
ForecastIQ/
├─ configs/
│  └─ config.yaml                # Central runtime paths/options
├─ prompts/
│  └─ briefing_writer.md         # LLM rewrite prompt (guardrailed)
├─ scripts/
│  ├─ run_dashboard.sh           # Start Streamlit app
│  ├─ test_phase2.py
│  ├─ test_phase25.py
│  ├─ test_phase3.py
│  ├─ test_phase3_llm_writer.py
│  ├─ test_phase3c.py
│  ├─ test_phase3c1_patch.py
│  ├─ test_phase4.py
│  └─ test_phase4_agentic_chat.py
├─ src/
│  ├─ gru_universal.py           # Core forecast pipeline + CLI
│  ├─ phase1_backend.py          # Forecast run orchestration + caching
│  ├─ phase2_backend.py          # Revisions/stability orchestration
│  ├─ phase25_backend.py         # Weather-load ops orchestration
│  ├─ phase3/                    # Deterministic briefing + policies + LLM writer mode
│  ├─ phase4/                    # Facts pack + simple summary + agentic chat tools/planner
│  ├─ llm/client.py              # Groq wrapper with retry/error handling
│  └─ ui/app.py                  # Streamlit dashboard
├─ .streamlit/
│  └─ config.toml                # UI theme
└─ README.md
```

## Phases and Outputs
All run outputs are organized by `RUN_ID=YYYYMMDDHH`:

```text
outputs/runs/<RUN_ID>/
```

### Phase 1: Forecast Package
- `forecast.csv`
- `forecast.json`
- `metrics.json`
- `weather_window.csv` (if available)
- `backtest_last_available.csv` (if overlap exists)

### Phase 2: Revision / Stability
- `phase2/forecast_matrix.csv`
- `phase2/per_hour_metrics.csv`
- `phase2/consensus_series.csv`
- `phase2/exceedance_proxy.csv`
- `phase2/peak_by_init.csv`
- `phase2/voi.csv`, `phase2/voi.json`
- `phase2/day_metrics.json`, `phase2/phase2_summary.json`

### Phase 2.5: Weather-Load Ops Coupling
- `phase25/weather_per_hour_metrics.csv`
- `phase25/weather_day_metrics.json`
- `phase25/weather_consensus_<var>.csv`
- `phase25/revision_pairs.csv`
- `phase25/attribution_fit.json`
- `phase25/correlation_table.csv`
- `phase25/joint_ops_risk.csv`
- `phase25/phase25_summary.json`

### Phase 3 / 3-C: Deterministic Briefing
- `phase3/phase3_input.json`
- `phase3/briefing.json`
- `phase3/action_items.json`
- `phase3/briefing.md`
- optional `phase3/briefing_llm.md` + meta

### Phase 4 / 4-B: Facts + Agentic Chat
- `phase4/facts_pack.json`
- `phase4/simple_summary.md`
- `phase4/tools/*.csv|*.json` (tool outputs and cache meta)
- `phase4/chat_logs/*.json` (chat turn logs)

## How LLM Is Used (and Not Used)
LLM is used for:
- rewriting deterministic briefing text (`briefing_llm.md`)
- plain-language summaries
- explaining tool outputs in chat

LLM is **not** used for:
- forecasting computations
- metric calculations
- artifact ground truth

### Safety Guardrails
- Numeric-token validation: reject output with numbers not present in evidence.
- Deterministic fallback if LLM fails validation or API call fails.
- Tool-based chat: LLM explains tool results, tools compute values.
- Evidence-only policy for responses with explicit source paths/fields.

## Requirements
- macOS/Linux
- Python 3.11+ recommended
- Local input data prepared in `data/` (kept local, typically not committed)

## Quickstart
1. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Configure LLM key (optional but recommended for LLM features)
```bash
cp .env.example .env  # if you add one locally, otherwise create .env
# set GROQ_API_KEY=...
```

4. Run dashboard
```bash
./scripts/run_dashboard.sh
```

## Common Commands
Run forecast package from backend:
```bash
python -c "from src.phase1_backend import run_for_date; print(run_for_date('2025-01-25','00')['run_id'])"
```

Run phase 3 briefing:
```bash
python -m src.phase3.run_phase3 --run_id 2025012500
```

Run phase 4 artifacts:
```bash
python -m src.phase4.run_phase4 --run_id 2025012500 --force
```

Run agentic chat test:
```bash
python scripts/test_phase4_agentic_chat.py
```

## Publish to GitHub
This folder currently may not be a git repo. To publish:

```bash
git init
git add .
git commit -m "Initial commit: ForecastIQ"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

- Keep raw data and generated run artifacts local unless you explicitly want versioned snapshots.


