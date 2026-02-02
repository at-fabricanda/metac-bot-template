# CLAUDE.md — Metaculus Forecasting Bot

This is AT's forecasting bot for the Metaculus AI Forecasting Tournament. The goal is to maximize Baseline score (accuracy vs. community prediction).

## Quick Start

```bash
# Install dependencies
poetry install

# Test mode (doesn't submit)
poetry run python main.py --mode test_questions

# Run on tournament (submits predictions)
poetry run python main.py --mode tournament
```

## Architecture

- **main.py**: Primary bot using `forecasting-tools` package
- **main_with_no_framework.py**: Standalone version with minimal dependencies
- **.github/workflows/**: Automation (runs every 30min via GitHub Actions)

## Current Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| Extremizing factor | 1.3 | LLMs hedge toward 0.5; research shows 1.26-1.5 optimal |
| Model | OpenRouter GPT-4o | Cost-effective, good reasoning |
| Research | AskNews | Real-time news integration |

## Forecasting Methodology

The bot implements superforecasting principles:

1. **Base rates first** (outside view) — What's the historical frequency?
2. **Inside view adjustment** — What makes this case different?
3. **Status quo bias** — World changes slowly; default to "things stay the same"
4. **Scenario analysis** — Concrete paths to Yes and No
5. **Pre-mortem** — Reasons the forecast could be wrong
6. **Precise probabilities** — Distinguish 60% vs 65% vs 70%

### Extremizing

LLMs systematically underconfident. Raw predictions are extremized:
```python
logit = log(p / (1-p))
extremized = 1 / (1 + exp(-factor * logit))
```

Factor 1.3 pushes 0.6 → 0.66, 0.7 → 0.77, etc.

## Winner Patterns (from top tournament bots)

### ✅ Implemented
- Extremizing (factor 1.3)
- Superforecasting prompts with base rates, inside view, scenarios
- Research integration (AskNews)

### 🔲 TODO: Multi-Model Ensemble
Top bots use 3-5 models, discard outliers, average. Example:
- 3x GPT-4o + 2x Claude Sonnet
- Discard highest and lowest
- Average remaining predictions

### 🔲 TODO: Outlier Filtering  
Before aggregating: remove predictions >2σ from median.

## Learnings Log

*Record what works and what doesn't. Update after analyzing resolved questions.*

### 2025-02-02 — Initial Setup
- Forked from Metaculus template
- Added extremizing (1.3 factor)
- Enhanced prompts with superforecasting methodology

<!-- 
Add entries like:
### YYYY-MM-DD — [Topic]
- What we tried:
- Result:
- Lesson:
-->

## Code Conventions

- Use `clean_indents()` for multi-line prompts
- All probabilities clamped to [0.01, 0.99]
- Log predictions with `logger.info()`
- Async throughout — respect rate limits

## Environment Variables

Required in `.env` or GitHub Secrets:
- `METACULUS_TOKEN` — Bot account token
- `OPENROUTER_API_KEY` — LLM access
- `ASKNEWS_CLIENT_ID` / `ASKNEWS_CLIENT_SECRET` — Research (optional)

## Testing Changes

Before pushing:
```bash
# Dry run on test questions
poetry run python main.py --mode test_questions

# Check specific question
poetry run python check_questions.py
```

## Links

- [Tournament Dashboard](https://www.metaculus.com/aib/)
- [forecasting-tools docs](https://github.com/Metaculus/forecasting-tools)
- [Bot Resources](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/)
- [Discord](https://discord.com/invite/NJgCC2nDfh) — #build-a-forecasting-bot

---

*This file is read by Claude Code at session start. Update the Learnings Log as you discover what improves scores.*
