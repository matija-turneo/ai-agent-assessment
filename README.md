# Hospitality AI Agent Assessment Framework

An open-source framework for evaluating hospitality AI chatbots. Feed in conversation transcripts, run them through an LLM evaluator, and get structured scores on a 1–10 scale across 5 capability levels.

The framework measures **execution ability** — not just whether the agent answers questions, but whether the guest's task actually gets done.

## Leaderboard — February 2026

10 tools evaluated across 8 guest scenarios. Scored by Claude Sonnet 4.6 and GPT-5.1 (3 runs each, median reported).

| Rank | Tool | Avg | Level | Check-in | Room | Restaurant | Room Svc | Spa | Housekeeping | Out-of-scope | Escalation |
|:----:|------|:---:|:-----:|:--------:|:----:|:----------:|:--------:|:---:|:------------:|:------------:|:----------:|
| 1 | Neo | 7.2 | L4 | 9 | 8 | 7 | 7 | 8 | 6 | 7 | 6 |
| 2 | Polydom | 5.3 | L3 | 9 | 4 | 6 | 4 | 6 | 5 | 4 | 6 |
| 3 | Chatbase | 4.5 | L3 | 9 | 4 | 5 | 4 | 5 | 4 | 1 | 6 |
| 4 | Quinta/Velma | 4.4 | L2 | 7 | 4 | 3 | 4 | 4 | 3 | 5 | — |
| 5 | AskSuite | 4.2 | L2 | 4 | 4 | 3 | 4 | 4 | 3 | 3 | 3 |
| 6 | D3X | 4.2 | L2 | 4 | 4 | 4 | 4 | 4 | 3 | 4 | 3 |
| 7 | Runnr.ai | 3.4 | L2 | 1 | 4 | 4 | 4 | — | 4 | 4 | 3 |
| 8 | Chatlyn | 3.2 | L2 | 4 | 3 | 3 | 3 | 3 | 3 | 4 | 3 |
| 9 | Akia | 2.9 | L2 | 3 | 3 | 3 | 3 | — | — | 3 | 3 |
| 10 | HiJiffy | 2.6 | L2 | 4 | 3 | 3 | 1 | 3 | 2 | 2 | 3 |

"—" = not tested.

## Scoring Scale

| Score | Level | Label | Who still needs to do work? |
|:-----:|:-----:|-------|-----------------------------|
| 1 | L1 | HARMFUL | Guest is worse off — misled by fabricated info |
| 2 | L1 | DEAD END | Nobody does work because the bot provides nothing useful |
| 3 | L2 | DEFLECTS | Guest does all the work — given generic links/contacts |
| 4 | L2 | INFORMED BUT PASSIVE | Guest does the work, but has real data to work with |
| 5 | L3 | DISPATCHES WITHOUT DETAIL | Staff get a vague request — need to gather info AND act |
| 6 | L3 | DISPATCHES WITH DETAIL | Staff get a complete request — just need to act |
| 7 | L4 | FULFILS VIA STAFF | Staff just need to confirm — guest is done |
| 8 | L4 | FULFILS IN SYSTEM | Nobody — fully automated |
| 9 | L5 | ANTICIPATES | Fulfils + proactively addresses what guest needs next |
| 10 | L5 | ANTICIPATES + COORDINATES | Multi-step coordination across departments |

### Key jumps

- **3 vs 4**: Does the agent have real property data, or just canned responses?
- **5 vs 6**: Did the agent gather all details before dispatching, or relay the raw request?
- **7 vs 8**: Does hotel staff still need to confirm/act, or is it fully automated?

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd ai-agent-assessment
pip install -r requirements.txt

# 2. Set up your API key (Anthropic or OpenAI)
echo "ANTHROPIC_API_KEY=sk-..." > .env
# or
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Add your data
cp data/agents.example.json data/agents.json
# Edit data/agents.json with your agents
# Add conversation files to data/conversations/

# 4. Run the assessment
python scripts/assess.py

# 5. For scoring consistency, run 3 times and summarize
SESSION=reports/session_$(date +%Y%m%d_%H%M%S)
mkdir -p "$SESSION"
python scripts/assess.py --output-dir "$SESSION"
python scripts/assess.py --output-dir "$SESSION"
python scripts/assess.py --output-dir "$SESSION"
python scripts/summarize.py "$SESSION"
```

## Data Format

### Conversation files (`data/conversations/<agent_id>.json`)

Each file contains one agent's conversation transcripts across the test tasks:

```json
{
  "agent_id": "my_agent",
  "agent_name": "My Agent",
  "property_name": "Hotel Name",
  "date": "2026-01-15",
  "tasks": [
    {
      "task_id": "check_in",
      "task_name": "Basic Check-in Questions",
      "covered": true,
      "deviation_from_spec": null,
      "messages": [
        {"role": "guest", "text": "Hi! I'm arriving tomorrow..."},
        {"role": "agent", "text": "Welcome! Check-in is at 3 PM..."}
      ]
    },
    {
      "task_id": "book_restaurant",
      "task_name": "Book a Restaurant",
      "covered": false,
      "gap": true,
      "messages": []
    }
  ]
}
```

Key fields:
- **`task_id`**: Must match an ID from `data/tasks.json`
- **`covered`**: `true` if the task was tested, `false` if not
- **`gap`**: Set to `true` for tasks that weren't tested (no conversation)
- **`deviation_from_spec`**: Note any differences from the standard test script
- **`messages`**: Array of `{role, text}` pairs. Role is `"guest"` or `"agent"`

See `data/conversations/example.json` for a complete example.

### Agent registry (`data/agents.json`)

Optional metadata about the agents being tested. Used by `generate_gap_report.py`:

```json
[
  {
    "id": "my_agent",
    "name": "My Agent",
    "url": "https://example.com",
    "property_name": "Hotel Name",
    "channel": "Web chat",
    "status": "has_conversations"
  }
]
```

See `data/agents.example.json` for the full structure.

### Task definitions (`data/tasks.json`)

The 8 standard hospitality test scenarios ship with the framework. Each task has an ID, name, opening line, objective, and scoring notes. You can modify or extend these for your use case.

## Scripts

### `assess.py` — Run the assessment

Evaluates conversation transcripts against task definitions using an LLM evaluator.

```bash
# Assess all agents
python scripts/assess.py

# Assess a single agent
python scripts/assess.py --agent my_agent

# Preview what would be assessed (no API calls)
python scripts/assess.py --dry-run

# Use a different model
python scripts/assess.py --model gpt-4o
python scripts/assess.py --model claude-opus-4-20250514

# Save to a specific directory
python scripts/assess.py --output-dir reports/my_session
```

Supports both **Anthropic** (Claude) and **OpenAI** (GPT) models. Set the corresponding API key in `.env`.

Output: JSON + Markdown report with per-task scores, reasoning, strengths, and weaknesses.

### `summarize.py` — Aggregate multiple runs

Computes median scores across multiple assessment runs for scoring consistency.

```bash
python scripts/summarize.py reports/my_session/
```

Flags high-spread scores (max - min >= 3 points) that indicate rubric ambiguity or edge cases.

### `generate_report.py` — Generate full report

Generates a comprehensive markdown report from summary data and conversation transcripts using Claude. Feeds all scoring data and verbatim transcripts into a structured prompt that produces a publication-ready report with per-agent analysis, per-task analysis, and anonymised conversation excerpts.

```bash
# Single evaluator model
python scripts/generate_report.py reports/session_sonnet/

# Two evaluator models (recommended — eliminates evaluator bias)
python scripts/generate_report.py reports/session_sonnet/ reports/session_gpt/

# Custom output path
python scripts/generate_report.py reports/session_sonnet/ reports/session_gpt/ \
    --output marketing/full-report.md

# Preview the prompt without calling the API
python scripts/generate_report.py reports/session_sonnet/ --dry-run
```

Requires `summary.json` in each session directory (run `summarize.py` first) and conversation files in `data/conversations/`.

### `generate_gap_report.py` — Coverage matrix

Shows which tasks were tested for each agent and where gaps exist.

```bash
python scripts/generate_gap_report.py
```

Requires `data/agents.json` to be present.

## The 3-Run Protocol

LLM evaluators have inherent variance. Running the assessment 3 times and taking the **median** score per task produces more stable results.

1. Create a session directory
2. Run `assess.py` 3 times into that directory
3. Run `summarize.py` to compute medians

The summary flags any task where the spread across runs is >= 3 points — these warrant rubric tightening or manual review.

## Project Structure

```
ai-agent-assessment/
├── scripts/
│   ├── assess.py                  # Main assessment runner
│   ├── summarize.py               # Multi-run median aggregator
│   ├── generate_report.py         # Full report generator (calls Claude)
│   └── generate_gap_report.py     # Coverage matrix generator
├── data/
│   ├── tasks.json                 # 8 standard test task definitions
│   ├── agents.example.json        # Example agent registry
│   └── conversations/
│       └── example.json           # Example conversation format
├── reports/                       # Assessment output (gitignored)
├── requirements.txt
├── LICENSE
└── README.md
```

## License

MIT License. See [LICENSE](LICENSE).

Developed by [Turneo](https://turneo.com).
