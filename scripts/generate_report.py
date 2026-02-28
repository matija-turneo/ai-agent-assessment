#!/usr/bin/env python3
"""
Generate the full assessment report from summary data and conversation transcripts.

Takes summary JSONs from one or more sessions (each produced by summarize.py),
loads conversation transcripts, and calls Claude to produce a structured
markdown report with per-agent analysis, per-task analysis, and
verbatim conversation excerpts.

Usage:
    # Single model session
    python scripts/generate_report.py reports/session_sonnet/

    # Two model sessions (recommended — eliminates evaluator bias)
    python scripts/generate_report.py reports/session_sonnet/ reports/session_gpt/

    # Custom output path
    python scripts/generate_report.py reports/session_sonnet/ reports/session_gpt/ \
        --output marketing/full-report.md

    # Use a different writer model
    python scripts/generate_report.py reports/session_sonnet/ --model claude-opus-4-20250514
"""

import argparse
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

load_dotenv(BASE_DIR / ".env")

SYSTEM_PROMPT = """\
You are an expert analyst writing a comprehensive assessment report on hospitality AI \
concierge tools. Your writing style is modelled on The Economist: authoritative, dry, \
precise. You let the data speak. You use short, declarative sentences. You do not use \
marketing language, superlatives, or filler. When a tool fails, you describe the failure \
with clinical precision. When a tool succeeds, you note it without fanfare.

You will receive:
1. Summary scoring data from one or two evaluator models (median scores per task per agent)
2. The full conversation transcripts for every agent
3. The task definitions and scoring rubric

Your job is to produce a complete markdown report with the exact structure below. \
Every section is mandatory. Do not skip or reorder sections.

---

## REPORT STRUCTURE

### 1. Title and Byline

```
# AI Concierge Assessment [YEAR]
## How [N] Hospitality Chatbots Perform Across Eight Real Guest Scenarios

*[Publisher Name] Research — [Month Year]*
```

### 2. Executive Summary

Start with a one-paragraph overview of the methodology (how many tools, how many \
scenarios, which evaluator models, how many passes).

Then an "Overall Ranking" table:

| Rank | Tool | [Model 1 name] | [Model 2 name] | Average | Level |
|------|------|:-:|:-:|:-:|:-:|

- Sort by Average score descending
- Average = mean of all model averages for that agent
- Level = the L1-L5 level corresponding to the average (L1: 1-2, L2: 3-4, L3: 5-6, L4: 7-8, L5: 9-10)
- If only one model session is provided, omit the second model column and the Average column

Then "Key Findings" — 5-7 bullet points summarising the most important patterns. Be specific \
with numbers. Reference the levels.

### 3. Disclaimer

This exact text (adjust the model name if different):

> This report — including all scoring, analysis, and written commentary — was generated \
entirely by AI (Claude Opus 4.6) without human editorial intervention. The conversations \
were conducted by a human tester, but the evaluation of those conversations, the assignment \
of scores, and the writing of this report were performed by AI models following the scoring \
rubric below. No scores were adjusted, no commentary was rewritten, and no rankings were \
altered after generation. This is done for fairness: the analysis reflects the rubric as \
applied by the model, not the opinions of the report's publisher.

Then three fairness bullet points:
- Different hotel properties — account for this in scoring
- Gaps in coverage — marked and excluded from averages
- Scores reflect bot behaviour, not hotel quality

### 4. The Framework: Five Levels of AI Concierge Capability

Opening line: "The framework measures one thing: **how far did the AI agent get toward \
completing the guest's task?**"

Table with columns: Level | Name | Score | What the Agent Did

| Level | Name | Score | What the Agent Did |
|-------|------|:-----:|-------------------|
| L1 | Dead End | 1-2 | Fabricated information, hallucinated capabilities, or left the guest with no useful path forward. |
| L2 | FAQ Bot | 3-4 | Retrieved information but left the guest to complete the task themselves — by phone, email, or website. |
| L3 | Dispatcher | 5-6 | Collected requirements and forwarded to staff. Never confirmed whether the request was fulfilled. |
| L4 | Concierge | 7-8 | Completed the task end-to-end. Checked availability, executed the request, confirmed the outcome. |
| L5 | Personal Butler | 9-10 | Completed the task and proactively addressed what the guest would need next — without being asked. |

Then "The Critical Jumps" subsection explaining L3 vs L4 and L2 vs L3.

### 5. Methodology

"The Eight Test Scenarios" table:

| # | Task | What We Asked | What "Good" Looks Like |

Populate from the task definitions provided.

Then "Protocol" bullets explaining the testing methodology (human tester, standardised \
script, which models, how many runs, median scoring, high-spread flagging).

### 6. Per-Agent Analysis

One section per agent, **ranked by average score** (highest first). Each section:

#### Heading: `### [Rank]. [Name] — [Level] (avg [score])`

Score table showing scores from each evaluator model:

| Task | [Model 1] | [Model 2] |
|------|:-:|:-:|

Use "—" for gaps. Bold the Average row.

Then:
- **Level: [Level] — [Name].** One-sentence summary.
- **Strengths:** bullet list (3-5 items). Reference specific tasks and scores.
- **Weaknesses:** bullet list (2-4 items). Reference specific tasks and scores.
- **Verbatim — [Task Name] (Score: [N]):** — One or two key conversation excerpts \
showing the tool's best AND worst moments. Use blockquote format:

> **Guest:** "..."
> **[Tool name]:** "..."

Choose excerpts that most vividly illustrate the score. For every agent, \
include 1-2 excerpts showing both strengths and weaknesses. No agent gets special treatment.

Add a brief editorial comment after each excerpt (one sentence).

### 7. Per-Task Analysis

One subsection per task. Analyse the pattern across ALL agents:

#### Format:
```
### [Task Name] — [Brief Theme]
```

For each task, write 3-5 paragraphs of analysis:

- **Opening paragraph:** What this task tests, why it matters for guest experience, and \
what overall pattern emerged across the agents tested. Be specific — reference how many \
tools scored at each level.
- **Best:** [Tool] ([score]) — 2-3 sentences explaining what the tool did right and why \
it earned this score. Reference the specific actions taken.
- **Runner-up:** [Tool] ([score]) — 2-3 sentences. If the runner-up used a notably \
different approach, describe it.
- **The rest / Worst:** Describe the common failure pattern among lower-scoring tools. \
Name specific tools and scores. If a tool scored 1-2, explain the failure in detail.
- **Pattern:** 2-3 sentences on the cross-agent insight — what systemic issue does \
this task reveal about the industry?
- **Include exactly 2 verbatim conversation excerpts** per task. Pick the two most \
illustrative moments — typically the best and worst performances, or two contrasting \
approaches. Use blockquote format. Add a one-sentence editorial comment after each excerpt.

### 8. Conclusion

3-4 paragraphs synthesising the key takeaways. Must address:
- Where the market is (most tools stuck at which level)
- Why the gap exists (system integration, not AI quality)
- What hotel operators should test
- End with a memorable closing line

### 9. Methodology Note

Bullet list:
- Assessment date
- Evaluator models (list all)
- Runs per model
- Total evaluations calculated (agents × tasks × models × runs)
- How conversations were conducted
- Link placeholder for the open-source framework

Then a disclosure line about the publisher.

---

## ANONYMISATION RULES

You MUST anonymise all identifying information in conversation excerpts:

- Guest names → `[guest name]`
- Hotel/property names → `[the hotel]` or `[the property]`
- Restaurant names → `[Restaurant A]`, `[Restaurant B]`, etc. (consistent within an agent)
- Spa names → `[the spa]`
- Specific room type names that are unique to a property → `[family suite]`, `[deluxe room]`, etc. \
  (keep generic descriptions like "double room" or "ocean view" as-is)
- Phone numbers → `[phone number]`
- Email addresses → `[email address]`
- Physical addresses → `[address]`
- Supermarket/shop names → `[Supermarket A]`, `[Supermarket B]`
- Pharmacy names → `[Pharmacy]`

Do NOT anonymise: agent/tool names, prices, times, dates, generic descriptions.

## WRITING RULES

- Never use exclamation marks
- Never use "cutting-edge", "innovative", "state-of-the-art", "revolutionary", or similar marketing words
- Use "the agent", "the tool", or the tool's name — never "the AI"
- When quoting conversations, use blockquote format with **bold** role labels
- Keep paragraphs short — 2-3 sentences max
- Every claim must be backed by a specific score or conversation excerpt
- Be harsh but fair. If a tool scores 3, call it what it is. If the top-ranked tool scores 6 on a task, call that out too — no tool gets softer language because of its overall rank.
"""


def load_summary(session_dir: Path) -> dict:
    """Load summary.json from a session directory."""
    summary_path = session_dir / "summary.json"
    if not summary_path.exists():
        print(f"No summary.json found in {session_dir}", file=sys.stderr)
        print("Run scripts/summarize.py first.", file=sys.stderr)
        sys.exit(1)
    with open(summary_path) as f:
        return json.load(f)


def load_conversations() -> list[dict]:
    """Load all conversation files."""
    conversations_dir = DATA_DIR / "conversations"
    results = []
    for path in sorted(conversations_dir.glob("*.json")):
        if path.stem == "example":
            continue
        with open(path) as f:
            results.append(json.load(f))
    return results


def load_tasks() -> list[dict]:
    """Load task definitions."""
    with open(DATA_DIR / "tasks.json") as f:
        return json.load(f)


def build_data_prompt(
    summaries: list[dict],
    conversations: list[dict],
    tasks: list[dict],
    publisher_name: str,
) -> str:
    """Build the user prompt containing all data for report generation."""
    sections = []

    # ── Summary data ──────────────────────────────────────────────────
    sections.append("# SCORING DATA\n")
    for i, summary in enumerate(summaries):
        model_name = summary.get("model", f"Model {i+1}")
        num_runs = summary.get("num_runs", "unknown")
        sections.append(f"## Evaluator Model: {model_name} ({num_runs} runs, median reported)\n")

        # Sort agents by avg score descending
        ranked = sorted(
            summary["agents"].items(),
            key=lambda x: x[1].get("avg_median", 0),
            reverse=True,
        )

        for agent_key, agent_data in ranked:
            name = agent_data["name"]
            avg = agent_data.get("avg_median", 0)
            sections.append(f"### {name} (avg: {avg})")
            for task in agent_data["tasks"]:
                if task.get("gap"):
                    sections.append(f"  - {task['task_name']}: GAP (not tested)")
                elif "median_score" in task:
                    spread_note = ""
                    if task.get("spread", 0) >= 3:
                        spread_note = f" [HIGH SPREAD: {task['spread']}]"
                    sections.append(
                        f"  - {task['task_name']}: {task['median_score']:.0f}{spread_note}"
                    )
                    # Include per-run reasoning if available
                    for j, run in enumerate(task.get("per_run", []), 1):
                        sections.append(
                            f"    Run {j} (score {run['score']}): {run.get('reasoning', '')}"
                        )
            sections.append("")

    # ── Task definitions ──────────────────────────────────────────────
    sections.append("\n# TASK DEFINITIONS\n")
    for task in tasks:
        sections.append(f"## {task['name']} (id: {task['id']})")
        sections.append(f"Opening line: {task['opening_line']}")
        sections.append(f"Objective: {task['objective']}")
        sections.append("Scoring notes:")
        for note in task.get("scoring_notes", []):
            sections.append(f"  - {note}")
        sections.append("")

    # ── Conversation transcripts ──────────────────────────────────────
    sections.append("\n# CONVERSATION TRANSCRIPTS\n")
    sections.append(
        "Below are the full conversation transcripts. Use these for verbatim "
        "excerpts in the report. Remember to anonymise per the rules.\n"
    )
    for conv in conversations:
        agent_name = conv["agent_name"]
        prop_name = conv.get("property_name", "")
        header = f"## {agent_name}"
        if prop_name:
            header += f" ({prop_name})"
        sections.append(header)

        for task_entry in conv["tasks"]:
            if task_entry.get("gap") or not task_entry.get("messages"):
                sections.append(f"### {task_entry['task_name']}: NOT TESTED\n")
                continue

            sections.append(f"### {task_entry['task_name']}")
            if task_entry.get("deviation_from_spec"):
                sections.append(f"Note: {task_entry['deviation_from_spec']}")

            for msg in task_entry["messages"]:
                role = "Guest" if msg["role"] == "guest" else agent_name
                sections.append(f"**{role}:** {msg['text']}")
            sections.append("")

    # ── Generation instructions ───────────────────────────────────────
    sections.append("\n# GENERATION INSTRUCTIONS\n")
    sections.append(
        f"Publisher name: {publisher_name}\n"
        "Generate the complete report following the structure in the system prompt. "
        "Include every section. Use all the data above — the scoring summaries for "
        "the numbers, and the conversation transcripts for verbatim excerpts. "
        "Anonymise all identifying information per the anonymisation rules. "
        "Output only the markdown report, nothing else."
    )

    return "\n".join(sections)


def generate_report(
    client: anthropic.Anthropic,
    data_prompt: str,
    model: str,
) -> str:
    """Call Claude to generate the full report using streaming."""
    chunks = []
    with client.messages.stream(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": data_prompt}],
    ) as stream:
        for text in stream.text_stream:
            chunks.append(text)
            print(text, end="", flush=True)
    print()  # newline after streaming
    return "".join(chunks)


def main():
    parser = argparse.ArgumentParser(
        description="Generate full assessment report from summary data and transcripts"
    )
    parser.add_argument(
        "session_dirs",
        nargs="+",
        help="One or more session directories containing summary.json",
    )
    parser.add_argument(
        "--output",
        default="marketing/full-report.md",
        help="Output file path (default: marketing/full-report.md)",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-20250514",
        help="Model to use for report generation (default: claude-opus-4-20250514)",
    )
    parser.add_argument(
        "--publisher",
        default="Turneo",
        help="Publisher name for the report byline (default: Turneo)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prompt instead of calling the API",
    )
    args = parser.parse_args()

    # Load all data
    summaries = []
    for session_dir in args.session_dirs:
        session_path = Path(session_dir)
        if not session_path.is_dir():
            print(f"Not a directory: {session_dir}", file=sys.stderr)
            sys.exit(1)
        summaries.append(load_summary(session_path))

    conversations = load_conversations()
    if not conversations:
        print("No conversation files found in data/conversations/", file=sys.stderr)
        sys.exit(1)

    tasks = load_tasks()

    # Build the data prompt
    data_prompt = build_data_prompt(summaries, conversations, tasks, args.publisher)

    if args.dry_run:
        print("=== SYSTEM PROMPT ===")
        print(SYSTEM_PROMPT)
        print("\n=== USER PROMPT ===")
        print(data_prompt)
        print(f"\n=== STATS ===")
        print(f"System prompt: ~{len(SYSTEM_PROMPT)} chars")
        print(f"User prompt: ~{len(data_prompt)} chars")
        print(f"Summaries: {len(summaries)}")
        print(f"Conversations: {len(conversations)}")
        print(f"Tasks: {len(tasks)}")
        return

    # Create API client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    print(f"Generating report with {args.model}...")
    print(f"  Summaries: {len(summaries)} session(s)")
    print(f"  Conversations: {len(conversations)} agent(s)")
    print(f"  Tasks: {len(tasks)}")

    report = generate_report(client, data_prompt, args.model)

    # Save output
    output_path = BASE_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
