#!/usr/bin/env python3
"""
AI Agent Assessment Script

Evaluates hospitality AI agent conversations using Claude API.
Scores each conversation on a 1-10 scale (5 capability levels) with detailed reasoning.

Usage:
    python scripts/assess.py                    # Assess all conversations
    python scripts/assess.py --agent my_agent       # Assess one agent
    python scripts/assess.py --dry-run           # Show what would be assessed without calling API
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic
import openai
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

load_dotenv(BASE_DIR / ".env")

SYSTEM_PROMPT = """\
You are an expert evaluator of AI-powered hospitality chatbots. You evaluate conversations \
on a 1-10 scale that measures the agent's ability to EXECUTE tasks for guests — not just \
answer questions. The scale is built on 5 capability levels (L1-L5), each spanning 2 points.

THE 10-POINT SCALE:

  1 = HARMFUL (L1): Agent fabricates information, hallucinates facilities or services \
that don't exist, or claims to take actions it did not take. Guest is actively misled. \
The guest is worse off than if there were no bot at all.

  2 = DEAD END (L1): Agent acknowledges the guest but provides no useful information \
or path forward. Generic filler responses ("we have options available"), canned "I can't \
help with that," or completely ignores the actual question. Nobody does any work because \
the bot provides nothing useful.

  3 = DEFLECTS (L2): Agent tells the guest to call reception, check the website, or \
email someone. Does not use any property-specific data — gives only generic contact info \
or links. Guest must do all the work themselves. The agent is essentially a phone directory.

  4 = INFORMED BUT PASSIVE (L2): Agent pulls real property data (room types, menu items, \
prices, hours, treatment options) and shares it with the guest, but only provides links \
or contact info to complete the task. The agent HAS data access but CANNOT act on it. \
Guest still completes the task elsewhere.

  5 = DISPATCHES WITHOUT DETAIL (L3): Agent forwards the guest's request to hotel staff \
but without gathering the necessary details first. Staff receive a vague or incomplete \
request and need to follow up with the guest or figure out specifics themselves. The guest \
doesn't need to act, but staff have extra work because the agent didn't prepare the request.

  6 = DISPATCHES WITH DETAIL (L3): Agent gathers all relevant details from the guest \
(dates, party size, preferences, allergies, room number, etc.) and THEN forwards a \
complete, well-prepared request to staff. Staff can act on it immediately without needing \
to gather more info. But the agent has NOT checked availability or confirmed the outcome — \
staff still need to execute and confirm.

  7 = FULFILS VIA STAFF (L4): Agent gathers all details, checks availability where \
possible, submits the request, and the task is completed — but hotel staff are still in \
the loop to confirm or act on the other end (e.g., restaurant confirms the reservation, \
maintenance acknowledges the ticket). Guest is fully done; staff just need to confirm. \
A payment link is acceptable where payment is required, provided the agent gathered all \
data and checked availability first.

  8 = FULFILS IN SYSTEM (L4): Agent checks live availability, makes the booking, or \
resolves the request directly in a system — no hotel staff needed in the loop. Fully \
automated end-to-end. Guest gets immediate confirmation with all details (dates, price, \
confirmation number). A payment link is acceptable where payment is required.

  9 = ANTICIPATES (L5): Agent fulfils the task AND proactively addresses what the guest \
will likely need next — suggests related services, mentions relevant policies (checkout \
time, cancellation policy), or offers to arrange follow-up steps the guest hasn't asked \
about yet. One step ahead.

  10 = ANTICIPATES + COORDINATES (L5): Agent handles multiple related needs \
simultaneously, cross-references guest context across the conversation, and coordinates \
across hotel departments or services in a single interaction. Personalised, multi-step, \
only possible with deep system integration.

KEY JUMPS BETWEEN LEVELS:
- 3 vs 4: Does the agent have real property data, or just canned "call us" responses?
- 5 vs 6: Did the agent gather all details before dispatching, or relay the raw request?
- 7 vs 8: Does hotel staff still need to confirm/act, or is it fully automated in a system?

You will receive:
1. A task definition (what the guest is trying to accomplish)
2. Task-specific scoring notes (things to watch for when evaluating this task)
3. The actual conversation between the guest and the AI agent
4. Any notes about deviations from the standard test script

CRITICAL SCORING RULES:
- Score based on what the AI agent ACTUALLY DID, not what the property offers or what \
the agent claims it can do.
- Match the agent's behavior to the scale level that best describes WHAT HAPPENED. \
If between levels, round down.
- The L3-L4 boundary (6 vs 7) is the most important: below 7, the agent DISPATCHED but \
did not confirm the outcome. At 7+, the task is DONE from the guest's perspective.
- "I've forwarded your request" without confirmation = L3 (5-6). "Your booking is \
confirmed for 8pm, party of 2" = L4 (7+). The difference is whether the loop is closed.
- Payment links are acceptable at L4 (7-8) ONLY IF the agent gathered all necessary data \
and checked availability first. A raw booking link without conversation = L2 (3-4).
- Fabrication / hallucination is always a 1, regardless of how helpful the rest of the \
conversation is.
- If a service doesn't exist at the property (e.g., no spa, no room service), evaluate \
how gracefully the agent communicates this and whether alternatives are offered.
- Consider deviation notes — if the tester didn't follow the script exactly, account for \
that but still evaluate the agent's actual performance.
- Scores 9-10 should be rare — they require genuine anticipation and multi-step \
coordination beyond the guest's explicit request.

Respond with ONLY a JSON object (no markdown, no extra text) in this exact format:
{
  "score": <1-10>,
  "reasoning": "<2-3 sentence explanation referencing the scale level matched>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "weaknesses": ["<weakness 1>", "<weakness 2>"]
}
"""


def load_tasks() -> dict:
    """Load task definitions keyed by task_id."""
    with open(DATA_DIR / "tasks.json") as f:
        tasks = json.load(f)
    return {t["id"]: t for t in tasks}


def load_conversations(agent_filter: str | None = None) -> list[dict]:
    """Load conversation files, optionally filtered by agent."""
    conversations_dir = DATA_DIR / "conversations"
    results = []
    for path in sorted(conversations_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        if agent_filter and data["agent_id"] != agent_filter:
            continue
        # Use filename stem as unique key to support multiple files per agent
        data["_file_key"] = path.stem
        results.append(data)
    return results


def build_evaluation_prompt(task: dict, conversation_entry: dict) -> str:
    """Build the user prompt for a single task evaluation."""
    messages_text = ""
    if conversation_entry.get("messages"):
        for msg in conversation_entry["messages"]:
            role = "Guest" if msg["role"] == "guest" else "AI Agent"
            messages_text += f"**{role}:** {msg['text']}\n\n"
    else:
        return ""  # Skip gaps

    notes = task.get("scoring_notes", [])
    notes_text = ""
    for note in notes:
        notes_text += f"  - {note}\n"

    deviation = conversation_entry.get("deviation_from_spec") or "None"

    return f"""\
## Task: {task['name']}

**Objective:** {task['objective']}

**Standard opening line:** {task['opening_line']}

### Scoring Notes
{notes_text}

### Deviation from test script
{deviation}

### Conversation
{messages_text}

Please evaluate this conversation and return your score as JSON.
"""


def _is_openai_model(model: str) -> bool:
    """Check if the model name indicates an OpenAI model."""
    return model.startswith(("gpt-", "o1", "o3", "o4"))


def _parse_json_response(text: str) -> dict:
    """Parse JSON from model response, handling markdown wrapping."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()
    return json.loads(text)


def evaluate_conversation(
    client,
    task: dict,
    conversation_entry: dict,
    model: str,
) -> dict | None:
    """Call LLM API to evaluate a single task conversation."""
    user_prompt = build_evaluation_prompt(task, conversation_entry)
    if not user_prompt:
        return None

    if _is_openai_model(model):
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=2048,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content
    else:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = response.content[0].text

    return _parse_json_response(text)


def generate_markdown_report(results: dict, timestamp: str) -> str:
    """Generate a human-readable markdown report from assessment results."""
    lines = [
        "# AI Agent Assessment Report",
        "",
        f"Generated: {timestamp}",
        "",
        "**Scale:** 1-10 across 5 levels. "
        "L1 (1-2) = harmful/dead end. L2 (3-4) = FAQ bot. "
        "L3 (5-6) = dispatcher. L4 (7-8) = fulfils. L5 (9-10) = anticipates.",
        "",
    ]

    # ── Build ranking data ──────────────────────────────────────────────
    ranking = []
    for agent_id, agent_data in results["agents"].items():
        scored_tasks = [
            t for t in agent_data["tasks"]
            if not t.get("gap") and not t.get("error") and t.get("score")
        ]
        if not scored_tasks:
            continue
        scores = [t["score"] for t in scored_tasks]
        avg = sum(scores) / len(scores)
        ranking.append({
            "id": agent_id,
            "name": agent_data["name"],
            "property": agent_data.get("property_name", ""),
            "avg": avg,
            "tasks_scored": len(scored_tasks),
            "tasks_total": len(agent_data["tasks"]),
        })
    ranking.sort(key=lambda r: r["avg"], reverse=True)

    # ── Ranking summary table ───────────────────────────────────────────
    lines.append("## Overall Ranking")
    lines.append("")
    lines.append(
        "| Rank | Agent | Property | Avg Score | Tasks |"
    )
    lines.append(
        "|------|-------|----------|:---------:|:-----:|"
    )
    for i, r in enumerate(ranking, 1):
        lines.append(
            f"| {i} | **{r['name']}** | {r['property']} "
            f"| **{r['avg']:.1f}** "
            f"| {r['tasks_scored']}/{r['tasks_total']} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Per-agent details ────────────────────────────────────────────────
    for agent_id, agent_data in results["agents"].items():
        name = agent_data["name"]
        prop = agent_data.get("property_name", "")
        lines.append(f"## {name}" + (f" ({prop})" if prop else ""))
        lines.append("")

        # Score table
        lines.append("| Task | Score |")
        lines.append("|------|:-----:|")

        all_scores = []

        for task_result in agent_data["tasks"]:
            if task_result.get("gap"):
                lines.append(f"| {task_result['task_name']} | — |")
                continue

            if task_result.get("error") or not task_result.get("score"):
                lines.append(f"| {task_result['task_name']} | ERR |")
                continue

            score = task_result["score"]
            lines.append(f"| {task_result['task_name']} | {score} |")
            all_scores.append(score)

        # Average row
        if all_scores:
            avg = sum(all_scores) / len(all_scores)
            lines.append(f"| **Average** | **{avg:.1f}** |")

        lines.append("")

        # Per-task details
        for task_result in agent_data["tasks"]:
            if task_result.get("gap"):
                lines.append(f"### {task_result['task_name']}")
                lines.append("**NOT TESTED** — Gap in assessment coverage.")
                lines.append("")
                continue

            if task_result.get("error") or not task_result.get("score"):
                lines.append(f"### {task_result['task_name']}")
                lines.append(
                    f"**ERROR** — {task_result.get('error', 'Unknown error')}"
                )
                lines.append("")
                continue

            lines.append(f"### {task_result['task_name']}")
            lines.append("")
            lines.append(
                f"**Score: {task_result['score']}/10** — "
                f"{task_result['reasoning']}"
            )
            lines.append("")

            if task_result.get("strengths"):
                lines.append("**Strengths:**")
                for s in task_result["strengths"]:
                    lines.append(f"- {s}")
                lines.append("")

            if task_result.get("weaknesses"):
                lines.append("**Weaknesses:**")
                for w in task_result["weaknesses"]:
                    lines.append(f"- {w}")
                lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Assess AI agent conversations")
    parser.add_argument("--agent", help="Assess only this agent ID")
    parser.add_argument("--dry-run", action="store_true", help="Show tasks without calling API")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-5-20250929",
        help="Model to use. Supports Anthropic (claude-*) and OpenAI (gpt-*, o1, o3, o4) models.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save output files (default: reports/)",
    )
    args = parser.parse_args()

    tasks = load_tasks()
    conversations = load_conversations(args.agent)

    if not conversations:
        print("No conversations found.", file=sys.stderr)
        if args.agent:
            print(f"Check that data/conversations/{args.agent}.json exists.", file=sys.stderr)
        sys.exit(1)

    # Dry run: just show what would be assessed
    if args.dry_run:
        for conv in conversations:
            print(f"\n{conv['agent_name']} ({conv.get('property_name', '')}):")
            for task_entry in conv["tasks"]:
                status = "GAP" if task_entry.get("gap") else "OK"
                print(f"  [{status}] {task_entry['task_name']}")
                if task_entry.get("deviation_from_spec"):
                    print(f"        Note: {task_entry['deviation_from_spec']}")
        return

    # Real run: create API client
    if _is_openai_model(args.model):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
            sys.exit(1)
        client = openai.OpenAI(api_key=api_key)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    results = {"timestamp": timestamp, "model": args.model, "agents": {}}

    for conv in conversations:
        file_key = conv.get("_file_key", conv["agent_id"])
        agent_name = conv["agent_name"]
        prop_name = conv.get("property_name", "")
        label = f"{agent_name} ({prop_name})" if prop_name else agent_name
        print(f"\nAssessing {label}...")

        results["agents"][file_key] = {
            "name": agent_name,
            "property_name": prop_name,
            "tasks": [],
        }

        for task_entry in conv["tasks"]:
            task_id = task_entry["task_id"]
            task_def = tasks.get(task_id)

            if not task_def:
                print(f"  WARNING: Unknown task_id '{task_id}', skipping.", file=sys.stderr)
                continue

            if task_entry.get("gap") or not task_entry.get("messages"):
                print(f"  [{task_entry['task_name']}] SKIPPED (gap)")
                results["agents"][file_key]["tasks"].append(
                    {
                        "task_id": task_id,
                        "task_name": task_entry["task_name"],
                        "gap": True,
                    }
                )
                continue

            print(f"  [{task_entry['task_name']}] Evaluating...", end=" ", flush=True)

            try:
                evaluation = evaluate_conversation(
                    client, task_def, task_entry, args.model
                )
                if evaluation:
                    task_result = {
                        "task_id": task_id,
                        "task_name": task_entry["task_name"],
                        "gap": False,
                        **evaluation,
                    }
                    results["agents"][file_key]["tasks"].append(task_result)
                    print(f"score={evaluation['score']}")
                else:
                    print("SKIPPED (no messages)")
            except json.JSONDecodeError as e:
                print(f"ERROR (invalid JSON from API): {e}", file=sys.stderr)
                results["agents"][file_key]["tasks"].append(
                    {
                        "task_id": task_id,
                        "task_name": task_entry["task_name"],
                        "error": str(e),
                    }
                )
            except (anthropic.APIError, openai.APIError) as e:
                print(f"API ERROR: {e}", file=sys.stderr)
                results["agents"][file_key]["tasks"].append(
                    {
                        "task_id": task_id,
                        "task_name": task_entry["task_name"],
                        "error": str(e),
                    }
                )

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"assessment_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON report: {json_path}")

    md_report = generate_markdown_report(results, timestamp)
    md_path = output_dir / f"assessment_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
