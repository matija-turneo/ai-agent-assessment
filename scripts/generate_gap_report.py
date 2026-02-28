#!/usr/bin/env python3
"""
Generate a gap analysis report from conversation data.

Reads all conversation files and task definitions, then produces a markdown
report showing which tasks were tested per agent and where gaps exist.

Usage:
    python scripts/generate_gap_report.py
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def main():
    # Load tasks
    with open(DATA_DIR / "tasks.json") as f:
        tasks = json.load(f)
    task_ids = [t["id"] for t in tasks]
    task_names = {t["id"]: t["name"] for t in tasks}

    # Load agents
    with open(DATA_DIR / "agents.json") as f:
        agents = json.load(f)

    # Load conversations
    conversations_dir = DATA_DIR / "conversations"
    conv_data = {}
    for path in sorted(conversations_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        conv_data[data["agent_id"]] = data

    # Build coverage matrix
    lines = [
        "# Gap Analysis Report",
        "",
        f"Generated from {len(conv_data)} conversation files across {len(agents)} agents.",
        "",
        "## Coverage Matrix",
        "",
    ]

    # Table header
    header = "| Task |"
    separator = "|------|"
    for agent in agents:
        header += f" {agent['name']} |"
        separator += "------|"
    lines.append(header)
    lines.append(separator)

    # Table rows
    for task_id in task_ids:
        row = f"| {task_names[task_id]} |"
        for agent in agents:
            cid = agent["id"]
            if cid not in conv_data:
                row += " — |"
                continue

            task_entries = [
                t for t in conv_data[cid]["tasks"] if t["task_id"] == task_id
            ]
            if not task_entries:
                row += " **MISSING** |"
            elif task_entries[0].get("gap"):
                row += " **GAP** |"
            elif task_entries[0].get("covered") == "partial":
                row += " Partial |"
            elif task_entries[0].get("covered"):
                row += " Yes |"
            else:
                row += " ? |"
        lines.append(row)

    lines.append("")
    lines.append(
        "**Legend:** Yes = fully tested | Partial = tested but deviated from spec "
        "| GAP = not tested | MISSING = task not in file | — = no conversation yet"
    )
    lines.append("")

    # Pending agents
    pending = [a for a in agents if a["status"] == "pending"]
    if pending:
        lines.append(f"## Agents Without Conversations ({len(pending)}/{len(agents)})")
        lines.append("")
        for a in pending:
            lines.append(f"- **{a['name']}**")
        lines.append("")

    # Detailed notes per agent with conversations
    lines.append("## Detailed Notes")
    lines.append("")

    for agent in agents:
        cid = agent["id"]
        if cid not in conv_data:
            continue

        conv = conv_data[cid]
        lines.append(f"### {agent['name']} ({conv.get('property_name', '')})")
        lines.append("")
        lines.append("| Task | Status | Notes |")
        lines.append("|------|--------|-------|")

        for task_entry in conv["tasks"]:
            if task_entry.get("gap"):
                status = "**GAP**"
            elif task_entry.get("covered") == "partial":
                status = "Partial"
            elif task_entry.get("covered"):
                status = "OK"
            else:
                status = "?"

            notes = task_entry.get("deviation_from_spec") or "—"
            lines.append(f"| {task_entry['task_name']} | {status} | {notes} |")

        lines.append("")

    report = "\n".join(lines)

    # Write report
    output_path = DATA_DIR / "gap_analysis.md"
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Gap analysis written to: {output_path}")


if __name__ == "__main__":
    main()
