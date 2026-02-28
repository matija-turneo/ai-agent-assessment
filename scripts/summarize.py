#!/usr/bin/env python3
"""
Summarize multiple assessment runs into a single report with median scores.

Reads all assessment_*.json files from a session directory, computes the
median score per task per agent, and generates a final summary report.

Usage:
    python scripts/summarize.py reports/session_20260224_120000/
"""

import json
import statistics
import sys
from pathlib import Path


def load_runs(session_dir: Path) -> list[dict]:
    """Load all assessment JSON files from a session directory."""
    files = sorted(session_dir.glob("assessment_*.json"))
    if not files:
        print(f"No assessment_*.json files found in {session_dir}", file=sys.stderr)
        sys.exit(1)

    runs = []
    for f in files:
        with open(f) as fh:
            runs.append(json.load(fh))
    return runs


def summarize(runs: list[dict]) -> dict:
    """Compute median scores across runs for each agent + task."""
    # Collect all scores: {agent_key: {task_id: [scores]}}
    score_map: dict[str, dict[str, list[int]]] = {}
    # Keep metadata from first run
    agent_meta: dict[str, dict] = {}

    for run in runs:
        for agent_key, agent_data in run["agents"].items():
            if agent_key not in score_map:
                score_map[agent_key] = {}
                agent_meta[agent_key] = {
                    "name": agent_data["name"],
                    "property_name": agent_data.get("property_name", ""),
                }

            for task in agent_data["tasks"]:
                task_id = task["task_id"]
                if task.get("gap") or task.get("error") or not task.get("score"):
                    continue

                if task_id not in score_map[agent_key]:
                    score_map[agent_key][task_id] = []
                score_map[agent_key][task_id].append(task["score"])

    # Build summary with median scores
    # Also collect per-run reasoning for reference
    reasoning_map: dict[str, dict[str, list[dict]]] = {}
    for run in runs:
        for agent_key, agent_data in run["agents"].items():
            if agent_key not in reasoning_map:
                reasoning_map[agent_key] = {}
            for task in agent_data["tasks"]:
                task_id = task["task_id"]
                if task.get("gap") or task.get("error") or not task.get("score"):
                    continue
                if task_id not in reasoning_map[agent_key]:
                    reasoning_map[agent_key][task_id] = []
                reasoning_map[agent_key][task_id].append({
                    "score": task["score"],
                    "reasoning": task.get("reasoning", ""),
                })

    summary = {
        "num_runs": len(runs),
        "model": runs[0].get("model", "unknown"),
        "agents": {},
    }

    for agent_key in score_map:
        meta = agent_meta[agent_key]
        tasks_summary = []

        for task_id, scores in score_map[agent_key].items():
            # Find task name from first run
            task_name = task_id
            for run in runs:
                for t in run["agents"].get(agent_key, {}).get("tasks", []):
                    if t["task_id"] == task_id and t.get("task_name"):
                        task_name = t["task_name"]
                        break
                break

            median = statistics.median(scores)
            per_run = reasoning_map.get(agent_key, {}).get(task_id, [])

            tasks_summary.append({
                "task_id": task_id,
                "task_name": task_name,
                "median_score": median,
                "scores": scores,
                "spread": max(scores) - min(scores),
                "per_run": per_run,
            })

        # Also include gaps from first run
        for run in runs:
            for t in run["agents"].get(agent_key, {}).get("tasks", []):
                if t.get("gap"):
                    # Only add if not already in tasks_summary
                    if not any(ts["task_id"] == t["task_id"] for ts in tasks_summary):
                        tasks_summary.append({
                            "task_id": t["task_id"],
                            "task_name": t.get("task_name", t["task_id"]),
                            "gap": True,
                        })
            break

        scored = [t for t in tasks_summary if not t.get("gap") and "median_score" in t]
        avg = (
            sum(t["median_score"] for t in scored) / len(scored) if scored else 0
        )

        summary["agents"][agent_key] = {
            **meta,
            "avg_median": round(avg, 1),
            "tasks": tasks_summary,
        }

    return summary


def generate_markdown(summary: dict) -> str:
    """Generate a markdown report from the summary."""
    lines = [
        "# AI Agent Assessment Summary",
        "",
        f"**Runs:** {summary['num_runs']} | **Model:** {summary['model']}",
        "",
        "Scores shown are **median** across runs. Spread = max - min score across runs.",
        "",
    ]

    # Ranking table
    ranking = []
    for agent_key, agent_data in summary["agents"].items():
        scored = [
            t for t in agent_data["tasks"]
            if not t.get("gap") and "median_score" in t
        ]
        if scored:
            ranking.append({
                "key": agent_key,
                "name": agent_data["name"],
                "property": agent_data.get("property_name", ""),
                "avg": agent_data["avg_median"],
                "tasks_scored": len(scored),
                "tasks_total": len(agent_data["tasks"]),
            })
    ranking.sort(key=lambda r: r["avg"], reverse=True)

    lines.append("## Overall Ranking")
    lines.append("")
    lines.append("| Rank | Agent | Property | Avg Score | Tasks |")
    lines.append("|------|-------|----------|:---------:|:-----:|")
    for i, r in enumerate(ranking, 1):
        lines.append(
            f"| {i} | **{r['name']}** | {r['property']} "
            f"| **{r['avg']:.1f}** "
            f"| {r['tasks_scored']}/{r['tasks_total']} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-agent detail
    for agent_key, agent_data in summary["agents"].items():
        name = agent_data["name"]
        prop = agent_data.get("property_name", "")
        lines.append(f"## {name}" + (f" ({prop})" if prop else ""))
        lines.append("")

        lines.append("| Task | Median | Scores | Spread |")
        lines.append("|------|:------:|--------|:------:|")

        for task in agent_data["tasks"]:
            if task.get("gap"):
                lines.append(f"| {task['task_name']} | — | — | — |")
                continue

            median = task["median_score"]
            scores_str = ", ".join(str(s) for s in task["scores"])
            spread = task["spread"]
            flag = " :warning:" if spread >= 3 else ""
            lines.append(
                f"| {task['task_name']} | **{median:.0f}** "
                f"| {scores_str} | {spread}{flag} |"
            )

        lines.append("")

        # Per-task reasoning from each run
        for task in agent_data["tasks"]:
            if task.get("gap") or "per_run" not in task:
                continue

            lines.append(f"### {task['task_name']}")
            lines.append(f"**Median: {task['median_score']:.0f}** | "
                         f"Scores: {', '.join(str(s) for s in task['scores'])}")
            lines.append("")

            for j, entry in enumerate(task["per_run"], 1):
                lines.append(
                    f"- **Run {j}** (score {entry['score']}): {entry['reasoning']}"
                )
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/summarize.py <session_dir>", file=sys.stderr)
        sys.exit(1)

    session_dir = Path(sys.argv[1])
    if not session_dir.is_dir():
        print(f"Not a directory: {session_dir}", file=sys.stderr)
        sys.exit(1)

    runs = load_runs(session_dir)
    print(f"Loaded {len(runs)} assessment runs from {session_dir}")

    summary = summarize(runs)

    # Save JSON summary
    json_path = session_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary: {json_path}")

    # Save markdown summary
    md = generate_markdown(summary)
    md_path = session_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown summary: {md_path}")


if __name__ == "__main__":
    main()
