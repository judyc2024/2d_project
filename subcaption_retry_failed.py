#!/usr/bin/env python3
"""Repair caption files left incomplete by transient Batch API failures.

The Jul 21 run finished with 153 requests returning ``server_error``, which
left 144 caption files written as ``*_subcaption_openai_partial_output.jsonl``
instead of ``*_subcaption_openai_output.jsonl``. This replays only the missing
requests against the chat endpoint, rebuilds each affected file from the
cached batch output plus the replayed results, and mirrors the finished file
into the tree consumed by match_figure_caption_v2.py.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI


def apply_env_file(path: Path) -> None:
    """Load KEY=value pairs, such as OPENAI_API_KEY=..."""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def load_vector_helpers(
    pmc_root: Path,
) -> tuple[str, Callable[[str], dict[str, str]]]:
    sys.path.insert(0, str(pmc_root.resolve()))

    from openpmcvl.granular.pipeline.subcaption import (  # type: ignore
        PROMPT,
        parse_subcaptions,
    )

    return PROMPT, parse_subcaptions


def iter_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return args.run_dir.resolve()

    latest_file = args.out_root / "_batch_api" / "latest_run.txt"

    if not latest_file.exists():
        raise FileNotFoundError(
            "No latest Batch API run was found. Pass --run-dir explicitly."
        )

    return Path(latest_file.read_text(encoding="utf-8").strip()).resolve()


def extract_chat_content(row: dict[str, Any]) -> str | None:
    response = row.get("response")

    if not response or response.get("status_code") != 200:
        return None

    choices = response.get("body", {}).get("choices") or []

    if not choices:
        return None

    return choices[0].get("message", {}).get("content") or ""


def successful_ids(run_dir: Path) -> set[str]:
    """custom_ids that came back with usable content in the cached output."""
    found: set[str] = set()

    for path in sorted(run_dir.glob("batch_*_raw_output.jsonl")):
        for row in iter_jsonl(path):
            if extract_chat_content(row) is not None:
                found.add(row["custom_id"])

    return found


def manifest_index(run_dir: Path) -> dict[str, tuple[str, int]]:
    """custom_id -> (target_output, record_index) for the whole run."""
    index: dict[str, tuple[str, int]] = {}

    for path in sorted(run_dir.glob("batch_*_manifest.jsonl")):
        for entry in iter_jsonl(path):
            index[entry["custom_id"]] = (
                entry["target_output"],
                entry["record_index"],
            )

    if not index:
        raise FileNotFoundError(f"No manifests found in {run_dir}")

    return index


def manifest_entries_for(
    run_dir: Path,
    targets: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Full manifest entries, grouped by target_output, for chosen targets."""
    grouped: dict[str, list[dict[str, Any]]] = {t: [] for t in targets}

    for path in sorted(run_dir.glob("batch_*_manifest.jsonl")):
        for entry in iter_jsonl(path):
            target = entry["target_output"]

            if target in grouped:
                grouped[target].append(entry)

    for entries in grouped.values():
        entries.sort(key=lambda entry: entry["record_index"])

    return grouped


def cached_content_for(
    run_dir: Path,
    wanted: set[str],
) -> dict[str, str]:
    """Batch results we already have on disk for the affected files."""
    contents: dict[str, str] = {}

    for path in sorted(run_dir.glob("batch_*_raw_output.jsonl")):
        for row in iter_jsonl(path):
            custom_id = row["custom_id"]

            if custom_id not in wanted:
                continue

            content = extract_chat_content(row)

            if content is not None:
                contents[custom_id] = content

    return contents


def request_caption(
    client: OpenAI,
    prompt: str,
    caption: str,
    model: str,
    max_tokens: int,
    max_attempts: int,
) -> str:
    """Mirror the Batch API request body exactly, with backoff on failure."""
    user_prompt = f"Caption: \n{caption}".strip()
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )

            return completion.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - transient API failures
            last_error = exc

            if attempt < max_attempts:
                time.sleep(min(2 ** attempt, 30))

    raise RuntimeError(
        f"Gave up after {max_attempts} attempts: {last_error}"
    )


def mirror_path(
    output_path: Path,
    source_root: Path,
    mirror_root: Path,
) -> Path | None:
    try:
        relative = output_path.relative_to(source_root)
    except ValueError:
        return None

    return mirror_root / relative


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay failed subcaption Batch API requests"
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "/nfs/turbo/umms-tocho-ns/data/2d_project/openi/"
            "subfigure_parsed_practice"
        ),
        help="Root the Batch API run was submitted against",
    )
    parser.add_argument(
        "--mirror-root",
        type=Path,
        default=Path(
            "/nfs/turbo/umms-tocho-ns/data/2d_project/openi/subfigure_parsed"
        ),
        help="Also copy each repaired file here; pass 'none' to skip",
    )
    parser.add_argument(
        "--pmc-root",
        type=Path,
        default=Path(
            "/nfs/turbo/umms-tocho-ns/code/chjudy/pmc-data-extraction"
        ),
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--remove-partial",
        action="store_true",
        help="Delete *_partial_output.jsonl once the full file is written",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be replayed without calling the API",
    )
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()

    args.out_root = args.out_root.resolve()
    args.pmc_root = args.pmc_root.resolve()

    if args.env_file:
        apply_env_file(args.env_file.resolve())

    run_dir = resolve_run_dir(args)
    index = manifest_index(run_dir)
    have = successful_ids(run_dir)
    missing = sorted(set(index) - have)

    if not missing:
        print(f"Nothing to replay: {run_dir} is already complete.")
        return 0

    targets = {index[custom_id][0] for custom_id in missing}
    print(
        f"Run: {run_dir}\n"
        f"Missing requests: {len(missing)} across {len(targets)} caption files"
    )

    if args.dry_run:
        for custom_id in missing:
            target, record_index = index[custom_id]
            print(f"  {custom_id} record={record_index} -> {target}")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or pass --env-file."
        )

    prompt, parse_subcaptions = load_vector_helpers(args.pmc_root)
    grouped = manifest_entries_for(run_dir, targets)
    wanted = {
        entry["custom_id"]
        for entries in grouped.values()
        for entry in entries
    }
    contents = cached_content_for(run_dir, wanted)
    client = OpenAI()

    def replay(custom_id: str) -> tuple[str, str | None, str | None]:
        target, record_index = index[custom_id]
        entry = next(
            e
            for e in grouped[target]
            if e["custom_id"] == custom_id
        )

        try:
            output = request_caption(
                client=client,
                prompt=prompt,
                caption=str(entry["item"].get("caption") or ""),
                model=args.model,
                max_tokens=args.max_tokens,
                max_attempts=args.max_attempts,
            )
        except RuntimeError as exc:
            return custom_id, None, str(exc)

        return custom_id, output, None

    replay_errors: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, (custom_id, output, error) in enumerate(
            pool.map(replay, missing), start=1
        ):
            if error is None:
                contents[custom_id] = output or ""
            else:
                replay_errors[custom_id] = error

            if done % 25 == 0 or done == len(missing):
                print(f"  replayed {done}/{len(missing)}")

    repaired = 0
    still_incomplete = 0
    mirror_root = (
        None
        if str(args.mirror_root).lower() == "none"
        else args.mirror_root.resolve()
    )

    for target, entries in grouped.items():
        records: list[dict[str, Any]] = []
        gaps = 0

        for entry in entries:
            output = contents.get(entry["custom_id"])

            if output is None:
                gaps += 1
                continue

            item = dict(entry["item"])
            subcaptions = parse_subcaptions(output)
            item["num_subcaptions"] = len(subcaptions)
            item["subcaptions"] = subcaptions
            item["llm_output"] = output
            records.append(item)

        output_path = Path(target)

        if gaps:
            still_incomplete += 1
            print(f"Still incomplete: {output_path} | missing={gaps}")
            continue

        save_jsonl(records, output_path)
        repaired += 1

        if mirror_root is not None:
            destination = mirror_path(
                output_path,
                args.out_root,
                mirror_root,
            )

            if destination is None:
                print(f"Not under --out-root, skipped mirror: {output_path}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output_path, destination)

        partial_path = output_path.with_name(
            output_path.name.replace("_output.jsonl", "_partial_output.jsonl")
        )

        if args.remove_partial and partial_path.exists():
            partial_path.unlink()

    print(
        "Repair summary: "
        f"repaired_files={repaired}, "
        f"still_incomplete={still_incomplete}, "
        f"replay_failures={len(replay_errors)}"
    )

    for custom_id, error in replay_errors.items():
        print(f"  {custom_id}: {error}")

    return 1 if still_incomplete or replay_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
