#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI


INPUT_GLOB = "*_subcaption_openai_input.jsonl"
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


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
    """
    Import the exact PROMPT and parse_subcaptions implementation from the
    user's local pmc-data-extraction repository.
    """
    sys.path.insert(0, str(pmc_root.resolve()))

    from openpmcvl.granular.pipeline.subcaption import (  # type: ignore
        PROMPT,
        parse_subcaptions,
    )

    return PROMPT, parse_subcaptions


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc

    return records


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def output_path_for(input_path: Path) -> Path:
    suffix = "_input.jsonl"

    if not input_path.name.endswith(suffix):
        raise ValueError(f"Unexpected prepared input filename: {input_path}")

    return input_path.with_name(
        input_path.name[: -len(suffix)] + "_output.jsonl"
    )


def write_chunk(
    run_dir: Path,
    chunk_number: int,
    requests: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> tuple[Path, Path]:
    request_path = run_dir / f"batch_{chunk_number:04d}_requests.jsonl"
    manifest_path = run_dir / f"batch_{chunk_number:04d}_manifest.jsonl"

    save_jsonl(requests, request_path)
    save_jsonl(manifest, manifest_path)

    return request_path, manifest_path


def get_request_counts(batch: Any) -> dict[str, int]:
    counts = getattr(batch, "request_counts", None)

    return {
        "total": int(getattr(counts, "total", 0) or 0),
        "completed": int(getattr(counts, "completed", 0) or 0),
        "failed": int(getattr(counts, "failed", 0) or 0),
    }


def submit(args: argparse.Namespace) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or pass --env-file."
        )

    if not 1 <= args.max_requests_per_batch <= 50_000:
        raise ValueError(
            "--max-requests-per-batch must be between 1 and 50000"
        )

    prompt, _ = load_vector_helpers(args.pmc_root)

    # These files are created by subcaption_separation.py when it runs
    # WITHOUT the old --run flag.
    input_files = sorted(args.out_root.rglob(INPUT_GLOB))

    if args.max_input_files is not None:
        input_files = input_files[: args.max_input_files]

    if not input_files:
        raise FileNotFoundError(
            f"No prepared files matching {INPUT_GLOB} under {args.out_root}. "
            "Run subcaption_separation.py without --run first."
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_root = args.out_root / "_batch_api"
    run_dir = batch_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    # status and collect default to the most recently submitted run.
    (batch_root / "latest_run.txt").write_text(
        str(run_dir),
        encoding="utf-8",
    )

    client = OpenAI()

    chunk_requests: list[dict[str, Any]] = []
    chunk_manifest: list[dict[str, Any]] = []
    chunk_number = 1
    total_requests = 0
    submitted_states: list[str] = []

    def flush_chunk() -> None:
        nonlocal chunk_number, chunk_requests, chunk_manifest

        if not chunk_requests:
            return

        request_path, manifest_path = write_chunk(
            run_dir=run_dir,
            chunk_number=chunk_number,
            requests=chunk_requests,
            manifest=chunk_manifest,
        )

        size_mb = request_path.stat().st_size / (1024 * 1024)

        if size_mb > 200:
            raise ValueError(
                f"{request_path} is {size_mb:.1f} MB. "
                "Reduce --max-requests-per-batch."
            )

        with request_path.open("rb") as file:
            uploaded_file = client.files.create(
                file=file,
                purpose="batch",
            )

        batch = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": (
                    f"Open-i subcaptions {timestamp} chunk {chunk_number}"
                )
            },
        )

        state_path = run_dir / f"batch_{chunk_number:04d}_state.json"
        state = {
            "batch_id": batch.id,
            "input_file_id": uploaded_file.id,
            "request_path": str(request_path),
            "manifest_path": str(manifest_path),
            "status": batch.status,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
        }
        state_path.write_text(
            json.dumps(state, indent=2),
            encoding="utf-8",
        )
        submitted_states.append(str(state_path))

        print(
            f"Submitted chunk {chunk_number}: "
            f"{len(chunk_requests)} requests, batch_id={batch.id}"
        )

        chunk_number += 1
        chunk_requests = []
        chunk_manifest = []

    for input_path in input_files:
        output_path = output_path_for(input_path)
        dataset = load_jsonl(input_path)

        for record_index, item in enumerate(dataset):
            caption = str(item.get("caption") or "")
            custom_id = f"request-{total_requests:012d}"

            # This exactly matches process_caption() in the local Vector script.
            user_prompt = f"Caption: \n{caption}".strip()

            chunk_requests.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": prompt,
                            },
                            {
                                "role": "user",
                                "content": user_prompt,
                            },
                        ],
                        "temperature": 0,
                        "max_tokens": args.max_tokens,
                    },
                }
            )

            # The manifest is local. It tells collect() where each result belongs.
            chunk_manifest.append(
                {
                    "custom_id": custom_id,
                    "source_input": str(input_path),
                    "target_output": str(output_path),
                    "record_index": record_index,
                    "item": item,
                }
            )

            total_requests += 1

            if len(chunk_requests) >= args.max_requests_per_batch:
                flush_chunk()

    flush_chunk()

    run_metadata = {
        "created_at": timestamp,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
        "input_file_count": len(input_files),
        "total_requests": total_requests,
        "state_files": submitted_states,
    }

    (run_dir / "run.json").write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Run directory: {run_dir}")
    print(
        f"Submitted {total_requests} requests "
        f"in {len(submitted_states)} batch(es)."
    )

    return 0


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return args.run_dir.resolve()

    latest_file = args.out_root / "_batch_api" / "latest_run.txt"

    if not latest_file.exists():
        raise FileNotFoundError(
            "No latest Batch API run was found. Pass --run-dir explicitly."
        )

    return Path(
        latest_file.read_text(encoding="utf-8").strip()
    ).resolve()


def retrieve_and_update_states(
    client: OpenAI,
    run_dir: Path,
) -> list[tuple[Path, dict[str, Any], Any]]:
    states: list[tuple[Path, dict[str, Any], Any]] = []

    for state_path in sorted(run_dir.glob("batch_*_state.json")):
        state = json.loads(
            state_path.read_text(encoding="utf-8")
        )
        batch = client.batches.retrieve(state["batch_id"])

        state.update(
            {
                "status": batch.status,
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
                "request_counts": get_request_counts(batch),
            }
        )
        state_path.write_text(
            json.dumps(state, indent=2),
            encoding="utf-8",
        )
        states.append((state_path, state, batch))

    return states


def status(args: argparse.Namespace) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or pass --env-file."
        )

    run_dir = resolve_run_dir(args)
    client = OpenAI()
    states = retrieve_and_update_states(client, run_dir)

    if not states:
        raise FileNotFoundError(
            f"No Batch API state files found in {run_dir}"
        )

    for state_path, state, _ in states:
        counts = state["request_counts"]

        print(
            f"{state_path.stem}: {state['status']} | "
            f"completed={counts['completed']}/{counts['total']} | "
            f"failed={counts['failed']}"
        )

    return 0


def extract_chat_content(row: dict[str, Any]) -> str | None:
    response = row.get("response")

    if not response or response.get("status_code") != 200:
        return None

    choices = response.get("body", {}).get("choices") or []

    if not choices:
        return None

    return choices[0].get("message", {}).get("content") or ""


def collect(args: argparse.Namespace) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or pass --env-file."
        )

    # Reuse the exact parser from the synchronous implementation.
    _, parse_subcaptions = load_vector_helpers(args.pmc_root)

    run_dir = resolve_run_dir(args)
    client = OpenAI()
    states = retrieve_and_update_states(client, run_dir)

    if not states:
        raise FileNotFoundError(
            f"No Batch API state files found in {run_dir}"
        )

    successful_outputs: dict[str, str] = {}
    failed_outputs: dict[str, dict[str, Any]] = {}
    all_manifest_entries: list[dict[str, Any]] = []
    unfinished = False

    for state_path, state, batch in states:
        all_manifest_entries.extend(
            load_jsonl(Path(state["manifest_path"]))
        )

        if batch.status not in TERMINAL_STATUSES:
            unfinished = True
            print(
                f"Skipping {state_path.stem}: status={batch.status}"
            )
            continue

        if batch.output_file_id:
            raw_output_path = state_path.with_name(
                state_path.name.replace(
                    "_state.json",
                    "_raw_output.jsonl",
                )
            )

            if not raw_output_path.exists():
                response = client.files.content(
                    batch.output_file_id
                )
                raw_output_path.write_text(
                    response.text,
                    encoding="utf-8",
                )

            for row in load_jsonl(raw_output_path):
                custom_id = row["custom_id"]
                content = extract_chat_content(row)

                if content is None:
                    failed_outputs[custom_id] = row
                else:
                    successful_outputs[custom_id] = content

        if batch.error_file_id:
            raw_error_path = state_path.with_name(
                state_path.name.replace(
                    "_state.json",
                    "_raw_errors.jsonl",
                )
            )

            if not raw_error_path.exists():
                response = client.files.content(
                    batch.error_file_id
                )
                raw_error_path.write_text(
                    response.text,
                    encoding="utf-8",
                )

            for row in load_jsonl(raw_error_path):
                failed_outputs[row["custom_id"]] = row

    entries_by_output: dict[
        str, list[dict[str, Any]]
    ] = defaultdict(list)

    for entry in all_manifest_entries:
        entries_by_output[entry["target_output"]].append(entry)

    complete_files = 0
    incomplete_files = 0

    for output_name, entries in entries_by_output.items():
        entries.sort(key=lambda entry: entry["record_index"])

        results: list[dict[str, Any]] = []
        missing_ids: list[str] = []

        for entry in entries:
            custom_id = entry["custom_id"]
            output = successful_outputs.get(custom_id)

            if output is None:
                missing_ids.append(custom_id)
                continue

            item = dict(entry["item"])
            subcaptions = parse_subcaptions(output)

            # These fields match the synchronous subcaption.py output.
            item["num_subcaptions"] = len(subcaptions)
            item["subcaptions"] = subcaptions
            item["llm_output"] = output

            results.append(item)

        output_path = Path(output_name)

        if missing_ids:
            incomplete_files += 1
            partial_path = output_path.with_name(
                output_path.name.replace(
                    "_output.jsonl",
                    "_partial_output.jsonl",
                )
            )

            if results:
                save_jsonl(results, partial_path)

            print(
                f"Incomplete: {output_path} | "
                f"missing={len(missing_ids)}/{len(entries)}"
            )
        else:
            save_jsonl(results, output_path)
            complete_files += 1
            print(f"Saved: {output_path}")

    print(
        "Collection summary: "
        f"complete_files={complete_files}, "
        f"incomplete_files={incomplete_files}, "
        f"failed_requests={len(failed_outputs)}"
    )

    return 1 if unfinished or incomplete_files else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open-i subcaption Batch API driver"
    )
    parser.add_argument(
        "mode",
        choices=("submit", "status", "collect"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "/nfs/turbo/umms-tocho-ns/data/2d_project/openi/"
            "subfigure_parsed_practice"
        ),
    )
    parser.add_argument(
        "--pmc-root",
        type=Path,
        default=Path(
            "/nfs/turbo/umms-tocho-ns/code/chjudy/"
            "pmc-data-extraction"
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Optional prior run directory; defaults to latest_run.txt",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--max-requests-per-batch",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--max-input-files",
        type=int,
        default=None,
        help=(
            "Testing only: limit the number of prepared "
            "20-caption input files"
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.out_root = args.out_root.resolve()
    args.pmc_root = args.pmc_root.resolve()

    if args.env_file:
        apply_env_file(args.env_file.resolve())

    if args.mode == "submit":
        return submit(args)

    if args.mode == "status":
        return status(args)

    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
