#!/usr/bin/env python3
"""Batch OCR letter detection for Open-i subfigures via the OpenAI Batch API."""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

MODALITIES = ("open_journal_of_neuroimaging", "stroke")

DATA_ROOT = Path(
    "/nfs/turbo/umms-tocho-ns/data/2d_project/oa/subfigure_parsed"
)
PROMPT = (
    "What is the letter in this image? Please output the letter only, "
    "with no other text or characters. If there is no letter detected, "
    "output an empty string."
)
TERMINAL = {"completed", "failed", "expired", "cancelled"}
# OpenAI Batch input files must stay under 200 MB; leave headroom for encoding.
DEFAULT_MAX_BATCH_MB = 190.0


def load_env(path: Path) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip().strip("\"'")


def require_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY unset. Export it or pass --env-file.")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Bad JSON in {path}:{i}: {e}") from e
    return rows


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def encode_image(path: Path) -> tuple[str, str]:
    mime = mimetypes.guess_type(str(path))[0] or {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(path.suffix.lower(), "image/jpeg")
    return mime, base64.b64encode(path.read_bytes()).decode()


def out_path(inp: Path, model: str) -> Path:
    stem = inp.name.removesuffix(".jsonl")
    return inp.with_name(f"{stem}_with_letters_{model.replace('/', '-')}.jsonl")


def discover(root: Path, modalities: list[str]) -> list[Path]:
    paths = []
    for mod in modalities:
        mod_dir = root / mod
        if not mod_dir.is_dir():
            print(f"Warning: missing {mod_dir}")
            continue
        for batch in sorted(p for p in mod_dir.iterdir() if p.is_dir()):
            paths.extend(
                sorted(
                    p
                    for p in batch.glob("*_subfigures.jsonl")
                    if not p.name.endswith("_failed.jsonl")
                    and "_with_letters_" not in p.name
                )
            )
    return paths


def batch_counts(batch: Any) -> dict[str, int]:
    c = getattr(batch, "request_counts", None)
    return {
        "total": int(getattr(c, "total", 0) or 0),
        "completed": int(getattr(c, "completed", 0) or 0),
        "failed": int(getattr(c, "failed", 0) or 0),
    }


def run_dir_from(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return args.run_dir.resolve()
    latest = args.data_root / "_ocr_batch_api" / "latest_run.txt"
    if not latest.exists():
        raise FileNotFoundError("No latest run. Pass --run-dir.")
    return Path(latest.read_text().strip()).resolve()


def refresh_states(
    client: OpenAI, run_dir: Path
) -> list[tuple[Path, dict[str, Any], Any]]:
    out = []
    for sp in sorted(run_dir.glob("batch_*_state.json")):
        state = json.loads(sp.read_text())
        batch = client.batches.retrieve(state["batch_id"])
        state.update(
            {
                "status": batch.status,
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
                "request_counts": batch_counts(batch),
            }
        )
        sp.write_text(json.dumps(state, indent=2))
        out.append((sp, state, batch))
    return out


def extract_text(row: dict[str, Any]) -> str | None:
    resp = row.get("response")
    if not resp or resp.get("status_code") != 200:
        return None
    body = resp.get("body") or {}
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    texts = []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") in {"output_text", "text"}:
                texts.append(str(part.get("text") or ""))
    return "".join(texts) if texts else None


def submit(args: argparse.Namespace) -> int:
    require_key()
    if not 1 <= args.max_requests_per_batch <= 50_000:
        raise ValueError("--max-requests-per-batch must be 1..50000")
    if not 1.0 <= args.max_batch_mb <= 200.0:
        raise ValueError("--max-batch-mb must be 1..200")

    files = discover(args.data_root, args.modalities)
    if args.skip_existing:
        files = [p for p in files if not out_path(p, args.model).exists()]
    if args.max_input_files is not None:
        files = files[: args.max_input_files]
    if not files:
        raise FileNotFoundError(f"No *_subfigures.jsonl under {args.data_root}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_root = args.data_root / "_ocr_batch_api"
    run_dir = batch_root / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (batch_root / "latest_run.txt").write_text(str(run_dir))

    client = OpenAI()
    reqs: list[dict[str, Any]] = []
    man: list[dict[str, Any]] = []
    chunk = 1
    total = 0
    missing = 0
    states: list[str] = []
    chunk_bytes = 0
    max_bytes = int(args.max_batch_mb * 1024 * 1024)

    def flush() -> None:
        nonlocal chunk, reqs, man, chunk_bytes
        if not reqs:
            return
        req_path = run_dir / f"batch_{chunk:04d}_requests.jsonl"
        man_path = run_dir / f"batch_{chunk:04d}_manifest.jsonl"
        save_jsonl(reqs, req_path)
        save_jsonl(man, man_path)
        mb = req_path.stat().st_size / (1024 * 1024)
        if mb > 200:
            raise ValueError(
                f"{req_path} is {mb:.1f} MB; lower --max-batch-mb "
                f"or --max-requests-per-batch"
            )

        with req_path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"description": f"OCR letters {ts} chunk {chunk}"},
        )
        sp = run_dir / f"batch_{chunk:04d}_state.json"
        sp.write_text(
            json.dumps(
                {
                    "batch_id": batch.id,
                    "input_file_id": uploaded.id,
                    "request_path": str(req_path),
                    "manifest_path": str(man_path),
                    "status": batch.status,
                    "output_file_id": batch.output_file_id,
                    "error_file_id": batch.error_file_id,
                },
                indent=2,
            )
        )
        states.append(str(sp))
        print(f"Submitted chunk {chunk}: {len(reqs)} reqs, {batch.id}, {mb:.1f} MB")
        chunk += 1
        reqs, man = [], []
        chunk_bytes = 0

    for inp in files:
        target = out_path(inp, args.model)
        for idx, item in enumerate(load_jsonl(inp)):
            img = Path(str(item.get("subfig_path") or ""))
            if not img.is_file():
                missing += 1
                print(f"Missing image, skipping: {img}")
                continue
            mime, b64 = encode_image(img)
            cid = f"request-{total:012d}"
            req = {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": args.model,
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": PROMPT},
                                {
                                    "type": "input_image",
                                    "image_url": f"data:{mime};base64,{b64}",
                                },
                            ],
                        }
                    ],
                },
            }
            req_bytes = len(json.dumps(req, ensure_ascii=False).encode()) + 1
            # Flush before adding if this request would push past the size cap.
            if reqs and (
                chunk_bytes + req_bytes > max_bytes
                or len(reqs) >= args.max_requests_per_batch
            ):
                flush()
            reqs.append(req)
            man.append(
                {
                    "custom_id": cid,
                    "source_input": str(inp),
                    "target_output": str(target),
                    "record_index": idx,
                    "item": item,
                }
            )
            chunk_bytes += req_bytes
            total += 1
    flush()

    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "created_at": ts,
                "model": args.model,
                "modalities": args.modalities,
                "input_file_count": len(files),
                "total_requests": total,
                "skipped_missing_images": missing,
                "max_batch_mb": args.max_batch_mb,
                "max_requests_per_batch": args.max_requests_per_batch,
                "state_files": states,
            },
            indent=2,
        )
    )
    print(f"Run directory: {run_dir}")
    print(f"Submitted {total} requests in {len(states)} batch(es).")
    if missing:
        print(f"Skipped {missing} missing image(s).")
    return 0


def status(args: argparse.Namespace) -> int:
    require_key()
    states = refresh_states(OpenAI(), run_dir_from(args))
    if not states:
        raise FileNotFoundError("No batch state files found.")
    for sp, state, _ in states:
        c = state["request_counts"]
        print(
            f"{sp.stem}: {state['status']} | "
            f"completed={c['completed']}/{c['total']} | failed={c['failed']}"
        )
    return 0


def collect(args: argparse.Namespace) -> int:
    require_key()
    run_dir = run_dir_from(args)
    client = OpenAI()
    states = refresh_states(client, run_dir)
    if not states:
        raise FileNotFoundError("No batch state files found.")

    ok: dict[str, str] = {}
    failed: dict[str, dict[str, Any]] = {}
    manifest: list[dict[str, Any]] = []
    unfinished = False

    for sp, state, batch in states:
        manifest.extend(load_jsonl(Path(state["manifest_path"])))
        if batch.status not in TERMINAL:
            unfinished = True
            print(f"Skipping {sp.stem}: status={batch.status}")
            continue

        if batch.output_file_id:
            raw = sp.with_name(sp.name.replace("_state.json", "_raw_output.jsonl"))
            if not raw.exists():
                raw.write_text(client.files.content(batch.output_file_id).text)
            for row in load_jsonl(raw):
                text = extract_text(row)
                if text is None:
                    failed[row["custom_id"]] = row
                else:
                    ok[row["custom_id"]] = text.strip()

        if batch.error_file_id:
            err = sp.with_name(sp.name.replace("_state.json", "_raw_errors.jsonl"))
            if not err.exists():
                err.write_text(client.files.content(batch.error_file_id).text)
            for row in load_jsonl(err):
                failed[row["custom_id"]] = row

    by_out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in manifest:
        by_out[e["target_output"]].append(e)

    complete = incomplete = 0
    for out_name, entries in by_out.items():
        entries.sort(key=lambda e: e["record_index"])
        rows, missing_ids = [], []
        for e in entries:
            letter = ok.get(e["custom_id"])
            if letter is None:
                missing_ids.append(e["custom_id"])
                continue
            item = dict(e["item"])
            item["letter"] = letter
            rows.append(item)

        out = Path(out_name)
        if missing_ids:
            incomplete += 1
            if rows:
                save_jsonl(
                    rows,
                    out.with_name(
                        out.name.replace("_with_letters_", "_with_letters_partial_", 1)
                    ),
                )
            print(f"Incomplete: {out} | missing={len(missing_ids)}/{len(entries)}")
        else:
            save_jsonl(rows, out)
            complete += 1
            print(f"Saved: {out}")

    print(
        f"Collection summary: complete_files={complete}, "
        f"incomplete_files={incomplete}, failed_requests={len(failed)}"
    )
    return 1 if unfinished or incomplete else 0


def main() -> int:
    p = argparse.ArgumentParser(description="Open-i subfigure letter OCR (Batch API)")
    p.add_argument("mode", choices=("submit", "status", "collect"))
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument(
        "--modalities",
        nargs="+",
        default=list(MODALITIES),
        choices=list(MODALITIES),
    )
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument(
        "--max-requests-per-batch",
        type=int,
        default=50_000,
        help="Hard cap on requests per OpenAI batch (API max is 50000).",
    )
    p.add_argument(
        "--max-batch-mb",
        type=float,
        default=DEFAULT_MAX_BATCH_MB,
        help=(
            "Flush a chunk once estimated JSONL size reaches this many MB "
            "(OpenAI limit is 200). Image OCR is usually size-bound."
        ),
    )
    p.add_argument("--max-input-files", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--env-file", type=Path)
    args = p.parse_args()
    args.data_root = args.data_root.resolve()
    if args.env_file:
        load_env(args.env_file.resolve())

    return {"submit": submit, "status": status, "collect": collect}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
