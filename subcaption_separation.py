#!/usr/bin/env python3
"""Build caption JSONLs for the full Open-i dataset and optionally process them."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def apply_env_file(path: Path) -> None:
    """Load KEY=value lines into os.environ (OPENAI_API_KEY, etc.)."""
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ[key] = val


def oa_rows(source_json: Path, data: list) -> list[dict]:
    """Rows for the OA layout: a flat list of figures per article directory."""
    article = source_json.parent.name
    img_dir = source_json.parent / "img"
    rows: list[dict] = []
    for item in data:
        cap = (item.get("caption") or "").strip()
        # Tables carry no subcaptions, so they are not worth an API request.
        if item.get("figType") != "Figure":
            continue
        fig_id = f"{item.get('figType')}{item.get('name')}"
        rows.append(
            {
                "figure_key": f"{article}_{fig_id}",
                "article_id": article,
                "figure_id": fig_id,
                "caption": cap,
                # renderURL points at the original extraction host, so rebuild
                # the path against the nested img/ dir we actually have.
                "image_path": str(img_dir / Path(item.get("renderURL", "")).name),
            }
        )
    return rows


def openi_rows(data: dict) -> list[dict]:
    rows: list[dict] = []
    for item in data.get("list", []):
        img = item.get("image") or {}
        cap = (img.get("caption") or "").strip()
        pmcid = str(item.get("pmcid") or "").strip()
        fig_id = str(img.get("id") or "").strip()
        rows.append(
            {
                "figure_key": f"{pmcid}_{fig_id}" if pmcid and fig_id else fig_id or pmcid,
                "uid": item.get("uid"),
                "pmcid": pmcid,
                "figure_id": fig_id,
                "caption": cap,
                "title": item.get("title"),
            }
        )
    return rows


def write_input_jsonl(openi_json: Path, out_jsonl: Path) -> int:
    with open(openi_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = oa_rows(openi_json, data) if isinstance(data, list) else openi_rows(data)
    if not rows:
        return 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for r in rows:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def discover_openi_jsons(openi_root: Path) -> list[tuple[str, Path]]:
    """Return all group/batch JSON files (Open-i modalities or OA journals)."""
    discovered: list[tuple[str, Path]] = []
    for group_dir in sorted(p for p in openi_root.iterdir() if p.is_dir()):
        # The */*.json depth keeps per-group files such as OA's stats.json out.
        json_files = list(group_dir.glob("*/*.json"))

        def batch_start(path: Path) -> tuple[int, str]:
            try:
                return int(path.parent.name.split("-", 1)[0]), path.name
            except ValueError:
                return sys.maxsize, path.name

        discovered.extend((group_dir.name, path) for path in sorted(json_files, key=batch_start))
    return discovered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--openi-root",
        type=Path,
        default=Path(
            "/nfs/turbo/umms-tocho-ns/data/2d_project/openi/unparsed"
        ),
        help="Root containing the four Open-i modality directories",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "/nfs/turbo/umms-tocho-ns/data/2d_project/openi/subfigure_parsed"
        ),
        help="Output root; modality and batch directory structure is mirrored here",
    )
    ap.add_argument(
        "--pmc-root",
        type=Path,
        default=Path("/nfs/turbo/umms-tocho-ns/code/chjudy/pmc-data-extraction"),
    )
    ap.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OPENAI_SUBCAPTION_MODEL", "gpt-4o-mini"),
    )
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional file with OPENAI_API_KEY=... lines (chmod 600 recommended)",
    )
    ap.add_argument(
        "--run",
        action="store_true",
        help="Invoke subcaption.py after writing input JSONL",
    )
    args = ap.parse_args()

    if args.env_file:
        apply_env_file(args.env_file.resolve())

    if args.run and not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Export it or use --env-file with KEY=value lines.",
            file=sys.stderr,
        )
        return 1

    openi_root = args.openi_root.resolve()
    out_root = args.out_root.resolve()
    sources = discover_openi_jsons(openi_root)
    if not sources:
        print(f"No Open-i JSON files found under {openi_root}", file=sys.stderr)
        return 1

    sub_py = (
        args.pmc_root / "openpmcvl" / "granular" / "pipeline" / "subcaption.py"
    )
    env = {**dict(os.environ), "PYTHONPATH": str(args.pmc_root.resolve())}
    failures: list[tuple[Path, str]] = []
    total_records = 0

    print(f"Found {len(sources)} Open-i batches.", file=sys.stderr)
    for index, (modality, openi_json) in enumerate(sources, start=1):
        batch = openi_json.parent.name
        file_prefix = f"{modality.removesuffix('_png')}_{batch}_subcaption_openai"
        out_dir = out_root / modality / batch
        inp = out_dir / f"{file_prefix}_input.jsonl"
        outp = out_dir / f"{file_prefix}_output.jsonl"

        try:
            count = write_input_jsonl(openi_json, inp)
            total_records += count
            print(
                f"[{index}/{len(sources)}] Wrote {count} caption records to {inp}",
                file=sys.stderr,
            )
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            failures.append((openi_json, f"input preparation failed: {exc}"))
            print(f"[{index}/{len(sources)}] Failed: {openi_json}: {exc}", file=sys.stderr)
            continue

        if not args.run or count == 0:
            continue

        cmd = [
            sys.executable,
            str(sub_py),
            "--input-file",
            str(inp),
            "--output-file",
            str(outp),
            "--model",
            args.model,
            "--max-tokens",
            str(args.max_tokens),
        ]
        print("Running:", " ".join(cmd), file=sys.stderr)
        try:
            rc = subprocess.call(cmd, cwd=str(args.pmc_root.resolve()), env=env)
        except OSError as exc:
            failures.append((openi_json, f"subcaption.py could not start: {exc}"))
            continue
        if rc == 0:
            print(f"Saved: {outp}", file=sys.stderr)
        else:
            failures.append((openi_json, f"subcaption.py exited with status {rc}"))

    action = "Processed" if args.run else "Prepared"
    print(
        f"{action} {len(sources) - len(failures)}/{len(sources)} batches "
        f"({total_records} caption records).",
        file=sys.stderr,
    )
    if not args.run:
        print("Skipping subcaption.py (pass --run). Requires OPENAI_API_KEY.", file=sys.stderr)
    if failures:
        print(f"{len(failures)} batch(es) failed:", file=sys.stderr)
        for path, reason in failures:
            print(f"  {path}: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
