#!/usr/bin/env python3
"""
Build granular-style metadata JSONL files and optionally run OpenPMCVL's
subfigure separation pipeline over the entire OA dataset.

The input directory is expected to look approximately like:

oa/unparsed
├── brain/
│   ├── 10.1093_brain_119.1.89/
│   │   ├── 10.1093_brain_119.1.89.json
│   │   └── img/*.png
│   └── ...
├── neuron/
└── stroke/

The output mirrors <journal>/<article> under the output root.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def make_tag(relative_dir: Path) -> str:
    """
    Convert a relative folder path into a safe filename prefix.

    Example:
        brain/10.1093_brain_119.1.89 -> brain_10.1093_brain_119.1.89
    """
    tag = "_".join(relative_dir.parts)
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)
    return tag.strip("_") or "oa"


def iter_article_dirs(journal_dir: Path):
    """Yield (article directory, its PNG figures) for one journal.

    Articles with an empty img/ directory are skipped, as are Table*.png
    renders, which hold no subfigures and mirror the figType filter used by
    subcaption_separation.py.
    """
    for article_dir in sorted(p for p in journal_dir.iterdir() if p.is_dir()):
        pngs = sorted(
            png
            for png in (article_dir / "img").glob("*.png")
            if png.name.startswith("Figure")
        )

        if pngs:
            yield article_dir, pngs


def load_captions(article_dir: Path) -> dict[str, str]:
    """Map image filename to caption using the article's sibling JSON file."""
    article_json = article_dir / f"{article_dir.name}.json"

    if not article_json.is_file():
        return {}

    try:
        items = json.loads(article_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"WARNING: Could not read {article_json}: {exc}",
            file=sys.stderr,
        )
        return {}

    # renderURL points at the original extraction host, so only the basename
    # is usable for matching against the local img/ directory.
    return {
        Path(item.get("renderURL", "")).name: item.get("caption") or ""
        for item in items
        if isinstance(item, dict)
    }


def build_metadata(
    pngs: list[Path],
    meta_out: Path,
    bad_images: list[dict[str, str]],
    article: str,
    captions: dict[str, str],
) -> int:
    """
    Create one metadata JSONL file for a directory of PNG files.

    Returns the number of valid records written.
    """
    records: list[str] = []

    for png in pngs:
        stem = png.stem

        try:
            with Image.open(png) as image:
                width, height = image.size
        except (OSError, UnidentifiedImageError) as exc:
            print(
                f"WARNING: Could not read image: {png}: {exc}",
                file=sys.stderr,
            )
            bad_images.append(
                {
                    "image_path": str(png),
                    "error": str(exc),
                }
            )
            continue

        # subfigure.py expects the ID to end in ".jpg" because it uses:
        # img_ids[i].split(".jpg")[0]
        record = {
            "id": f"{stem}.jpg",
            "PMC_ID": article,
            "caption": captions.get(png.name, ""),
            "image_path": str(png),
            "width": width,
            "height": height,
            "media_id": stem,
            "media_url": "",
            "media_name": png.name,
            "keywords": [],
            "is_medical": True,
        }

        records.append(json.dumps(record, ensure_ascii=False))

    if not records:
        return 0

    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(
        "\n".join(records) + "\n",
        encoding="utf-8",
    )

    return len(records)


def run_subfigure(
    *,
    subfigure_script: Path,
    checkpoint: Path,
    pmc_root: Path,
    meta_out: Path,
    save_dir: Path,
    rcd_file: Path,
    batch_size: int,
    num_workers: int,
    gpu: str,
    score_threshold: float,
    nms_threshold: float,
) -> int:
    """Run OpenPMCVL's subfigure.py for one image directory."""
    command = [
        sys.executable,
        str(subfigure_script),
        "--separation_model",
        str(checkpoint),
        "--eval_file",
        str(meta_out),
        "--save_path",
        str(save_dir),
        "--rcd_file",
        str(rcd_file),
        "--score_threshold",
        str(score_threshold),
        "--nms_threshold",
        str(nms_threshold),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--gpu",
        str(gpu),
    ]

    env = os.environ.copy()

    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{pmc_root}:{existing_pythonpath}"
        if existing_pythonpath
        else str(pmc_root)
    )

    print("\nRunning command:", file=sys.stderr)
    print(" ".join(command), file=sys.stderr)

    completed = subprocess.run(
        command,
        cwd=str(pmc_root),
        env=env,
        check=False,
    )

    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "/nfs/turbo/umms-tocho-ns/data/2d_project/oa/unparsed"
        ),
        help="Root containing the OA journal folders.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/nfs/turbo/umms-tocho-ns/data/2d_project/oa/subfigure_parsed"
        ),
        help="Root where metadata and subfigure outputs will be written.",
    )

    parser.add_argument(
        "--journals",
        nargs="+",
        default=None,
        help="Journal folders to process. Defaults to every journal found.",
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
        "--run",
        action="store_true",
        help="Run subfigure.py after generating each metadata file.",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip folders whose subfigure JSONL output already exists.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing other folders if one invocation fails.",
    )

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.4)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    pmc_root = args.pmc_root.resolve()

    subfigure_script = (
        pmc_root
        / "openpmcvl"
        / "granular"
        / "pipeline"
        / "subfigure.py"
    )

    checkpoint = (
        pmc_root
        / "openpmcvl"
        / "granular"
        / "checkpoints"
        / "subfigure_detector.pth"
    )

    if not input_root.is_dir():
        print(
            f"ERROR: Input root does not exist: {input_root}",
            file=sys.stderr,
        )
        return 1

    if args.run:
        if not subfigure_script.is_file():
            print(
                f"ERROR: subfigure.py not found: {subfigure_script}",
                file=sys.stderr,
            )
            return 1

        if not checkpoint.is_file():
            print(
                f"ERROR: Checkpoint not found: {checkpoint}",
                file=sys.stderr,
            )
            return 1

    output_root.mkdir(parents=True, exist_ok=True)

    total_folders = 0
    completed_folders = 0
    skipped_folders = 0
    failed_folders: list[dict[str, object]] = []
    bad_images: list[dict[str, str]] = []
    total_records = 0

    journals = args.journals or sorted(
        p.name for p in input_root.iterdir() if p.is_dir()
    )

    for journal_name in journals:
        journal_dir = input_root / journal_name

        if not journal_dir.is_dir():
            print(
                f"WARNING: Journal directory not found; skipping: "
                f"{journal_dir}",
                file=sys.stderr,
            )
            continue

        print(f"\nDiscovering articles under {journal_dir}")

        for article_dir, pngs in iter_article_dirs(journal_dir):
            total_folders += 1
            png_dir = article_dir / "img"

            relative_dir = article_dir.relative_to(input_root)
            save_dir = output_root / relative_dir
            save_dir.mkdir(parents=True, exist_ok=True)

            tag = make_tag(relative_dir)

            meta_out = save_dir / f"{tag}_meta.jsonl"
            rcd_file = save_dir / f"{tag}_subfigures.jsonl"

            print(
                f"\n[{total_folders}] Processing: {png_dir}\n"
                f"    PNG files: {len(pngs)}\n"
                f"    Output:    {save_dir}"
            )

            if (
                args.run
                and args.skip_existing
                and rcd_file.is_file()
                and rcd_file.stat().st_size > 0
            ):
                print(f"    Skipping existing output: {rcd_file}")
                skipped_folders += 1
                continue

            record_count = build_metadata(
                pngs=pngs,
                meta_out=meta_out,
                bad_images=bad_images,
                article=article_dir.name,
                captions=load_captions(article_dir),
            )

            if record_count == 0:
                print(
                    f"WARNING: No valid images found in {png_dir}",
                    file=sys.stderr,
                )
                failed_folders.append(
                    {
                        "input_directory": str(png_dir),
                        "return_code": None,
                        "reason": "No valid images",
                    }
                )

                if not args.continue_on_error:
                    break

                continue

            total_records += record_count

            print(
                f"    Wrote {record_count} records to {meta_out}"
            )

            if not args.run:
                completed_folders += 1
                continue

            return_code = run_subfigure(
                subfigure_script=subfigure_script,
                checkpoint=checkpoint,
                pmc_root=pmc_root,
                meta_out=meta_out,
                save_dir=save_dir,
                rcd_file=rcd_file,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                gpu=args.gpu,
                score_threshold=args.score_threshold,
                nms_threshold=args.nms_threshold,
            )

            if return_code == 0:
                completed_folders += 1
                print(f"    Completed: {png_dir}")
            else:
                print(
                    f"ERROR: subfigure.py failed for {png_dir} "
                    f"with return code {return_code}",
                    file=sys.stderr,
                )

                failed_folders.append(
                    {
                        "input_directory": str(png_dir),
                        "metadata_file": str(meta_out),
                        "return_code": return_code,
                        "reason": "subfigure.py failed",
                    }
                )

                if not args.continue_on_error:
                    break

        if failed_folders and not args.continue_on_error:
            break

    if bad_images:
        bad_images_file = output_root / "bad_images.jsonl"
        bad_images_file.write_text(
            "\n".join(
                json.dumps(item, ensure_ascii=False)
                for item in bad_images
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nBad-image log: {bad_images_file}")

    if failed_folders:
        failures_file = output_root / "failed_folders.jsonl"
        failures_file.write_text(
            "\n".join(
                json.dumps(item, ensure_ascii=False)
                for item in failed_folders
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Failure log: {failures_file}")

    print("\nProcessing summary")
    print(f"  Discovered folders: {total_folders}")
    print(f"  Completed folders:  {completed_folders}")
    print(f"  Skipped folders:    {skipped_folders}")
    print(f"  Failed folders:     {len(failed_folders)}")
    print(f"  Metadata records:   {total_records}")
    print(f"  Unreadable images:  {len(bad_images)}")

    return 1 if failed_folders else 0


if __name__ == "__main__":
    raise SystemExit(main())