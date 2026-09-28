import json
from pathlib import Path

DATA_ROOT = Path("/nfs/turbo/umms-tocho-ns/data/2d_project/oa/subfigure_parsed")
MODALITIES = ("brain", "interventional_neuroradiology", "journal_of_neuroimaging", "journal_of_neurosurgery", "neuron", "neurosurgical_focus", "open_journal_of_neuroimaging", "stroke")


def find_one(folder: Path, pattern: str) -> Path | None:
    matches = sorted(folder.glob(pattern))
    if not matches:
        return None
    return matches[0]


def process_batch(folder: Path) -> tuple[str, dict[int, int]]:
    matched_path = folder / "matched.jsonl"
    unmatched_path = folder / "unmatched.jsonl"

    subcaptions_path = find_one(folder, "*_subcaption_openai_output.jsonl")
    subfigures_path = find_one(folder, "*_subfigures_with_letters_gpt-5.4-mini.jsonl")

    if subcaptions_path is None or subfigures_path is None:
        return "skipped_missing_inputs", {1: 0, 2: 0, 3: 0}

    # Step 1: Build a lookup dictionary from the subcaptions file
    # Example:
    # caption_lookup["3809434_F1"]["Subfigure-C"] gives the caption for Subfigure-C
    caption_lookup = {}
    full_caption_lookup = {}
    no_subcaption_figures = set()

    match_case_label_1=0
    match_case_label_2=0
    match_case_label_3=0

    with open(subcaptions_path, "r") as subcaptions:
        for c_line in subcaptions:
            subcaption_line = json.loads(c_line)

            figure_key = subcaption_line["figure_key"]
            subcaptions_dict = subcaption_line["subcaptions"]

            caption_lookup[figure_key] = subcaptions_dict
            full_caption_lookup[figure_key] = subcaption_line["caption"]
            if subcaption_line.get("llm_output") == "NO":
                no_subcaption_figures.add(figure_key)

    # Step 2: Loop through subfigures and append the matching subcaption
    with open(subfigures_path, "r") as subfigures, \
         open(matched_path, "w") as matched_out, \
         open(unmatched_path, "w") as unmatched_out:

        for f_line in subfigures:
            subfigure_line = json.loads(f_line)

            subfigure_source_id = subfigure_line["source_fig_id"]
            subfigure_letter = subfigure_line["letter"].upper()

            subcaption_key = f"Subfigure-{subfigure_letter}"

            if subfigure_source_id in no_subcaption_figures:
                subfigure_line["subcaption"] = full_caption_lookup[subfigure_source_id]
                subfigure_line["match_case_label"] = 3
                match_case_label_3 += 1
                matched_out.write(json.dumps(subfigure_line) + "\n")

            elif (
                subfigure_source_id in caption_lookup
                and subcaption_key in caption_lookup[subfigure_source_id]
            ):
                subcaption = caption_lookup[subfigure_source_id][subcaption_key]

                # This adds a new field to the subfigure JSON object
                subfigure_line["subcaption"] = subcaption
                subfigure_line["match_case_label"] = 1
                match_case_label_1 += 1
                matched_out.write(json.dumps(subfigure_line) + "\n")

            else:
                # Save unmatched cases separately so you can inspect them later
                subfigure_line["subcaption"] = full_caption_lookup.get(subfigure_source_id)
                subfigure_line["missing_subcaption_key"] = subcaption_key
                subfigure_line["match_case_label"] = 2
                match_case_label_2 += 1

                unmatched_out.write(json.dumps(subfigure_line) + "\n")

    return "processed", {
        1: match_case_label_1,
        2: match_case_label_2,
        3: match_case_label_3,
    }


def main() -> None:
    counts = {
        "processed": 0,
        "skipped_missing_inputs": 0,
    }
    label_totals = {1: 0, 2: 0, 3: 0}

    for modality in MODALITIES:
        modality_dir = DATA_ROOT / modality
        if not modality_dir.is_dir():
            print(f"Missing modality directory, skipping: {modality_dir}")
            continue

        for folder in sorted(p for p in modality_dir.iterdir() if p.is_dir()):
            status, label_counts = process_batch(folder)
            counts[status] += 1
            for label, n in label_counts.items():
                label_totals[label] += n
            print(f"{status}: {folder}")

    print(
        "Done. "
        f"processed={counts['processed']} "
        f"skipped_missing_inputs={counts['skipped_missing_inputs']}"
    )

    print(f"match case label 1 = {label_totals[1]}")
    print(f"match case label 2 = {label_totals[2]}")
    print(f"match case label 3 = {label_totals[3]}")


if __name__ == "__main__":
    main()
