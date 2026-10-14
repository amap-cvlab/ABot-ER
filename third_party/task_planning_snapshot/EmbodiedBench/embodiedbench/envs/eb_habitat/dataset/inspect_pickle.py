#!/usr/bin/env python3
"""Normalize bulk-move instructions in an EmbodiedBench habitat pickle."""

import argparse
import os
import pickle
import re
from typing import Any


SOURCE_LOCATION_PATTERNS = [
    re.compile(
        r"(?P<prefix>\bPut all the .+?) from the (?P<src>.+?) on the (?P<dst>.+?)(?P<end>[.?!])$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<prefix>\bPlace all the .+?) from the (?P<src>.+?) onto the (?P<dst>.+?)(?P<end>[.?!])$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<prefix>\bTransfer all .+?) on the (?P<src>.+?) to the (?P<dst>.+?)(?P<end>[.?!])$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<prefix>\bGather up all .+?) from the (?P<src>.+?) and set them on the (?P<dst>.+?)(?P<end>[.?!])$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<prefix>\bRelocate every .+?) from (?P<src>.+?) onto the (?P<dst>.+?)(?P<end>[.?!])$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<prefix>\bMove all the .+?) from the (?P<src>.+?) to the (?P<dst>.+?)(?P<end>[.?!])$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<prefix>\bPlace all the .+?) found on the (?P<src>.+?) onto the (?P<dst>.+?)(?P<end>[.?!])$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<prefix>\bTransport all .+?) on the (?P<src>.+?) and put them on the (?P<dst>.+?)(?P<end>[.?!])$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<prefix>\bShift all the .+?) on the (?P<src>.+?) over to (?P<dst>.+?)(?P<end>[.?!])$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<prefix>\bDeposit all the .+?) from the (?P<src>.+?) onto (?P<dst>.+?)(?P<end>[.?!])$",
        re.IGNORECASE,
    ),
]


def load_dataset(path: str) -> dict[str, Any]:
    with open(path, "rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict) or "all_eps" not in data:
        raise ValueError(f"{path} is not an EmbodiedBench habitat dataset pickle.")

    episodes = data["all_eps"]
    if not isinstance(episodes, list):
        raise ValueError(f"{path} has an unexpected all_eps structure.")
    return data


def normalize_instruction(instruction: str) -> str:
    for pattern in SOURCE_LOCATION_PATTERNS:
        match = pattern.match(instruction.strip())
        if not match:
            continue

        prefix = match.group("prefix").strip()
        dst = match.group("dst").strip()
        end = match.group("end")

        replacement_suffix = {
            0: f" on the {dst}{end}",
            1: f" onto the {dst}{end}",
            2: f" to the {dst}{end}",
            3: f" and set them on the {dst}{end}",
            4: f" onto the {dst}{end}",
            5: f" to the {dst}{end}",
            6: f" onto the {dst}{end}",
            7: f" and put them on the {dst}{end}",
            8: f" over to {dst}{end}",
            9: f" onto {dst}{end}",
        }
        cleaned = prefix + replacement_suffix[SOURCE_LOCATION_PATTERNS.index(pattern)]
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
        return cleaned

    return re.sub(r"\s+", " ", instruction).strip()


def update_episodes(
    episodes: list[dict[str, Any]], keyword: str, limit: int
) -> tuple[int, int]:
    match_count = 0
    changed_count = 0

    for index, episode in enumerate(episodes[:limit]):
        instruction = str(episode.get("instruction", ""))
        if keyword not in instruction:
            continue

        match_count += 1
        instruct_id = episode.get("instruct_id", "<missing>")
        cleaned = normalize_instruction(instruction)

        print(f"[{index}] instruct_id={instruct_id}")
        print(f"  original: {instruction}")
        print(f"  cleaned : {cleaned}")
        if cleaned != instruction:
            episode["instruction"] = cleaned
            changed_count += 1

    print(
        f"\nmatched {match_count} instructions in the first {min(limit, len(episodes))} episodes."
    )
    print(f"updated {changed_count} instructions.")
    return match_count, changed_count


def default_output_path(path: str) -> str:
    root, ext = os.path.splitext(path)
    if not ext:
        ext = ".pickle"
    if root.endswith("_old"):
        return f"{root[:-4]}{ext}"
    return f"{root}_cleaned{ext}"


def save_dataset(data: dict[str, Any], output_path: str) -> None:
    with open(output_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find and normalize instructions in an EmbodiedBench habitat pickle."
    )
    parser.add_argument("path", help="Path to the habitat pickle file.")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Only inspect the first N episodes. Default: 50.",
    )
    parser.add_argument(
        "--keyword",
        default=" all ",
        help="Substring used to filter instructions. Default: ' all '.",
    )
    parser.add_argument(
        "--output",
        help="Path to save the cleaned pickle. Default: <input>_cleaned.pickle, or remove the _old suffix if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset(args.path)
    episodes = data["all_eps"]
    _, changed_count = update_episodes(episodes, args.keyword, max(1, args.limit))
    output_path = args.output or default_output_path(args.path)
    save_dataset(data, output_path)
    print(f"saved cleaned pickle to: {output_path}")
    if changed_count == 0:
        print("warning: no instruction text was changed, but the file was still saved.")


if __name__ == "__main__":
    main()
