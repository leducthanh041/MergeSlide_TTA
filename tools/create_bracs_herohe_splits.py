#!/usr/bin/env python3
"""Create ten CLAM-format splits for BRACS and HEROHE."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


BRACS_LABELS = {"Group_BT": 0, "Group_AT": 1, "Group_MT": 2}
HEROHE_LABELS = {"Negative": 0, "Positive": 1}
SPLIT_NAMES = ("train", "val", "test")


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _score_assignment(
    counts: np.ndarray,
    class_counts: np.ndarray,
    target_counts: np.ndarray,
    target_classes: np.ndarray,
) -> float:
    count_scale = np.maximum(target_counts, 1)
    class_scale = np.maximum(target_classes, 1)
    return float(
        np.square((counts - target_counts) / count_scale).sum()
        + 2.0 * np.square((class_counts - target_classes) / class_scale).sum()
    )


def grouped_stratified_split(
    df: pd.DataFrame,
    group_col: str,
    target_counts: list[int],
    target_classes: list[list[int]],
    seed: int,
    attempts: int,
    fixed_assignment: dict[Hashable, int] | None = None,
    free_split_ids: tuple[int, ...] = (0, 1, 2),
    progress_desc: str | None = None,
) -> Split:
    """Approximate CLAM's stratified split while keeping groups indivisible."""
    n_classes = len(target_classes[0])
    grouped = []
    for group, part in df.groupby(group_col, sort=False):
        class_vector = np.bincount(part["label_id"], minlength=n_classes)
        grouped.append((group, len(part), class_vector))

    rng = np.random.default_rng(seed)
    target_n = np.asarray(target_counts, dtype=float)
    target_y = np.asarray(target_classes, dtype=float)
    best_score = float("inf")
    best_assignment: dict[Hashable, int] | None = None
    fixed_assignment = fixed_assignment or {}

    candidate_range = tqdm(
        range(attempts),
        desc=progress_desc or f"Optimize {group_col}",
        unit="candidate",
        dynamic_ncols=True,
    )
    for _ in candidate_range:
        counts = np.zeros(3, dtype=float)
        classes = np.zeros((3, n_classes), dtype=float)
        assignment: dict[Hashable, int] = dict(fixed_assignment)

        for group, size, class_vector in grouped:
            if group in fixed_assignment:
                split_id = fixed_assignment[group]
                counts[split_id] += size
                classes[split_id] += class_vector

        free_indices = [i for i, item in enumerate(grouped) if item[0] not in fixed_assignment]
        order = rng.permutation(free_indices)

        # Large groups are assigned early; random jitter creates distinct folds.
        order = sorted(
            order,
            key=lambda i: grouped[i][1] + rng.uniform(0, max(2, grouped[i][1] * 0.2)),
            reverse=True,
        )
        for idx in order:
            group, size, class_vector = grouped[idx]
            choices = []
            for split_id in free_split_ids:
                next_counts = counts.copy()
                next_classes = classes.copy()
                next_counts[split_id] += size
                next_classes[split_id] += class_vector
                choices.append(
                    _score_assignment(next_counts, next_classes, target_n, target_y)
                    + rng.uniform(0, 1e-6)
                )
            split_id = int(np.argmin(choices))
            assignment[group] = split_id
            counts[split_id] += size
            classes[split_id] += class_vector

        # Reject partitions missing a class.
        if np.any(classes == 0):
            continue
        score = _score_assignment(counts, classes, target_n, target_y)
        if score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError(f"Could not create a valid grouped split for {group_col}")

    parts = []
    for split_id in range(3):
        groups = {g for g, assigned in best_assignment.items() if assigned == split_id}
        parts.append(df[df[group_col].isin(groups)].copy())
    return Split(*parts)


def make_bracs_splits(
    df: pd.DataFrame,
    seed: int,
    attempts: int,
    setting: str,
    match_overall_distribution: bool,
) -> list[Split]:
    data = df.copy()
    data["label_id"] = data["label"].map(BRACS_LABELS)
    if data["label_id"].isna().any():
        raise ValueError("BRACS contains unknown labels")
    data["label_id"] = data["label_id"].astype(int)

    target_counts = [394, 65, 87]
    if match_overall_distribution:
        target_classes = [
            [191, 64, 139],
            [31, 11, 23],
            [42, 14, 31],
        ]
    else:
        target_classes = [
            [202, 52, 140],
            [30, 14, 21],
            [32, 23, 32],
        ]
    return [
        grouped_stratified_split(
            data,
            group_col="case_id",
            target_counts=target_counts,
            target_classes=target_classes,
            seed=seed + fold,
            attempts=attempts,
            progress_desc=f"BRACS {setting} fold {fold}",
        )
        for fold in range(10)
    ]


def make_herohe_ind_splits(
    df: pd.DataFrame,
    seed: int,
    attempts: int,
) -> list[Split]:
    data = df.copy()
    data["label_id"] = data["label"].map(HEROHE_LABELS)
    if data["label_id"].isna().any():
        raise ValueError("HEROHE contains unknown labels")
    data["label_id"] = data["label_id"].astype(int)
    data["source"] = data["slide_id"].str.extract(r"^(train|test)_", expand=False)
    data["group_id"] = data["source"] + "_" + data["case_id"].astype(str)

    if len(data) != 508 or data["label_id"].value_counts().to_dict() != {0: 306, 1: 202}:
        raise ValueError("Unexpected HEROHE dataset composition")

    return [
        grouped_stratified_split(
            data,
            group_col="group_id",
            target_counts=[358, 65, 85],
            target_classes=[
                [216, 142],
                [39, 26],
                [51, 34],
            ],
            seed=seed + fold,
            attempts=attempts,
            progress_desc=f"HEROHE IND fold {fold}",
        )
        for fold in range(10)
    ]


def add_herohe_laboratory(annotation: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    metadata["slide_id"] = metadata["new_filename"].astype(str).str.removesuffix(".pt")
    labs = metadata[["slide_id", "Laboratory"]].drop_duplicates("slide_id")
    data = annotation.merge(labs, on="slide_id", how="left", validate="one_to_one")
    if data["Laboratory"].isna().any():
        missing = data.loc[data["Laboratory"].isna(), "slide_id"].tolist()
        raise ValueError(f"Missing HEROHE laboratory metadata: {missing[:5]}")
    data["domain_id"] = "lab_" + data["Laboratory"].astype(int).astype(str)
    data["label_id"] = data["label"].map(HEROHE_LABELS).astype(int)
    return data


def make_herohe_ood_splits(
    df: pd.DataFrame,
    metadata_path: Path,
    seed: int,
    attempts: int,
) -> list[Split]:
    data = add_herohe_laboratory(df, metadata_path)
    overall = np.bincount(data["label_id"], minlength=2) / len(data)
    target_counts = np.asarray([358, 65, 85])
    target_classes = np.rint(target_counts[:, None] * overall[None, :]).astype(int)
    domains = sorted(data["domain_id"].unique())
    rng = np.random.default_rng(seed + 1000)
    rng.shuffle(domains)
    n_test_domains = max(1, round(len(domains) * 0.17))

    splits = []
    for fold in range(10):
        start = (fold * n_test_domains) % len(domains)
        test_domains = {
            domains[(start + offset) % len(domains)]
            for offset in range(n_test_domains)
        }
        splits.append(
        grouped_stratified_split(
            data,
            group_col="domain_id",
            target_counts=target_counts.tolist(),
            target_classes=target_classes.tolist(),
            seed=seed + 1000 + fold,
            attempts=attempts,
            fixed_assignment={domain: 2 for domain in test_domains},
            free_split_ids=(0, 1),
            progress_desc=f"HEROHE OOD fold {fold}",
            )
        )
    return splits


def validate_split(split: Split, group_col: str, domain_col: str | None = None) -> None:
    slide_sets = [set(getattr(split, name)["slide_id"]) for name in SPLIT_NAMES]
    group_sets = [set(getattr(split, name)[group_col]) for name in SPLIT_NAMES]
    for i in range(3):
        for j in range(i + 1, 3):
            if slide_sets[i] & slide_sets[j]:
                raise AssertionError("Slide leakage detected")
            if group_sets[i] & group_sets[j]:
                raise AssertionError(f"{group_col} leakage detected")
    if domain_col:
        domain_sets = [set(getattr(split, name)[domain_col]) for name in SPLIT_NAMES]
        for i in range(3):
            for j in range(i + 1, 3):
                if domain_sets[i] & domain_sets[j]:
                    raise AssertionError(f"{domain_col} leakage detected")
    for name in SPLIT_NAMES:
        if getattr(split, name)["label_id"].nunique() < 2:
            raise AssertionError(f"{name} does not contain all required classes")


def clam_frame(split: Split) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for name in SPLIT_NAMES:
        part = getattr(split, name).reset_index(drop=True)
        columns[name] = part["slide_id"]
        columns[f"{name}_label"] = part["label_id"]
    return pd.DataFrame(columns)


def write_or_preview(
    splits: list[Split],
    output_dir: Path,
    dataset: str,
    group_col: str,
    domain_col: str | None,
    write: bool,
    overwrite: bool,
) -> None:
    audit_rows = []
    fold_signatures = []
    pending_writes = []
    for fold, split in enumerate(splits):
        validate_split(split, group_col=group_col, domain_col=domain_col)
        fold_signatures.append(
            tuple(
                tuple(sorted(getattr(split, name)["slide_id"].astype(str)))
                for name in SPLIT_NAMES
            )
        )
        for name in SPLIT_NAMES:
            part = getattr(split, name)
            counts = part["label_id"].value_counts().sort_index().to_dict()
            row = {
                "dataset": dataset,
                "fold": fold,
                "split": name,
                "n_slides": len(part),
                "n_groups": part[group_col].nunique(),
                "class_counts": str(counts),
            }
            if domain_col:
                row["n_domains"] = part[domain_col].nunique()
                row["domains"] = ",".join(sorted(part[domain_col].astype(str).unique()))
            audit_rows.append(row)

        path = output_dir / f"splits_{fold}.csv"
        if write:
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
            pending_writes.append((clam_frame(split), path))

    if len(set(fold_signatures)) != len(splits):
        raise AssertionError(
            f"{dataset} contains duplicate fold assignments: "
            f"{len(set(fold_signatures))}/{len(splits)} unique"
        )

    audit = pd.DataFrame(audit_rows)
    print(f"\n=== {dataset}: {output_dir} ===")
    print(audit[["fold", "split", "n_slides", "n_groups", "class_counts"]].to_string(index=False))
    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
        for frame, path in pending_writes:
            frame.to_csv(path)
        audit.to_csv(output_dir / "split_audit.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/dataset"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempts", type=int, default=3000)
    parser.add_argument(
        "--settings",
        choices=("all", "ind", "ood"),
        default="all",
        help="Generate only IND, only OOD, or both settings",
    )
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overwrite and not args.write:
        raise ValueError("--overwrite requires --write")

    ind_root = args.dataset_root / "wsi_dataset_annotation"
    ood_root = args.dataset_root / "wsi_dataset_annotation_cross_sites"
    herohe_metadata = args.dataset_root / "HEROHE" / "merged_groundtruth.csv"

    bracs_ind = pd.read_csv(ind_root / "bracs" / "bracs.csv")
    bracs_ood = pd.read_csv(ood_root / "bracs" / "bracs.csv")
    herohe_ind = pd.read_csv(ind_root / "herohe" / "herohe.csv")
    herohe_ood = pd.read_csv(ood_root / "herohe" / "herohe.csv")

    if args.settings in ("all", "ind"):
        write_or_preview(
            make_bracs_splits(
                bracs_ind,
                args.seed,
                args.attempts,
                "IND",
                match_overall_distribution=True,
            ),
            ind_root / "bracs",
            "BRACS IND",
            "case_id",
            None,
            args.write,
            args.overwrite,
        )
        write_or_preview(
            make_herohe_ind_splits(herohe_ind, args.seed, args.attempts),
            ind_root / "herohe",
            "HEROHE IND",
            "group_id",
            None,
            args.write,
            args.overwrite,
        )

    if args.settings in ("all", "ood"):
        print(
            "\n[WARN] BRACS OOD is a case-disjoint proxy because no "
            "acquisition-site metadata is available."
        )
        write_or_preview(
            make_bracs_splits(
                bracs_ood,
                args.seed + 1000,
                args.attempts,
                "OOD",
                match_overall_distribution=False,
            ),
            ood_root / "bracs",
            "BRACS OOD proxy",
            "case_id",
            None,
            args.write,
            args.overwrite,
        )
        write_or_preview(
            make_herohe_ood_splits(
                herohe_ood,
                herohe_metadata,
                args.seed,
                args.attempts,
            ),
            ood_root / "herohe",
            "HEROHE OOD",
            "domain_id",
            "domain_id",
            args.write,
            args.overwrite,
        )

    mode = "written" if args.write else "preview only; pass --write to save"
    print(f"\nDone: {mode}.")


if __name__ == "__main__":
    main()
