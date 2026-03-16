"""Annotate sequence-level mutations in weizmann-style CSV files.

This script compares each sequence against a designated wild-type ID and emits:
- A compact mutation label (e.g., L:S31N;H:G52T)
- Exploded mutation columns (mut1_chain, mut1_pos, mut1_from, mut1_to, ...)
- A computed mutation count and consistency check versus the existing Mutations column
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class MutationEvent:
    chain: str
    position: int
    from_res: str
    to_res: str

    def compact(self) -> str:
        return f"{self.chain}:{self.from_res}{self.position}{self.to_res}"


def _normalize_id(id_value: str) -> str:
    """Normalize IDs to compare plain IDs and *_pdb IDs equivalently."""
    if id_value.endswith("_pdb"):
        return id_value[:-4]
    return id_value


def _split_sequence(seq: str) -> Tuple[str, str]:
    """Split sequence into light and heavy chains using the single underscore delimiter."""
    parts = seq.split("_")
    if len(parts) != 2:
        raise ValueError(f"Sequence must contain exactly one '_' delimiter. Found: {seq[:80]}...")
    return parts[0], parts[1]


def _diff_chain(wt_chain: str, mut_chain: str, chain_code: str) -> List[MutationEvent]:
    """Return substitution events relative to wild type for one chain."""
    if len(wt_chain) != len(mut_chain):
        raise ValueError(
            f"Length mismatch for chain {chain_code}: WT={len(wt_chain)} vs mutant={len(mut_chain)}"
        )

    events: List[MutationEvent] = []
    for idx, (wt_res, mut_res) in enumerate(zip(wt_chain, mut_chain), start=1):
        if wt_res != mut_res:
            events.append(
                MutationEvent(
                    chain=chain_code,
                    position=idx,
                    from_res=wt_res,
                    to_res=mut_res,
                )
            )
    return events


def _events_for_sequence(seq: str, wt_light: str, wt_heavy: str) -> List[MutationEvent]:
    light, heavy = _split_sequence(seq)
    return _diff_chain(wt_light, light, "L") + _diff_chain(wt_heavy, heavy, "H")


def annotate_mutations(
    df: pd.DataFrame,
    wildtype_id: str,
) -> pd.DataFrame:
    required_cols = {"ID", "Sequence", "Mutations"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    id_norm = df["ID"].astype(str).map(_normalize_id)
    wt_mask = id_norm == _normalize_id(wildtype_id)
    wt_count = int(wt_mask.sum())
    if wt_count != 1:
        raise ValueError(
            "Expected exactly one wild-type row matching ID "
            f"'{wildtype_id}'. Found {wt_count}."
        )

    wt_seq = str(df.loc[wt_mask, "Sequence"].iloc[0])
    wt_light, wt_heavy = _split_sequence(wt_seq)

    unique_sequences = df["Sequence"].astype(str).unique()
    cache: Dict[str, List[MutationEvent]] = {}
    for seq in unique_sequences:
        cache[seq] = _events_for_sequence(seq, wt_light, wt_heavy)

    df_out = df.copy()

    events_per_row = [cache[str(seq)] for seq in df_out["Sequence"].astype(str)]
    compact_labels = [";".join(ev.compact() for ev in events) for events in events_per_row]
    computed_counts = [len(events) for events in events_per_row]

    df_out["mutation_label"] = compact_labels
    df_out["mutation_count_computed"] = computed_counts

    mutations_numeric = pd.to_numeric(df_out["Mutations"], errors="coerce")
    df_out["mutation_count_matches"] = mutations_numeric.eq(df_out["mutation_count_computed"])

    max_events = max(computed_counts) if computed_counts else 0
    for i in range(1, max_events + 1):
        df_out[f"mut{i}_chain"] = ""
        df_out[f"mut{i}_pos"] = pd.NA
        df_out[f"mut{i}_from"] = ""
        df_out[f"mut{i}_to"] = ""

    for row_idx, events in enumerate(events_per_row):
        for i, ev in enumerate(events, start=1):
            df_out.at[row_idx, f"mut{i}_chain"] = ev.chain
            df_out.at[row_idx, f"mut{i}_pos"] = ev.position
            df_out.at[row_idx, f"mut{i}_from"] = ev.from_res
            df_out.at[row_idx, f"mut{i}_to"] = ev.to_res

    return df_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate mutation identities for weizmann-style CSV files.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="weizmann.csv",
        help="Input CSV path (must contain ID, Sequence, Mutations columns).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="weizmann_with_mutation_annotations.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--wildtype_id",
        type=str,
        default="010101010101010101",
        help="Wild-type ID (plain or *_pdb form accepted).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_csv, dtype={"ID": str, "Sequence": str})
    df_out = annotate_mutations(df=df, wildtype_id=args.wildtype_id)

    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df_out.to_csv(args.output_csv, index=False)

    mismatch_count = int((~df_out["mutation_count_matches"]).sum())
    print(f"Rows processed: {len(df_out)}")
    print(f"Max computed mutations in a row: {int(df_out['mutation_count_computed'].max())}")
    print(f"Rows with mismatch against Mutations column: {mismatch_count}")


if __name__ == "__main__":
    main()
