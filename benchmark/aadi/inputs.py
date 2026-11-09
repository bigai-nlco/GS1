"""Pinned public inputs for the Artificial Analysis Decision Index adaptation."""

from pathlib import Path

from gs1.model.util import file_hash

METHODOLOGY = "https://artificialanalysis.ai/methodology/decision-index"
DATA_DIR = Path("datasets/aadi-v1")
CONVERSION = "gs1.aadi.public"
SOURCE = {
    "aa_omniscience": {
        "repo": "ArtificialAnalysis/AA-Omniscience-Public",
        "revision": "e4883edbb9f5ccf2b2a8fdc6fb65e01a58e99849",
        "file": "AA-Omniscience_dataset_public.csv",
        "sha256": "1e04603dafa3bd0d16d8151f07a5eb74c43d90c3a85d2fca120da2359174d02f",
        "split": "public",
    },
    "hle_multichoice": {
        "repo": "cais/hle",
        "revision": "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29",
        "file": "data/test-00000-of-00001.parquet",
        "sha256": "6d0ee0602e8aea6b159509577e884f48ecac7b8e3f6822a35f51335a446c726a",
        "split": "test",
    },
    "when2call": {
        "repo": "nvidia/When2Call",
        "revision": "0582f7749df63a96fdc3070932e83e72396ace53",
        "file": "test/when2call_test_mcq.jsonl",
        "sha256": "8c3694e583eeeb8dbc297e6cd90da70efc68efa4b6adb7227523e828c6b7b14c",
        "split": "test",
    },
    "wildguard_test": {
        "repo": "allenai/wildguardmix",
        "revision": "d29c47f41c8b51348b5c8e8c81c039b3132b66d1",
        "file": "test/wildguard_test.parquet",
        "sha256": "6ccc2909c1ae6d41424fac69f1fc32535b1de39cb8f80407d81e8bc64a0bebca",
        "split": "test",
    },
}

FROZEN = {
    "aa_omniscience": {
        "records": 600,
        "sha256": "fc566110f5b4bea6566f4bb3d3f82bbfc51d9b38daaef9e2c06b279358584b14",
        "official_subset_exact": False,
    },
    "hle_multichoice": {
        "records": 513,
        "sha256": "3452fd77d0148496e3b2ab3ca741cd6e8d8b695621d09ec91fe95f54770543bc",
        "official_subset_exact": True,
    },
    "wildguard_test": {
        "records": 856,
        "sha256": "1bf43d721bf4ca62ff2bc3d3a47dae153f4bfe0d866a5bd00978a52d21c01626",
        "official_subset_exact": False,
        "unanimous_rows": 880,
        "excluded_source_rows": (
            46,
            47,
            48,
            54,
            58,
            65,
            71,
            72,
            75,
            80,
            83,
            84,
            85,
            90,
            98,
            202,
            238,
            265,
            334,
            393,
            395,
            398,
            893,
            894,
        ),
        "refusal_probe_model": "gpt-6-luna",
        "refusal_probe_audit_sha256": "bb9fcf61f0aad5e13b78f63148fc56853d43a36538ea9443e4c850bbe58866f9",
    },
    "when2call": {
        "records": 3423,
        "sha256": "85f543b464bf043816161413fee9f8439a2958a9efac5eb6cea1d91abf6ae9cd",
        "official_subset_exact": True,
    },
}


def frozen_dataset(name, data_dir=DATA_DIR):
    """Validate one final bundled test file without preparing or downloading data."""
    spec = FROZEN[name]
    path = Path(data_dir) / name / "test.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"AADI final test file is missing: {path}")
    if file_hash(path) != spec["sha256"]:
        raise ValueError(f"AADI final test checksum mismatch: {path}")
    with path.open(encoding="utf-8") as stream:
        count = sum(bool(line.strip()) for line in stream)
    if count != spec["records"]:
        raise ValueError(f"AADI final test record count mismatch: {path}")
    return {
        "origin": "official_huggingface_test",
        "methodology": METHODOLOGY,
        "source": SOURCE[name],
        "source_sha256": SOURCE[name]["sha256"],
        "test_sha256": spec["sha256"],
        "counts": {"test_records": count},
        "official_subset_exact": spec["official_subset_exact"],
        **({k: v for k, v in spec.items() if k not in {"records", "sha256", "official_subset_exact"}}),
    }
