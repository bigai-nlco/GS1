"""Run a GS1 export on JevBench's pinned public tasks and scorer."""

import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
JEVBENCH_REVISION = "fcc69279a933f882ccc4489eb14bf467647dbdb7"  # v1.4.2.2
PUBLIC_FILES = {
    "original": "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180",
    "easy": "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b",
    "hard": "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb",
}


def public_tasks(cache_dir: Path, names: list[str], offline: bool):
    from jevbench.tasks import load_jsonl

    cache = cache_dir / JEVBENCH_REVISION
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = cache / f"{name}.jsonl"
        expected = PUBLIC_FILES[name]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            if offline:
                raise FileNotFoundError(f"Pinned JevBench dataset missing or corrupt: {path}")
            url = (
                "https://raw.githubusercontent.com/fstandhartinger/jevbench/"
                f"{JEVBENCH_REVISION}/datasets/public/{name}.jsonl"
            )
            print(f"Downloading {url}", file=sys.stderr, flush=True)
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"JevBench checksum mismatch for {name}")
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        paths.append(path)
    tasks = [task for path in paths for task in load_jsonl(str(path))]
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("JevBench public tasks contain duplicate IDs")
    return paths, tasks
