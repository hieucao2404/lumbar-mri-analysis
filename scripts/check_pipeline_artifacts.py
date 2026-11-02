"""Verify the separately shared weights before running the paired pipeline."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("docs/handoff/paired_artifacts.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    failed = False
    for artifact in manifest["artifacts"]:
        path = args.root / artifact["path"]
        if not path.is_file():
            status = "missing"
        else:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            status = "ok" if digest == artifact["sha256"] else "hash_mismatch"
        print(f"{status}: {artifact['path']}")
        failed |= status != "ok"
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
