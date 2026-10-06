from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECRET = re.compile(
    rb"https://hooks\.slack(?:-gov)?\.com/services/[A-Za-z0-9_-]{8,}/[A-Za-z0-9_-]{8,}/[A-Za-z0-9_-]{12,}"
    rb"|https://(?:\w+\.)?discord(?:app)?\.com/api/webhooks/[0-9]{12,}/[A-Za-z0-9_-]{20,}"
    rb"|\b(?:gh[pousr]_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9-]{25,})\b"
    rb"|^HOLDINGS\s*=\s*\[",
    re.MULTILINE,
)


def publication_paths(root: Path = ROOT) -> list[str]:
    contract = json.loads((root / "config/publication_allowlist.json").read_text())
    manifest_path = root / contract["bundle_manifest"]
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != contract["bundle_manifest_sha256"]:
        raise RuntimeError("bundle_manifest_hash_mismatch")
    manifest = json.loads(raw)
    paths = [*contract["files"], contract["bundle_manifest"]]
    for name, digest in manifest["file_sha256"].items():
        path = manifest_path.parent / name
        if not path.resolve().is_relative_to(manifest_path.parent.resolve()) or path.is_symlink():
            raise RuntimeError("bundle_path_rejected")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("bundle_file_hash_mismatch")
        paths.append(path.relative_to(root).as_posix())
    if len(paths) != len(set(paths)):
        raise RuntimeError("publication_allowlist_duplicates")
    return paths


def check(staged: bool = False, root: Path = ROOT) -> list[str]:
    allowed = publication_paths(root)
    tracked_result = subprocess.run(
        ["git", "--no-optional-locks", "ls-files", "-z"], cwd=root, capture_output=True, check=True,
    )
    tracked = {name.decode() for name in tracked_result.stdout.split(b"\0") if name}
    if tracked - set(allowed):
        raise RuntimeError("unapproved_tracked_publication_path")
    if staged:
        proc = subprocess.run(
            ["git", "--no-optional-locks", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"],
            cwd=root, capture_output=True, check=True,
        )
        paths = [name.decode() for name in proc.stdout.split(b"\0") if name]
    else:
        paths = allowed
    if set(paths) - set(allowed):
        raise RuntimeError("unapproved_publication_path")
    for name in paths:
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise RuntimeError("publication_symlink_or_external_path")
        if staged:
            data = subprocess.run(
                ["git", "show", f":{name}"], cwd=root, capture_output=True, check=True,
            ).stdout
        else:
            data = path.read_bytes()
        if SECRET.search(data):
            raise RuntimeError("credential_or_holdings_literal_rejected")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the explicit public publication boundary; does not stage files.")
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    paths = check(args.staged)
    print(json.dumps({"publication_check": "passed", "audited_files": len(paths), "staged": args.staged}))


if __name__ == "__main__":
    main()
