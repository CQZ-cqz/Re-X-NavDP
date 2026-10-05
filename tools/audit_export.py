"""Read-only export integrity, Python syntax and repository-internal symlink audit."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, help="optional original x-navdp directory; only read, never modify")
    p.add_argument("--check-export-hashes", action="store_true",
                   help="verify initial relocated contents; expected to differ after intentional new development")
    args = p.parse_args()
    manifest = json.loads((ROOT/"docs/source_manifest.json").read_text())
    for row in manifest["files"]:
        destination = ROOT/row["destination"]
        if not destination.is_file():
            raise ValueError(f"missing export: {destination}")
        if args.source and digest(args.source/row["source"]) != row["source_sha256"]:
            raise ValueError(f"original source changed: {row['source']}")
        if args.check_export_hashes and digest(destination) != row["export_sha256"]:
            raise ValueError(f"export changed: {row['destination']}")
    links, python_files = 0, 0
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            links += 1
            if not path.exists() or not path.resolve().is_relative_to(ROOT):
                raise ValueError(f"broken or external symlink: {path}")
        elif path.is_file() and path.suffix == ".py":
            ast.parse(path.read_bytes(), filename=str(path))
            python_files += 1
    print(json.dumps(dict(source_files=len(manifest["files"]), python_syntax_checked=python_files,
                          internal_links=links, original_checked=bool(args.source), status="PASS"), indent=2))


if __name__ == "__main__":
    main()
