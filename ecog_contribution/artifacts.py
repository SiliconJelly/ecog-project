"""Artifact validation and provenance; shared by CLI and read-only dashboard."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT=ROOT/"results/contribution_suite"


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def code_hash():
    h=hashlib.sha256()
    for p in sorted((ROOT/"ecog_contribution").glob("*.py")):
        h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()


def validate_cache(output=DEFAULT_OUTPUT, data=None):
    output=Path(output)
    try:
        manifest=json.loads((output/"manifest.json").read_text())
        if manifest.get("schema_version")!=1 or manifest["code_sha256"]!=code_hash():
            return False,"Analysis code changed; regenerate results."
        data=Path(data or manifest["data_path"])
        if not data.exists() or sha256(data)!=manifest["data_sha256"]:
            return False,"Source recording is missing or changed."
        for name,digest in manifest.get("protected_source_hashes",{}).items():
            source=ROOT/name
            if not source.exists() or sha256(source)!=digest:
                return False,f"Protected reproduction input is missing or changed: {name}"
        for name,digest in manifest["files"].items():
            p=output/name
            if not p.exists() or sha256(p)!=digest:
                return False,f"Missing or changed artifact: {name}"
        return True,"Verified results"
    except (OSError,ValueError,KeyError):
        return False,"Results are missing or incomplete. Run python -m ecog_contribution run."


def cache_fingerprint(output=DEFAULT_OUTPUT):
    """Invalidate UI caching when any source or artifact changes on disk."""
    output=Path(output)
    try:
        manifest=json.loads((output/"manifest.json").read_text())
        paths=[output/"manifest.json",Path(manifest["data_path"])]
        paths.extend(ROOT/name for name in manifest.get("protected_source_hashes",{}))
        paths.extend(output/name for name in manifest["files"])
        return tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in paths)
    except (OSError,ValueError,KeyError):
        return ()
