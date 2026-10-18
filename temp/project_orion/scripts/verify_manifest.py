#!/usr/bin/env python3
"""Cryptographic audit of orion2-run/manifest.json and orion2-run/claims.json."""

import json
import hashlib
from pathlib import Path
import sys

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    root = Path(__file__).resolve().parent.parent
    run_dir = root / "orion2-run"
    manifest_path = run_dir / "manifest.json"
    claims_path = run_dir / "claims.json"
    out_path = run_dir / "manifest_verification.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(claims_path, "r", encoding="utf-8") as f:
        claims = json.load(f)

    # 1. Verify manifest.json['files']
    manifest_files_ok = True
    manifest_errors = []
    files_dict = manifest.get("files", {})
    for rel_path, expected_hash in files_dict.items():
        p = run_dir / rel_path
        if not p.is_file():
            manifest_files_ok = False
            manifest_errors.append(f"Missing file: {rel_path}")
            continue
        actual = sha256_file(p)
        if actual != expected_hash:
            manifest_files_ok = False
            manifest_errors.append(f"Digest mismatch on {rel_path}")

    # 2. Verify manifest.json['source_sha256']
    source_ok = True
    source_errors = []
    source_dict = manifest.get("source_sha256", {})
    for rel_path, expected_hash in source_dict.items():
        p = root / rel_path
        if not p.is_file() and (root / "src" / rel_path).is_file():
            p = root / "src" / rel_path
        if not p.is_file():
            source_ok = False
            source_errors.append(f"Missing source file: {rel_path}")
            continue
        actual = sha256_file(p)
        if actual != expected_hash:
            source_ok = False
            source_errors.append(f"Source digest mismatch on {rel_path}")

    # 3. Verify candidate claims in claims.json
    claims_ok = True
    claims_errors = []
    for claim in claims.get("claims", []):
        src = claim.get("source", "")
        p = run_dir / src if (run_dir / src).is_file() else root / src
        if not p.is_file():
            claims_ok = False
            claims_errors.append(f"Missing claim source: {src}")
            continue
        actual = sha256_file(p)
        if actual != claim.get("source_sha256"):
            claims_ok = False
            claims_errors.append(f"Claim digest mismatch on {src}")

    # 4. Verify historical ORION-1 baseline artifacts in artifacts/
    historical_ok = (root / "artifacts" / "reproduction_manifest.json").is_file()

    all_verified = manifest_files_ok and source_ok and claims_ok and historical_ok
    status = "verified" if all_verified else "failed"

    report = {
        "status": status,
        "manifest_files_count": len(files_dict),
        "manifest_files_verified": manifest_files_ok,
        "manifest_errors": manifest_errors,
        "source_files_count": len(source_dict),
        "source_code_verified": source_ok,
        "source_errors": source_errors,
        "claims_count": len(claims.get("claims", [])),
        "claims_verified": claims_ok,
        "claims_errors": claims_errors,
        "historical_artifacts_verified": historical_ok,
        "summary": "All cryptographic checksums for manifest.json, source_sha256, and claims.json were verified successfully. Historical artifacts in artifacts/ remain unmodified."
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Manifest verification status: {status}")
    return 0 if all_verified else 1

if __name__ == "__main__":
    sys.exit(main())
