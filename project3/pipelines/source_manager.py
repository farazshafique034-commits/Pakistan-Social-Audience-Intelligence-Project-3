import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from project3.config.settings import BASE_DIR, PROCESSED_DIR, REPORT_DIR
from project3.analytics.data_validation import validate_posts, validate_comments


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_sources():
    manifest_path = BASE_DIR / "config" / "data_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run the dataset generator first")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {}

    for label, relative_path in manifest["files"].items():
        source = BASE_DIR.parent / relative_path
        if not source.exists():
            raise FileNotFoundError(f"Missing: {source}")

        entries[label] = {
            "path": str(source.relative_to(BASE_DIR.parent)),
            "size_bytes": source.stat().st_size,
            "sha256": file_hash(source)
        }

    report = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "declared_provenance": manifest.get("provenance"),
        "declared_summary": manifest.get("summary"),
        "files": entries
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "source_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("PROJECT 3 SOURCE AUDIT")
    print("Provenance:", report["declared_provenance"])
    print("Verified files:", len(entries))
    print("Source audit saved:", output)
    print("PHASE 2 PART 3 COMPLETED SUCCESSFULLY")


def import_dataset(args):
    posts_path = Path(args.posts)
    comments_path = Path(args.comments)

    posts = validate_posts(pd.read_csv(posts_path, encoding="utf-8-sig"))
    comments = validate_comments(
        pd.read_csv(comments_path, encoding="utf-8-sig"), posts
    )

    if not posts["data_origin"].eq(args.origin).all():
        raise ValueError("Post provenance does not match selected origin")

    if not comments["data_origin"].eq(args.origin).all():
        raise ValueError("Comment provenance does not match selected origin")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = PROCESSED_DIR / "imports" / stamp
    destination.mkdir(parents=True, exist_ok=False)

    posts.to_csv(destination / "posts.csv", index=False, encoding="utf-8-sig")
    comments.to_csv(destination / "comments.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "data_origin": args.origin,
        "source_name": args.source_name,
        "source_url": args.source_url,
        "permission_note": args.permission_note,
        "total_posts": len(posts),
        "total_comments": len(comments),
        "original_posts_sha256": file_hash(posts_path),
        "original_comments_sha256": file_hash(comments_path)
    }

    (destination / "provenance.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("Imported posts:", len(posts))
    print("Imported comments:", len(comments))
    print("Import folder:", destination)


def main():
    parser = argparse.ArgumentParser(description="Project 3 source management")
    parser.add_argument("--posts")
    parser.add_argument("--comments")
    parser.add_argument("--origin", choices=[
        "public_api", "licensed_dataset", "manual_import"
    ])
    parser.add_argument("--source-name")
    parser.add_argument("--source-url")
    parser.add_argument("--permission-note")
    args = parser.parse_args()

    if any([args.posts, args.comments, args.origin, args.source_name,
            args.source_url, args.permission_note]):
        required = [args.posts, args.comments, args.origin,
                    args.source_name, args.source_url, args.permission_note]
        if not all(required):
            parser.error("All six import arguments are required")
        import_dataset(args)
    else:
        audit_sources()


if __name__ == "__main__":
    main()
