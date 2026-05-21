"""
upload_github.py — automated Git commit + push, skipping files > 50 MB.

Usage:
    python upload_github.py                        # commit all eligible changes
    python upload_github.py --message "feat: ..."  # custom commit message
    python upload_github.py --remote origin --branch master
    python upload_github.py --dry-run              # show what would be added

Files larger than MAX_MB are automatically excluded from staging.
.gitignore rules are respected; this script adds an extra size guard.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

MAX_MB = 50
MAX_BYTES = MAX_MB * 1024 * 1024

REPO_ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, check=check, cwd=REPO_ROOT,
        capture_output=capture, text=True,
    )


def _run_out(cmd: list[str]) -> str:
    result = _run(cmd, capture=True)
    return result.stdout.strip()


def get_untracked_and_modified() -> list[Path]:
    """Return all untracked + modified files known to git status."""
    out = _run_out(["git", "status", "--porcelain"])
    paths = []
    for line in out.splitlines():
        line = line.rstrip("\r")
        if not line.strip():
            continue
        # Format: "XY PATH" where XY is two status chars and PATH follows a space.
        # Split on the first run of whitespace to skip the status code robustly
        # (leading space may be absent on some platforms/git versions).
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        filepath = parts[1].strip()
        # Handle renamed files: "old -> new"
        if " -> " in filepath:
            filepath = filepath.split(" -> ")[-1].strip()
        paths.append(REPO_ROOT / filepath)
    return paths


def filter_by_size(paths: list[Path], max_bytes: int, verbose: bool = True) -> tuple[list[Path], list[Path]]:
    """Split paths into (eligible, skipped_too_large)."""
    eligible, skipped = [], []
    for p in paths:
        if not p.exists():
            eligible.append(p)   # deleted files — git handles them fine
            continue
        try:
            size = p.stat().st_size
        except OSError:
            eligible.append(p)
            continue
        if size > max_bytes:
            skipped.append(p)
            if verbose:
                mb = size / (1024 * 1024)
                print(f"[skip >50 MB] {p.relative_to(REPO_ROOT)}  ({mb:.1f} MB)")
        else:
            eligible.append(p)
    return eligible, skipped


def stage_files(paths: list[Path], dry_run: bool = False) -> int:
    """git add each eligible path; return count of actually staged files."""
    if not paths:
        return 0
    rel_paths = [str(p.relative_to(REPO_ROOT)) for p in paths]
    if dry_run:
        print("[dry-run] would stage:")
        for rp in rel_paths:
            print(f"  {rp}")
        return len(rel_paths)
    # Stage in batches of 200 to stay under ARG_MAX
    batch_size = 200
    for i in range(0, len(rel_paths), batch_size):
        _run(["git", "add", "--"] + rel_paths[i : i + batch_size])
    return len(rel_paths)


def has_staged_changes() -> bool:
    out = _run_out(["git", "diff", "--cached", "--name-only"])
    return bool(out.strip())


def commit(message: str, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[dry-run] would commit: {message!r}")
        return
    _run(["git", "commit", "-m", message])


def push(remote: str, branch: str, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[dry-run] would push to {remote}/{branch}")
        return
    print(f"[push] {remote} {branch}")
    _run(["git", "push", remote, branch])


def write_gitignore_hint(skipped: list[Path]) -> None:
    """Suggest .gitignore entries for oversized files."""
    if not skipped:
        return
    print("\n[hint] Add these to .gitignore to suppress future warnings:")
    for p in skipped:
        rel = p.relative_to(REPO_ROOT).as_posix()
        print(f"  {rel}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Git commit + push, skipping files > 50 MB")
    ap.add_argument("--message", "-m", default=None,
                    help="Commit message (auto-generated if omitted)")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--branch", default=None,
                    help="Branch to push (default: current branch)")
    ap.add_argument("--no-push", action="store_true",
                    help="Commit but do not push")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would happen without making any changes")
    ap.add_argument("--max-mb", type=float, default=MAX_MB,
                    help=f"Size threshold in MB (default: {MAX_MB})")
    args = ap.parse_args()

    max_bytes = int(args.max_mb * 1024 * 1024)

    # Detect current branch
    branch = args.branch or _run_out(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    print(f"[branch] {branch}")

    # Collect changed files
    changed = get_untracked_and_modified()
    if not changed:
        print("[info] Nothing to commit (working tree clean).")
        return

    eligible, skipped = filter_by_size(changed, max_bytes, verbose=True)
    write_gitignore_hint(skipped)

    staged = stage_files(eligible, dry_run=args.dry_run)
    if staged == 0:
        print("[info] No eligible files to stage.")
        return

    if not args.dry_run and not has_staged_changes():
        print("[info] Nothing staged — all changes may already be committed.")
        return

    # Auto-generate commit message if not provided
    message = args.message
    if not message:
        n_files = staged
        n_skipped = len(skipped)
        suffix = f" (skipped {n_skipped} file(s) > {args.max_mb:.0f} MB)" if n_skipped else ""
        message = f"chore: update {n_files} file(s){suffix}"
    print(f"[commit] {message!r}")

    commit(message, dry_run=args.dry_run)

    if not args.no_push:
        push(args.remote, branch, dry_run=args.dry_run)
        print("[done] pushed successfully.")
    else:
        print("[done] committed locally (--no-push).")


if __name__ == "__main__":
    main()
