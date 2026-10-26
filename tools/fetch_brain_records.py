"""Recover and maintain Brain simulation records via the API.

Fetches every alpha visible under one or more Brain accounts from
``GET /users/self/alphas``, normalizes each into the master.jsonl record
schema, and appends to size-rotated JSONL shards that each stay under the
GitHub 100 MB file limit. Designed both as a one-time disaster recovery
(lost local logs) and as an ongoing running log: reruns resume from a
date cursor and dedupe by alpha id, so only new records are appended.

Output layout (committable; not covered by .gitignore):

    records/
      brain/
        alphas_part0001.jsonl   normalized records + raw_alpha_detail
        alphas_part0002.jsonl   (new shard opened before 95 MB by default)
        os_part0001.jsonl       yearly-stats + PnL for submitted alphas
        competition_IQC2026S1.json
        state.json              resume cursor + run metadata
      brain2/
        ...

Pagination: Brain caps offset on /users/self/alphas, so this pages in
ascending dateCreated order and, when the offset ceiling is reached,
restarts with a ``dateCreated>=<last seen>`` filter (duplicates at the
boundary are dropped via the seen-id set).

Usage:
    python tools/fetch_brain_records.py --all
    python tools/fetch_brain_records.py --account brain2 --credentials brain2.txt
    python tools/fetch_brain_records.py --all --merge-master
    python tools/fetch_brain_records.py --account brain --full   # rescan from t=0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.brain.api import (
    _request_json_with_retries,
    authenticate_session,
    fetch_pnl_series_with_retry,
    fetch_yearly_stats,
    is_auth_expired_response,
    sign_in,
)

ALPHAS_URL = "https://api.worldquantbrain.com/users/self/alphas"
COMPETITION_URL = "https://api.worldquantbrain.com/competitions/{cid}/alphas"
PAGE_LIMIT = 100
# Brain rejects large offsets on /users/self/alphas; stay safely below.
OFFSET_CEILING = 9_900
DEFAULT_ACCOUNTS = [("brain", "brain.txt"), ("brain2", "brain2.txt")]


# ---------------------------------------------------------------------------
# Sharded running log
# ---------------------------------------------------------------------------

class ShardedJsonlWriter:
    """Append JSONL records to <stem>_partNNNN.jsonl files under a size cap.

    Resumes on the highest existing part; opens the next part when writing
    one more line would push the current file past max_bytes. Keeps every
    shard under the GitHub per-file upload limit.
    """

    def __init__(self, directory: Path, stem: str, max_bytes: int):
        self.directory = directory
        self.stem = stem
        self.max_bytes = max_bytes
        self.directory.mkdir(parents=True, exist_ok=True)
        self._index = self._latest_index()
        self._handle = None
        self._size = 0

    def _part_path(self, index: int) -> Path:
        return self.directory / f"{self.stem}_part{index:04d}.jsonl"

    def _latest_index(self) -> int:
        indices = []
        for path in self.directory.glob(f"{self.stem}_part*.jsonl"):
            suffix = path.stem.rsplit("_part", 1)[-1]
            if suffix.isdigit():
                indices.append(int(suffix))
        return max(indices) if indices else 1

    def _open_current(self) -> None:
        path = self._part_path(self._index)
        self._size = path.stat().st_size if path.exists() else 0
        self._handle = path.open("a")

    def write(self, record: dict) -> None:
        line = json.dumps(record, default=str) + "\n"
        encoded = len(line.encode("utf-8"))
        if self._handle is None:
            self._open_current()
        if self._size + encoded > self.max_bytes and self._size > 0:
            self._handle.close()
            self._index += 1
            self._open_current()
        self._handle.write(line)
        self._size += encoded

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def part_paths(self) -> list[Path]:
        return sorted(self.directory.glob(f"{self.stem}_part*.jsonl"))


# ---------------------------------------------------------------------------
# Record normalization (master.jsonl-compatible)
# ---------------------------------------------------------------------------

def _f(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def normalize_alpha(detail: dict, account: str) -> dict:
    """Map one /users/self/alphas result onto the master.jsonl record schema."""
    is_stats = detail.get("is") or {}
    os_stats = detail.get("os") or {}
    regular = detail.get("regular") or {}
    return {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "expression": regular.get("code") or "",
        "alpha_id": detail.get("id"),
        "status": "done",
        "settings": detail.get("settings") or {},
        "is_sharpe": _f(is_stats.get("sharpe")),
        "is_fitness": _f(is_stats.get("fitness")),
        "is_turnover": _f(is_stats.get("turnover")),
        "is_returns": _f(is_stats.get("returns")),
        "is_drawdown": _f(is_stats.get("drawdown")),
        "is_margin": _f(is_stats.get("margin")),
        "is_ic": None,
        "os_sharpe": _f(os_stats.get("sharpe")),
        "os_fitness": _f(os_stats.get("fitness")),
        "os_turnover": _f(os_stats.get("turnover")),
        "os_returns": _f(os_stats.get("returns")),
        "grade": detail.get("grade"),
        "checks": is_stats.get("checks"),
        "date_created": detail.get("dateCreated"),
        "date_submitted": detail.get("dateSubmitted"),
        "brain_status": detail.get("status"),
        "stage": detail.get("stage"),
        "account": account,
        "source": "api_backfill",
        "raw_alpha_detail": detail,
    }


def is_submitted(detail: dict) -> bool:
    return bool(detail.get("dateSubmitted")) or detail.get("stage") == "OS"


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _get_page(session, params: dict, credentials: str, sleep_seconds: float) -> dict:
    """One page with rate-limit retries and transparent re-auth on expiry."""
    for _ in range(2):
        response = session.get(ALPHAS_URL, params=params)
        if is_auth_expired_response(response):
            print("  session expired; re-authenticating ...")
            if not authenticate_session(session, credentials, max_attempts=3):
                raise RuntimeError("Re-authentication failed mid-fetch.")
            continue
        break
    if response.status_code == 429 or response.status_code >= 500:
        return _request_json_with_retries(session, ALPHAS_URL, params=params)
    response.raise_for_status()
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return response.json()


def _rewind_iso(timestamp: str, seconds: float) -> str:
    """Move an ISO timestamp back so boundary records are re-fetched, not skipped."""
    try:
        return (datetime.fromisoformat(timestamp) - timedelta(seconds=seconds)).isoformat()
    except ValueError:
        return timestamp


def fetch_all_alphas(
    session,
    credentials: str,
    *,
    seen_ids: set[str],
    date_cursor: str | None,
    sleep_seconds: float,
    max_records: int = 0,
):
    """Yield alpha details in ascending dateCreated order, starting at cursor.

    Uses offset paging within a dateCreated> window and slides the window
    forward whenever the offset ceiling is reached, so any history depth is
    reachable despite Brain's offset cap. The cursor is rewound slightly at
    every boundary; the seen-id set absorbs the resulting overlap.
    """
    # Generous rewind on resume in case the API filter is strictly-greater.
    cursor = _rewind_iso(date_cursor, 3600.0) if date_cursor else None
    prev_boundary: str | None = None
    fetched = 0
    while True:
        offset = 0
        window_last_date = None
        while offset < OFFSET_CEILING:
            params: dict = {
                "limit": PAGE_LIMIT,
                "offset": offset,
                "order": "dateCreated",
            }
            if cursor:
                params["dateCreated>"] = cursor
            payload = _get_page(session, params, credentials, sleep_seconds)
            results = payload.get("results") or []
            count = int(payload.get("count") or 0)
            if offset == 0:
                where = f"dateCreated>={cursor}" if cursor else "from beginning"
                print(f"  window {where}: {count} records total")
            if not results:
                return
            for detail in results:
                alpha_id = str(detail.get("id") or "")
                window_last_date = detail.get("dateCreated") or window_last_date
                if not alpha_id or alpha_id in seen_ids:
                    continue
                seen_ids.add(alpha_id)
                yield detail
                fetched += 1
                if max_records and fetched >= max_records:
                    return
            offset += PAGE_LIMIT
            if offset >= count:
                return
        # Hit the offset ceiling with records remaining: slide the window.
        if not window_last_date or (prev_boundary and window_last_date <= prev_boundary):
            print("  WARNING: date cursor stalled; stopping to avoid a loop.")
            return
        prev_boundary = window_last_date
        cursor = _rewind_iso(window_last_date, 1.0)


def fetch_os_extras(session, alpha_id: str, *, pnl_attempts: int) -> dict:
    """Yearly stats plus daily PnL for one submitted alpha."""
    yearly = fetch_yearly_stats(session, alpha_id)
    pnl = fetch_pnl_series_with_retry(session, alpha_id, max_attempts=pnl_attempts)
    return {
        "alpha_id": alpha_id,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "yearly_stats": {str(k): v for k, v in (yearly or {}).items()},
        "pnl": pnl or {},
    }


def fetch_competition_snapshot(session, competition_id: str) -> list[dict]:
    url = COMPETITION_URL.format(cid=competition_id)
    results: list[dict] = []
    offset = 0
    while True:
        payload = _request_json_with_retries(
            session, url, params={"limit": PAGE_LIMIT, "offset": offset}
        )
        page = payload.get("results") or []
        results.extend(page)
        if len(page) < PAGE_LIMIT:
            return results
        offset += PAGE_LIMIT


# ---------------------------------------------------------------------------
# Per-account state
# ---------------------------------------------------------------------------

def load_state(account_dir: Path) -> dict:
    path = account_dir / "state.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(account_dir: Path, state: dict) -> None:
    account_dir.mkdir(parents=True, exist_ok=True)
    (account_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")


def scan_seen_ids(writer: ShardedJsonlWriter) -> tuple[set[str], str | None]:
    """Collect alpha ids and the max dateCreated from existing shards."""
    seen: set[str] = set()
    max_date: str | None = None
    for path in writer.part_paths():
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                alpha_id = rec.get("alpha_id")
                if alpha_id:
                    seen.add(str(alpha_id))
                created = rec.get("date_created")
                if created and (max_date is None or created > max_date):
                    max_date = created
    return seen, max_date


# ---------------------------------------------------------------------------
# Account run
# ---------------------------------------------------------------------------

def run_account(account: str, credentials: str, args) -> dict:
    account_dir = Path(args.out_dir) / account
    max_bytes = int(args.max_shard_mb * 1024 * 1024)
    alphas_writer = ShardedJsonlWriter(account_dir, "alphas", max_bytes)
    os_writer = ShardedJsonlWriter(account_dir, "os", max_bytes)

    print(f"[{account}] scanning existing shards ...")
    seen_ids, shard_max_date = scan_seen_ids(alphas_writer)
    state = load_state(account_dir)
    cursor = None if args.full else (state.get("date_cursor") or shard_max_date)
    print(f"[{account}] {len(seen_ids)} records already on disk; cursor={cursor or 'none'}")

    print(f"[{account}] signing in with {credentials} ...")
    session = sign_in(credentials)

    submitted_ids: list[str] = list(state.get("os_fetched_ids") or [])
    submitted_seen = set(submitted_ids)
    n_new = 0
    last_date = cursor
    try:
        for detail in fetch_all_alphas(
            session,
            credentials,
            seen_ids=seen_ids,
            date_cursor=cursor,
            sleep_seconds=args.sleep,
            max_records=args.max_records,
        ):
            record = normalize_alpha(detail, account)
            alphas_writer.write(record)
            n_new += 1
            created = record.get("date_created")
            if created and (last_date is None or created > last_date):
                last_date = created
            if is_submitted(detail) and not args.skip_os_extras:
                alpha_id = str(detail.get("id"))
                if alpha_id not in submitted_seen:
                    print(f"  [{account}] OS extras for submitted alpha {alpha_id} ...")
                    os_writer.write(
                        fetch_os_extras(session, alpha_id, pnl_attempts=args.pnl_attempts)
                    )
                    submitted_seen.add(alpha_id)
                    submitted_ids.append(alpha_id)
            if n_new % 500 == 0:
                alphas_writer.flush()
                os_writer.flush()
                save_state(account_dir, {
                    **state,
                    "date_cursor": last_date,
                    "os_fetched_ids": submitted_ids,
                    "last_run_at": datetime.now(timezone.utc).isoformat(),
                })
                print(f"  [{account}] {n_new} new records so far (through {last_date})")
    finally:
        alphas_writer.close()
        os_writer.close()
        save_state(account_dir, {
            **state,
            "date_cursor": last_date,
            "os_fetched_ids": submitted_ids,
            "total_on_disk": len(seen_ids),
            "last_run_at": datetime.now(timezone.utc).isoformat(),
            "last_run_new_records": n_new,
        })

    if args.competition_id:
        print(f"[{account}] fetching competition snapshot {args.competition_id} ...")
        try:
            snapshot = fetch_competition_snapshot(session, args.competition_id)
            out = account_dir / f"competition_{args.competition_id}.json"
            out.write_text(json.dumps({
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "competition_id": args.competition_id,
                "results": snapshot,
            }, indent=2, default=str) + "\n")
            print(f"[{account}] competition snapshot: {len(snapshot)} alphas -> {out}")
        except Exception as exc:  # snapshot is best-effort, never fail the backfill
            print(f"[{account}] competition snapshot failed: {exc}")

    shards = [str(p.relative_to(ROOT)) for p in alphas_writer.part_paths()]
    print(f"[{account}] done: {n_new} new records, {len(seen_ids)} total, shards: {shards}")
    return {"account": account, "new": n_new, "total": len(seen_ids), "shards": shards}


def merge_into_master(out_dir: Path, accounts: list[str]) -> None:
    from core.logs import merge_session_to_master

    for account in accounts:
        for path in sorted((out_dir / account).glob("alphas_part*.jsonl")):
            n_new, n_updated = merge_session_to_master(path)
            print(f"[merge] {path.name} ({account}): +{n_new} new, {n_updated} updated in master.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--account", help="account label, e.g. brain or brain2")
    parser.add_argument("--credentials", help="credentials file for --account")
    parser.add_argument("--all", action="store_true",
                        help="fetch both default accounts (brain.txt, brain2.txt)")
    parser.add_argument("--out-dir", default=str(ROOT / "records"),
                        help="output root (default: records/, committable)")
    parser.add_argument("--max-shard-mb", type=float, default=95.0,
                        help="rotate shards before this size (default 95, under GitHub's 100 MB cap)")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="seconds between list pages (default 1.0)")
    parser.add_argument("--full", action="store_true",
                        help="ignore the resume cursor and rescan from the beginning")
    parser.add_argument("--max-records", type=int, default=0,
                        help="stop after N new records (testing; 0 = no limit)")
    parser.add_argument("--skip-os-extras", action="store_true",
                        help="skip yearly-stats/PnL fetches for submitted alphas")
    parser.add_argument("--pnl-attempts", type=int, default=10,
                        help="max PnL fetch attempts per submitted alpha (default 10)")
    parser.add_argument("--competition-id", default="IQC2026S1",
                        help="competition snapshot to save per account ('' to skip)")
    parser.add_argument("--merge-master", action="store_true",
                        help="after fetching, merge shards into artifacts/sim/master.jsonl")
    args = parser.parse_args()

    if args.all and args.account:
        parser.error("--all and --account are mutually exclusive; "
                     "--all means both accounts, --account targets one.")
    if args.all:
        targets = DEFAULT_ACCOUNTS
    elif args.account:
        targets = [(args.account, args.credentials or f"{args.account}.txt")]
    else:
        parser.error("Pass --all or --account/--credentials.")

    summaries = []
    for account, credentials in targets:
        if not Path(credentials).expanduser().exists():
            print(f"[{account}] SKIP: credentials file {credentials} not found")
            continue
        try:
            summaries.append(run_account(account, credentials, args))
        except Exception as exc:
            print(f"[{account}] FAILED: {exc}")

    if args.merge_master and summaries:
        merge_into_master(Path(args.out_dir), [s["account"] for s in summaries])

    print("\nSummary:")
    for s in summaries:
        print(f"  {s['account']}: +{s['new']} new, {s['total']} total, {len(s['shards'])} shard(s)")


if __name__ == "__main__":
    main()
