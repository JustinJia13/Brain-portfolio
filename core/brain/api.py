"""Minimal WorldQuant Brain API helpers used by the hybrid search pipeline."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from os.path import expanduser
from pathlib import Path
from urllib.parse import urljoin
from urllib.parse import urlsplit

import requests
from requests.auth import HTTPBasicAuth


BRAIN_AUTH_URL = "https://api.worldquantbrain.com/authentication"
_AUTH_LOCK = threading.Lock()
_AUTH_RECENT_SUCCESS_SECONDS = 30.0
_PERSONA_AUTH_POLL_SECONDS = 5.0
_PERSONA_AUTH_TIMEOUT_SECONDS = 300.0
_PERSONA_STATUS_MAX_POSTS = 3


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _persona_status_max_posts() -> int:
    return _int_env("WQ_BRAIN_BIOMETRIC_STATUS_POSTS", _PERSONA_STATUS_MAX_POSTS)


def _persona_finalize_max_posts() -> int:
    return _int_env("WQ_BRAIN_BIOMETRIC_FINALIZE_POSTS", 1)


def _persona_settle_seconds() -> float:
    return _float_env("WQ_BRAIN_BIOMETRIC_SETTLE_SECONDS", 60.0)


def _persona_enabled_for_credentials(credentials_path: str) -> bool:
    """Only the primary Brain credential is allowed to trigger face-scan auth."""

    return Path(expanduser(credentials_path)).name == "brain.txt"


def auth_attempt_budget_status() -> dict[str, int | str]:
    """Local auth-attempt budget removed; reported as unlimited for compatibility."""

    return {"date": "", "limit": -1, "used": 0, "remaining": -1, "path": "(disabled)"}


def reserve_auth_attempt(kind: str, url: str, *, logger=None) -> bool:
    """Local daily auth-attempt budget removed — always allow.

    The platform's own rate limit (HTTP 429) is the only auth constraint now.
    """

    return True


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        version = Path("/proc/version").read_text(errors="ignore").lower()
    except OSError:
        version = ""
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text(errors="ignore").lower()
    except OSError:
        release = ""
    return "microsoft" in f"{version}\n{release}" or "wsl" in release


def _windows_executable(name: str) -> str:
    for candidate in (
        f"/mnt/c/WINDOWS/System32/{name}",
        f"/mnt/c/WINDOWS/system32/{name}",
        name,
    ):
        if candidate == name or Path(candidate).exists():
            return candidate
    return name


def _open_browser_url(url: str) -> bool:
    if _is_wsl():
        commands = [
            [
                _windows_executable("powershell.exe"),
                "-NoProfile",
                "-Command",
                "Start-Process",
                "-FilePath",
                url,
            ],
            [_windows_executable("cmd.exe"), "/c", "start", "", url],
        ]
        for command in commands:
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                continue
            if completed.returncode == 0:
                return True
        return False

    import webbrowser

    return bool(webbrowser.open(url, new=2))


def _brain_biometric_url(response: requests.Response) -> str | None:
    www_auth = (response.headers.get("WWW-Authenticate") or "").lower()
    location = response.headers.get("Location")
    if not location:
        return None
    candidate = urljoin(response.url, location)
    netloc = urlsplit(candidate).netloc.lower()
    if "worldquantbrain.com" in netloc and ("persona" in www_auth or "persona" in location.lower()):
        return candidate
    return None


def _persona_inquiry_id(response: requests.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    if isinstance(payload, dict) and payload.get("inquiry"):
        return str(payload["inquiry"])
    return None


def _persona_inquiry_hint(response: requests.Response) -> str | None:
    inquiry_id = _persona_inquiry_id(response)
    if inquiry_id:
        return f"https://inquiry.withpersona.com/verify?inquiry-id={inquiry_id}"
    return None


def _open_persona_url_and_wait(persona_url: str, *, logger) -> bool:
    logger.warning("Brain requires Persona biometric authentication.")
    logger.warning("Opening biometric URL: %s", persona_url)
    if not _open_browser_url(persona_url):
        logger.warning("Could not open a browser automatically. Open this URL manually: %s", persona_url)

    if not sys.stdin.isatty():
        logger.error("Cannot wait for biometric authentication because stdin is not interactive.")
        return False

    input("Complete the face scan in the browser, then press Enter here to continue ... ")
    return True


def _complete_persona_authentication(
    session: requests.Session,
    auth_response: requests.Response,
    *,
    logger,
) -> bool:
    biometric_url = _brain_biometric_url(auth_response)
    inquiry_hint = _persona_inquiry_hint(auth_response)
    if not biometric_url:
        if inquiry_hint:
            logger.error(
                "Brain requested biometric authentication but did not expose the "
                "Brain callback Location header. Persona URL was: %s",
                inquiry_hint,
            )
        return False

    if not _open_persona_url_and_wait(biometric_url, logger=logger):
        return False

    settle_seconds = _persona_settle_seconds()
    if settle_seconds > 0:
        logger.info(
            "Waiting %.0fs for Brain/Persona verification state to propagate.",
            settle_seconds,
        )
        time.sleep(settle_seconds)

    deadline = time.monotonic() + _PERSONA_AUTH_TIMEOUT_SECONDS
    next_url = biometric_url
    status_posts = 0
    while time.monotonic() < deadline:
        max_status_posts = _persona_status_max_posts()
        if status_posts >= max_status_posts:
            logger.error(
                "Stopped after %d biometric status POSTs. "
                "Set WQ_BRAIN_BIOMETRIC_STATUS_POSTS only if you intentionally want more.",
                max_status_posts,
            )
            return False
        status_posts += 1
        try:
            response = session.post(next_url)
        except requests.RequestException as exc:
            logger.warning("Biometric status check failed: %s", exc)
            time.sleep(_PERSONA_AUTH_POLL_SECONDS)
            continue

        if response.status_code == 201:
            setattr(session, "_wq_last_auth_at", time.monotonic())
            logger.info("Brain biometric authentication successful.")
            return True

        if response.status_code == 429:
            sleep_seconds = _auth_retry_sleep_seconds(
                response,
                attempt=1,
                default_sleep_seconds=_PERSONA_AUTH_POLL_SECONDS,
                max_sleep_seconds=60.0,
            )
            logger.warning("Biometric status check rate limited; sleeping %.0fs.", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        followup_url = _brain_biometric_url(response)
        if followup_url and followup_url != next_url:
            next_url = followup_url
            logger.warning("Brain returned a follow-up biometric URL: %s", next_url)

        logger.info("Biometric status check returned HTTP %s; polling again.", response.status_code)
        time.sleep(_PERSONA_AUTH_POLL_SECONDS)

    logger.error("Timed out waiting for Brain biometric authentication.")
    return False


def _complete_persona_inquiry_authentication(
    session: requests.Session,
    auth_response: requests.Response,
    basic_auth: HTTPBasicAuth,
    *,
    logger,
) -> bool:
    """Handle Brain responses that expose only a Persona inquiry id.

    Some Brain auth responses return {"inquiry": "..."} without a Brain
    callback Location header.  The hosted Persona URL is browser-facing; after
    the user completes it, finalize by asking Brain for authentication again,
    not by POSTing to Persona.
    """

    inquiry_id = _persona_inquiry_id(auth_response)
    persona_url = _persona_inquiry_hint(auth_response)
    if not inquiry_id or not persona_url:
        return False

    if not _open_persona_url_and_wait(persona_url, logger=logger):
        return False

    settle_seconds = _persona_settle_seconds()
    if settle_seconds > 0:
        logger.info(
            "Waiting %.0fs for Brain/Persona verification state to propagate.",
            settle_seconds,
        )
        time.sleep(settle_seconds)

    max_posts = _persona_finalize_max_posts()
    for post_index in range(1, max_posts + 1):
        try:
            response = session.post(BRAIN_AUTH_URL, auth=basic_auth)
        except requests.RequestException as exc:
            logger.warning("Brain auth finalization failed: %s", exc)
            return False

        if response.status_code == 201:
            setattr(session, "_wq_last_auth_at", time.monotonic())
            logger.info("Brain biometric authentication successful.")
            return True

        if response.status_code == 429:
            sleep_seconds = _auth_retry_sleep_seconds(
                response,
                attempt=post_index,
                default_sleep_seconds=_PERSONA_AUTH_POLL_SECONDS,
                max_sleep_seconds=60.0,
            )
            logger.warning("Brain auth finalization rate limited; sleeping %.0fs.", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        followup_url = _brain_biometric_url(response)
        if followup_url:
            logger.warning("Brain returned a biometric callback URL after Persona scan.")
            return _complete_persona_authentication(session, response, logger=logger)

        next_inquiry_id = _persona_inquiry_id(response)
        if next_inquiry_id:
            next_url = _persona_inquiry_hint(response)
            if next_inquiry_id != inquiry_id:
                logger.error(
                    "Brain returned a new Persona inquiry after the scan. "
                    "Stopping to avoid creating an inquiry loop. New Persona URL: %s",
                    next_url,
                )
                return False
            logger.warning(
                "Brain still reports the same Persona inquiry as incomplete "
                "(finalize POST %d/%d).",
                post_index,
                max_posts,
            )
        else:
            logger.warning(
                "Brain auth finalization returned HTTP %s: %s",
                response.status_code,
                response.text[:200],
            )

        if post_index < max_posts:
            time.sleep(_PERSONA_AUTH_POLL_SECONDS)

    logger.error(
        "Brain biometric authentication did not finalize after %d Brain auth POST(s).",
        max_posts,
    )
    return False


def load_credentials(credentials_path: str = "brain.txt") -> tuple[str, str]:
    """Load the username/password pair expected by the Brain API."""

    with open(expanduser(credentials_path)) as handle:
        credentials = json.load(handle)
    if not isinstance(credentials, list) or len(credentials) != 2:
        raise ValueError("Expected brain.txt to contain a two-item JSON list: [username, password].")
    username, password = credentials
    return str(username), str(password)


def is_auth_expired_response(response: requests.Response) -> bool:
    """Return True when Brain indicates the current session needs re-auth."""

    return response.status_code == 401 or (
        response.status_code == 200
        and "Incorrect authentication" in response.text
    )


def _auth_retry_sleep_seconds(
    response: requests.Response | None,
    *,
    attempt: int,
    default_sleep_seconds: float,
    max_sleep_seconds: float,
) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(float(retry_after), 1.0)
            except ValueError:
                pass
    return min(max_sleep_seconds, default_sleep_seconds * (2 ** max(attempt - 1, 0)))


def authenticate_session(
    session: requests.Session,
    credentials_path: str = "brain.txt",
    *,
    max_attempts: int = 1,
    default_sleep_seconds: float = 30.0,
    max_sleep_seconds: float = 300.0,
    logger=None,
    force: bool = False,
) -> bool:
    """Authenticate an existing session with lock/backoff protection.

    Several simulation workers can discover an expired Brain session at the
    same time.  This helper serializes re-authentication inside the process and
    backs off on auth 429s so those workers do not hammer /authentication.
    """

    log = logger or logging.getLogger(__name__)
    username, password = load_credentials(credentials_path)
    basic_auth = HTTPBasicAuth(username, password)
    session.auth = None

    with _AUTH_LOCK:
        last_auth = float(getattr(session, "_wq_last_auth_at", 0.0) or 0.0)
        if not force and last_auth and time.monotonic() - last_auth < _AUTH_RECENT_SUCCESS_SECONDS:
            log.info("Recent Brain authentication already succeeded; skipping duplicate re-auth.")
            return True

        for attempt in range(1, max_attempts + 1):
            response: requests.Response | None = None
            try:
                response = session.post(BRAIN_AUTH_URL, auth=basic_auth)
                if response.status_code == 429:
                    sleep_seconds = _auth_retry_sleep_seconds(
                        response,
                        attempt=attempt,
                        default_sleep_seconds=default_sleep_seconds,
                        max_sleep_seconds=max_sleep_seconds,
                    )
                    log.warning(
                        "Brain authentication rate limited; sleeping %.0fs before retry "
                        "(attempt %d/%d).",
                        sleep_seconds,
                        attempt,
                        max_attempts,
                    )
                    time.sleep(sleep_seconds)
                    continue

                if response.status_code == 401:
                    if _brain_biometric_url(response):
                        if not _persona_enabled_for_credentials(credentials_path):
                            log.error(
                                "Brain requested Persona biometric authentication for %s, "
                                "but interactive Persona auth is only enabled for brain.txt.",
                                credentials_path,
                            )
                            return False
                        return _complete_persona_authentication(session, response, logger=log)
                    inquiry_hint = _persona_inquiry_hint(response)
                    if inquiry_hint:
                        if not _persona_enabled_for_credentials(credentials_path):
                            log.error(
                                "Brain requested Persona biometric authentication for %s, "
                                "but interactive Persona auth is only enabled for brain.txt.",
                                credentials_path,
                            )
                            return False
                        return _complete_persona_inquiry_authentication(
                            session,
                            response,
                            basic_auth,
                            logger=log,
                        )

                response.raise_for_status()
                setattr(session, "_wq_last_auth_at", time.monotonic())
                log.info("Brain authentication successful.")
                return True
            except requests.HTTPError as exc:
                response = exc.response
                if response is not None and response.status_code == 429:
                    sleep_seconds = _auth_retry_sleep_seconds(
                        response,
                        attempt=attempt,
                        default_sleep_seconds=default_sleep_seconds,
                        max_sleep_seconds=max_sleep_seconds,
                    )
                    log.warning(
                        "Brain authentication rate limited; sleeping %.0fs before retry "
                        "(attempt %d/%d).",
                        sleep_seconds,
                        attempt,
                        max_attempts,
                    )
                    time.sleep(sleep_seconds)
                    continue
                log.error("Brain authentication failed: %s", exc)
                return False
            except requests.RequestException as exc:
                if attempt >= max_attempts:
                    log.error("Brain authentication failed after %d attempts: %s", max_attempts, exc)
                    return False
                sleep_seconds = _auth_retry_sleep_seconds(
                    response,
                    attempt=attempt,
                    default_sleep_seconds=default_sleep_seconds,
                    max_sleep_seconds=max_sleep_seconds,
                )
                log.warning(
                    "Brain authentication error: %s; sleeping %.0fs before retry "
                    "(attempt %d/%d).",
                    exc,
                    sleep_seconds,
                    attempt,
                    max_attempts,
                )
                time.sleep(sleep_seconds)

    return False


def sign_in(credentials_path: str = "brain.txt") -> requests.Session:
    """Authenticate and return a requests session ready for API calls."""

    session = requests.Session()
    if not authenticate_session(session, credentials_path, force=True):
        raise RuntimeError("Failed to authenticate with WorldQuant Brain.")
    return session


def _retry_sleep_seconds(response: requests.Response, default_sleep_seconds: float) -> float:
    """Pick a wait time for rate-limited requests, respecting Retry-After when available."""

    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(float(retry_after), 1.0)
        except ValueError:
            pass
    return default_sleep_seconds


def _request_json_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, str | int],
    max_retries: int = 20,
    default_sleep_seconds: float = 60.0,
) -> dict:
    """GET one JSON payload, retrying on Brain rate limits and transient failures."""

    for attempt in range(1, max_retries + 1):
        response = session.get(url, params=params)

        if response.status_code == 429:
            sleep_seconds = _retry_sleep_seconds(response, default_sleep_seconds)
            print(
                f"Rate limited on {url} offset={params.get('offset', 0)} "
                f"(attempt {attempt}/{max_retries}); sleeping {sleep_seconds:.0f}s..."
            )
            time.sleep(sleep_seconds)
            continue

        response.raise_for_status()
        payload = response.json()

        message = str(payload.get("message", "")).lower() if isinstance(payload, dict) else ""
        if "rate limit" in message:
            sleep_seconds = _retry_sleep_seconds(response, default_sleep_seconds)
            print(
                f"Brain API reported rate limit on {url} offset={params.get('offset', 0)} "
                f"(attempt {attempt}/{max_retries}); sleeping {sleep_seconds:.0f}s..."
            )
            time.sleep(sleep_seconds)
            continue

        return payload

    raise RuntimeError(
        f"Exceeded rate-limit retry budget for {url} at offset={params.get('offset', 0)}."
    )


def _paged_get(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, str | int],
    limit: int = 50,
    max_retries: int = 20,
    default_sleep_seconds: float = 60.0,
) -> list[dict]:
    """Fetch one paginated Brain endpoint into a single list of records."""

    first_payload = _request_json_with_retries(
        session,
        url,
        params={**params, "limit": limit, "offset": 0},
        max_retries=max_retries,
        default_sleep_seconds=default_sleep_seconds,
    )
    results = list(first_payload.get("results", []))
    total_count = int(first_payload.get("count") or len(results))

    for offset in range(limit, total_count, limit):
        payload = _request_json_with_retries(
            session,
            url,
            params={**params, "limit": limit, "offset": offset},
            max_retries=max_retries,
            default_sleep_seconds=default_sleep_seconds,
        )
        results.extend(payload.get("results", []))

    return results


def get_datafields(
    session: requests.Session,
    search_scope: dict[str, str | int],
    *,
    dataset_id: str = "",
    search: str = "",
    limit: int = 50,
    max_retries: int = 20,
    default_sleep_seconds: float = 60.0,
) -> list[dict]:
    """Fetch all data-fields for one dataset/search-scope combination."""

    params = {
        "instrumentType": search_scope["instrumentType"],
        "region": search_scope["region"],
        "delay": search_scope["delay"],
        "universe": search_scope["universe"],
    }
    if dataset_id:
        params["dataset.id"] = dataset_id
    if search:
        params["search"] = search
    return _paged_get(
        session,
        "https://api.worldquantbrain.com/data-fields",
        params=params,
        limit=limit,
        max_retries=max_retries,
        default_sleep_seconds=default_sleep_seconds,
    )


def fetch_yearly_stats(session, alpha_id: str) -> dict[int, dict]:
    """Fetch per-year performance breakdown from Brain recordsets API.

    Returns {year: {sharpe, fitness, turnover, returns, drawdown, margin, ...}}
    or {} on any error (e.g. alpha not yet processed, 404, timeout).

    Endpoint: GET /alphas/{alpha_id}/recordsets/yearly-stats.
    """
    url = f"https://api.worldquantbrain.com/alphas/{alpha_id}/recordsets/yearly-stats"
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            return {}
        data = resp.json()
        schema = data.get("schema", {})
        props = schema.get("properties", [])
        cols = [p["name"] for p in props]
        records = data.get("records", [])
        result: dict[int, dict] = {}
        for row in records:
            if not row:
                continue
            year = int(row[0])
            result[year] = dict(zip(cols[1:], row[1:]))
        return result
    except Exception:
        return {}


def fetch_yearly_stats_with_retry(
    session,
    alpha_id: str,
    *,
    max_attempts: int = 6,
    sleep_seconds: float = 30.0,
) -> dict[int, dict]:
    """fetch_yearly_stats with retry until non-empty or timeout.

    Brain's yearly-stats endpoint returns {} immediately after a simulation
    completes — the backtest breakdown is computed asynchronously and typically
    becomes available within 1–3 minutes.  This wrapper polls until the result
    is non-empty or the retry budget is exhausted.

    Default: 6 attempts × 30 s = up to 3 minutes total wait.
    """
    attempts = max(1, int(max_attempts))
    for attempt in range(attempts):
        stats = fetch_yearly_stats(session, alpha_id)
        if stats:
            return stats
        if attempt < attempts - 1:
            time.sleep(sleep_seconds)
    return {}


def fetch_pnl_series(
    session,
    alpha_id: str,
    *,
    timeout: float = 30.0,
) -> dict[str, float] | None:
    """Fetch daily PnL timeseries for any alpha (submitted or not).

    Return values:
      dict  — PnL data (may be empty dict {} if alpha has no data)
      None  — rate limited; caller should back off and retry

    Brain uses two rate-limit signals:
      - HTTP 429 with Retry-After header (hard limit)
      - HTTP 200 with empty records[] and X-RateLimit-Remaining-Hour == 0
        (soft limit — easy to miss without checking the header)

    Endpoint: GET /alphas/{alpha_id}/recordsets/pnl
    """
    url = f"https://api.worldquantbrain.com/alphas/{alpha_id}/recordsets/pnl"
    try:
        resp = session.get(url, timeout=timeout)

        # Hard rate limit
        if resp.status_code == 429:
            return None

        if resp.status_code != 200 or not resp.content:
            return {}

        data = resp.json()
        records = data.get("records") or []

        # Soft rate limit: 200 with empty records and quota exhausted
        remaining = resp.headers.get("X-RateLimit-Remaining-Hour")
        if not records and remaining is not None and int(remaining) == 0:
            return None

        result: dict[str, float] = {}
        for row in records:
            if row and len(row) >= 2 and row[1] is not None:
                result[str(row[0])] = float(row[1])
        return result
    except Exception:
        return {}


def fetch_pnl_series_with_retry(
    session,
    alpha_id: str,
    *,
    max_attempts: int | None = None,
    sleep_seconds: float = 1.0,
    timeout: float = 30.0,
) -> dict[str, float]:
    """fetch_pnl_series with retry until non-empty data is returned.

    Retries indefinitely (max_attempts=None) at sleep_seconds intervals on
    both rate limits (None) and async delay (empty {}).  Pass max_attempts=N
    to cap retries (e.g. for portfolio loading where some alphas have no data).
    """
    attempt = 0
    while max_attempts is None or attempt < max_attempts:
        attempt += 1
        pnl = fetch_pnl_series(session, alpha_id, timeout=timeout)
        if pnl is None:
            print(f"  [PnL] rate limited for {alpha_id} — waiting {sleep_seconds:.0f}s ...")
            time.sleep(sleep_seconds)
            continue
        if pnl:
            return pnl
        print(f"  [PnL] empty for {alpha_id} — waiting {sleep_seconds:.0f}s ...")
        time.sleep(sleep_seconds)
    return {}


def fetch_competition_alpha_ids(
    session,
    competition_id: str = "IQC2026S1",
    *,
    limit: int = 100,
) -> list[str]:
    """Return list of alpha IDs in a competition portfolio.

    Fetches all pages until exhausted.
    """
    url = f"https://api.worldquantbrain.com/competitions/{competition_id}/alphas"
    alpha_ids: list[str] = []
    offset = 0
    while True:
        try:
            resp = session.get(url, params={"limit": limit, "offset": offset}, timeout=30)
            if resp.status_code != 200:
                break
            data = resp.json()
            results = data.get("results") or []
            for a in results:
                aid = a.get("id") or a.get("alpha")
                if aid:
                    alpha_ids.append(str(aid))
            if len(results) < limit:
                break
            offset += limit
        except Exception:
            break
    return alpha_ids


__all__ = [
    "auth_attempt_budget_status", "authenticate_session", "get_datafields",
    "is_auth_expired_response", "load_credentials", "reserve_auth_attempt", "sign_in",
    "fetch_yearly_stats", "fetch_yearly_stats_with_retry",
    "fetch_pnl_series", "fetch_pnl_series_with_retry",
    "fetch_competition_alpha_ids",
]
