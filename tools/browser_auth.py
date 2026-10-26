"""Browser-session authentication helper for WorldQuant Brain.

This script intentionally does not try to finish the Persona/biometric flow
through repeated Basic Auth calls.  If Brain requires biometrics, each direct
POST to /authentication can create a fresh inquiry.  The useful fallback is to
authenticate in a real browser once, export the browser cookies, and test
whether those cookies can access Brain API endpoints.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.parse import urljoin
from urllib.parse import urlsplit, urlunsplit

import requests
from requests.auth import HTTPBasicAuth

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wq_alpha.brain.api import auth_attempt_budget_status, load_credentials, reserve_auth_attempt


API_AUTH_URL = "https://api.worldquantbrain.com/authentication"
DEFAULT_PLATFORM_URL = "https://platform.worldquantbrain.com/"
DEFAULT_PERSONA_URL_TEMPLATE = "https://inquiry.withpersona.com/verify?inquiry-id={inquiry_id}"
DEFAULT_PROFILE_DIR = Path("artifacts/auth/browser_profile")
DEFAULT_COOKIES_PATH = Path("artifacts/auth/browser_worldquant_cookies.json")
DEFAULT_STORAGE_PATH = Path("artifacts/auth/browser_storage.json")
DEFAULT_PROBE_PATH = Path("artifacts/auth/browser_session_probe.json")
DEFAULT_API_PROBE_URL = "https://api.worldquantbrain.com/competitions/IQC2026S1/alphas?limit=1&offset=0"


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
    marker = f"{version}\n{release}"
    return "microsoft" in marker or "wsl" in marker


def _windows_executable(name: str) -> str:
    for candidate in (
        f"/mnt/c/WINDOWS/System32/{name}",
        f"/mnt/c/WINDOWS/system32/{name}",
        name,
    ):
        if candidate == name or Path(candidate).exists():
            return candidate
    return name


def _wsl_to_windows_path(path: Path) -> str:
    if not _is_wsl():
        return str(path)
    try:
        return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return str(path)


def _find_browser_executable(explicit: str = "") -> str:
    if explicit:
        return explicit

    candidates = [
        "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
        "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe",
        "/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe",
        "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
        "google-chrome",
        "chromium",
        "chromium-browser",
        "microsoft-edge",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.is_absolute() and path.exists():
            return str(path)
        if not path.is_absolute():
            found = subprocess.run(
                ["bash", "-lc", f"command -v {candidate}"],
                check=False,
                capture_output=True,
                text=True,
            )
            if found.returncode == 0 and found.stdout.strip():
                return found.stdout.strip()
    raise FileNotFoundError("Could not find Chrome/Edge. Pass --browser-path explicitly.")


def _json_payload(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _persona_inquiry_url(inquiry_id: str, template: str) -> str:
    return template.format(inquiry_id=quote(inquiry_id, safe=""))


def _is_worldquant_url(url: str) -> bool:
    return "worldquantbrain.com" in urlsplit(url).netloc.lower()


def _is_persona_url(url: str) -> bool:
    return "withpersona.com" in urlsplit(url).netloc.lower()


def _basic_auth_probe(
    credentials_path: str,
    *,
    timeout: float,
    persona_template: str = DEFAULT_PERSONA_URL_TEMPLATE,
) -> dict[str, Any]:
    username, password = load_credentials(credentials_path)
    session = requests.Session()
    return _auth_post_probe(
        session,
        timeout=timeout,
        auth=HTTPBasicAuth(username, password),
        persona_template=persona_template,
    )


def _auth_post_probe(
    session: requests.Session,
    *,
    timeout: float,
    auth: HTTPBasicAuth | None = None,
    kind: str = "auth_initial",
    persona_template: str = DEFAULT_PERSONA_URL_TEMPLATE,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "auth_url": API_AUTH_URL,
        "method": "POST",
        "ok": False,
        "status_code": None,
        "content_type": None,
        "retry_after": None,
        "ratelimit_reset": None,
        "www_authenticate": None,
        "location": None,
        "biometric_url": None,
        "persona_url": None,
        "inquiry": None,
        "body_preview": "",
        "budget": auth_attempt_budget_status(),
        "error": None,
    }
    if not reserve_auth_attempt(kind, API_AUTH_URL):
        result["error"] = "Daily Brain auth attempt budget exhausted."
        result["budget"] = auth_attempt_budget_status()
        return result
    try:
        response = session.post(API_AUTH_URL, timeout=timeout, auth=auth)
    except requests.RequestException as exc:
        result["error"] = str(exc)
        return result

    result["ok"] = response.ok
    result["status_code"] = response.status_code
    result["content_type"] = response.headers.get("Content-Type")
    result["retry_after"] = response.headers.get("Retry-After")
    result["ratelimit_reset"] = response.headers.get("RateLimit-Reset")
    result["www_authenticate"] = response.headers.get("WWW-Authenticate")
    result["location"] = response.headers.get("Location")
    if result["location"]:
        location_url = urljoin(response.url, str(result["location"]))
        if _is_worldquant_url(location_url):
            result["biometric_url"] = location_url
        elif _is_persona_url(location_url):
            result["persona_url"] = location_url
    result["body_preview"] = response.text[:500]
    payload = _json_payload(response.text)
    if payload and payload.get("inquiry"):
        result["inquiry"] = str(payload["inquiry"])
        result["persona_url"] = _persona_inquiry_url(str(payload["inquiry"]), persona_template)
    return result


def _post_existing_auth_url(session: requests.Session, url: str, *, timeout: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "auth_url": url,
        "method": "POST",
        "ok": False,
        "status_code": None,
        "content_type": None,
        "retry_after": None,
        "ratelimit_reset": None,
        "www_authenticate": None,
        "location": None,
        "biometric_url": None,
        "persona_url": None,
        "body_preview": "",
        "budget": auth_attempt_budget_status(),
        "error": None,
    }
    if not reserve_auth_attempt("biometric_status", url):
        result["error"] = "Daily Brain auth attempt budget exhausted."
        result["budget"] = auth_attempt_budget_status()
        return result
    try:
        response = session.post(url, timeout=timeout)
    except requests.RequestException as exc:
        result["error"] = str(exc)
        return result

    result["ok"] = response.ok
    result["status_code"] = response.status_code
    result["content_type"] = response.headers.get("Content-Type")
    result["retry_after"] = response.headers.get("Retry-After")
    result["ratelimit_reset"] = response.headers.get("RateLimit-Reset")
    result["www_authenticate"] = response.headers.get("WWW-Authenticate")
    result["location"] = response.headers.get("Location")
    if result["location"]:
        location_url = urljoin(response.url, str(result["location"]))
        if _is_worldquant_url(location_url):
            result["biometric_url"] = location_url
        elif _is_persona_url(location_url):
            result["persona_url"] = location_url
    result["body_preview"] = response.text[:500]
    return result


def _auth_status_probe(session: requests.Session, method: str, *, timeout: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "auth_url": API_AUTH_URL,
        "method": method.upper(),
        "ok": False,
        "status_code": None,
        "content_type": None,
        "retry_after": None,
        "ratelimit_reset": None,
        "body_preview": "",
        "error": None,
    }
    try:
        response = session.request(method.upper(), API_AUTH_URL, timeout=timeout)
    except requests.RequestException as exc:
        result["error"] = str(exc)
        return result

    result["ok"] = response.ok
    result["status_code"] = response.status_code
    result["content_type"] = response.headers.get("Content-Type")
    result["retry_after"] = response.headers.get("Retry-After")
    result["ratelimit_reset"] = response.headers.get("RateLimit-Reset")
    result["body_preview"] = response.text[:500]
    return result


def _print_basic_probe(result: dict[str, Any], persona_template: str) -> None:
    print(f"{result.get('method', 'POST')} auth status: {result.get('status_code')}")
    if result.get("budget"):
        budget = result["budget"]
        print(
            "Auth attempt budget: "
            f"{budget.get('used')}/{budget.get('limit')} used, {budget.get('remaining')} remaining"
        )
    if result.get("retry_after") or result.get("ratelimit_reset"):
        print(f"Retry-After: {result.get('retry_after')}  RateLimit-Reset: {result.get('ratelimit_reset')}")
    if result.get("www_authenticate") or result.get("location"):
        print(f"WWW-Authenticate: {result.get('www_authenticate')}  Location: {result.get('location')}")
    if result.get("biometric_url"):
        print(f"Brain biometric URL: {result['biometric_url']}")
    if result.get("persona_url"):
        print(f"Persona URL: {result['persona_url']}")
    elif result.get("inquiry"):
        url = _persona_inquiry_url(str(result["inquiry"]), persona_template)
        print(f"Persona inquiry: {result['inquiry']}")
        print(f"Persona URL: {url}")
    elif result.get("error"):
        print(f"Error: {result['error']}")
    elif result.get("body_preview"):
        print(str(result["body_preview"]).replace("\n", " ")[:300])


def _wsl_host_candidates() -> list[str]:
    hosts: list[str] = []
    try:
        for line in Path("/etc/resolv.conf").read_text(errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "nameserver":
                hosts.append(parts[1])
    except OSError:
        pass
    return hosts


def _candidate_cdp_hosts(explicit_host: str = "") -> list[str]:
    hosts: list[str] = []
    if explicit_host:
        hosts.append(explicit_host)
    hosts.extend(["127.0.0.1", "localhost"])
    if _is_wsl():
        hosts.extend(_wsl_host_candidates())

    unique: list[str] = []
    for host in hosts:
        if host and host not in unique:
            unique.append(host)
    return unique


def _wait_for_cdp(port: int, *, timeout: float, hosts: list[str]) -> str | None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        for host in hosts:
            try:
                response = requests.get(f"http://{host}:{port}/json/version", timeout=2)
                if response.ok:
                    return host
            except requests.RequestException as exc:
                last_error = f"{host}: {exc}"
        time.sleep(0.5)
    print(f"WSL could not reach Chrome DevTools directly. Last error: {last_error}")
    return None


def _list_cdp_targets(host: str, port: int) -> list[dict[str, Any]]:
    response = requests.get(f"http://{host}:{port}/json", timeout=5)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else []


async def _cdp_request(websocket_url: str, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    import websockets

    async with websockets.connect(websocket_url) as websocket:
        await websocket.send(json.dumps({"id": 1, "method": method, "params": params or {}}))
        while True:
            message = json.loads(await websocket.recv())
            if message.get("id") == 1:
                return message


def _rewrite_ws_host(websocket_url: str, host: str, port: int) -> str:
    parts = urlsplit(websocket_url)
    if parts.hostname in {"127.0.0.1", "localhost", "::1"} and host not in {"127.0.0.1", "localhost"}:
        return urlunsplit((parts.scheme, f"{host}:{port}", parts.path, parts.query, parts.fragment))
    return websocket_url


def _get_cdp_cookies(host: str, port: int) -> list[dict[str, Any]]:
    targets = _list_cdp_targets(host, port)
    page = next((t for t in targets if t.get("type") == "page" and t.get("webSocketDebuggerUrl")), None)
    if not page:
        raise RuntimeError("No debuggable browser page target found.")

    websocket_url = _rewrite_ws_host(str(page["webSocketDebuggerUrl"]), host, port)
    response = asyncio.run(_cdp_request(websocket_url, "Network.getAllCookies"))
    if response.get("error"):
        response = asyncio.run(_cdp_request(websocket_url, "Storage.getCookies"))
    if response.get("error"):
        raise RuntimeError(f"Could not read browser cookies through CDP: {response['error']}")
    cookies = response.get("result", {}).get("cookies", [])
    return cookies if isinstance(cookies, list) else []


def _get_cdp_page_target(host: str, port: int) -> dict[str, Any]:
    targets = _list_cdp_targets(host, port)
    page = next(
        (
            t
            for t in targets
            if t.get("type") == "page"
            and "worldquantbrain.com" in str(t.get("url", "")).lower()
            and t.get("webSocketDebuggerUrl")
        ),
        None,
    )
    if page:
        return page
    page = next((t for t in targets if t.get("type") == "page" and t.get("webSocketDebuggerUrl")), None)
    if not page:
        raise RuntimeError("No debuggable browser page target found.")
    return page


def _storage_script() -> str:
    return """
(() => {
  const dumpStorage = (storage) => {
    const out = {};
    for (let i = 0; i < storage.length; i += 1) {
      const key = storage.key(i);
      out[key] = storage.getItem(key);
    }
    return out;
  };
  return {
    href: location.href,
    origin: location.origin,
    localStorage: dumpStorage(localStorage),
    sessionStorage: dumpStorage(sessionStorage),
  };
})()
""".strip()


def _get_cdp_storage(host: str, port: int) -> dict[str, Any]:
    page = _get_cdp_page_target(host, port)
    websocket_url = _rewrite_ws_host(str(page["webSocketDebuggerUrl"]), host, port)
    response = asyncio.run(
        _cdp_request(
            websocket_url,
            "Runtime.evaluate",
            {"expression": _storage_script(), "returnByValue": True},
        )
    )
    if response.get("error"):
        raise RuntimeError(f"Could not read browser storage through CDP: {response['error']}")
    value = response.get("result", {}).get("result", {}).get("value", {})
    return value if isinstance(value, dict) else {}


def _write_windows_cookie_exporter(script_path: Path) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script = r'''
param(
  [int]$Port,
  [string]$OutPath,
  [string]$StoragePath = "",
  [int]$TimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"

function Get-CdpTargets {
  param([int]$Port, [int]$TimeoutSeconds)
  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  $lastError = ""
  while ((Get-Date) -lt $deadline) {
    try {
      return Invoke-RestMethod -UseBasicParsing -Uri "http://127.0.0.1:$Port/json" -TimeoutSec 2
    } catch {
      $lastError = $_.Exception.Message
      Start-Sleep -Milliseconds 500
    }
  }
  throw "Chrome DevTools endpoint was not ready on Windows localhost:$Port. Last error: $lastError"
}

function Invoke-Cdp {
  param([string]$WebSocketUrl, [string]$Method, [hashtable]$Params = @{})

  $client = [System.Net.WebSockets.ClientWebSocket]::new()
  $client.ConnectAsync([Uri]$WebSocketUrl, [Threading.CancellationToken]::None).GetAwaiter().GetResult()

  $payload = @{ id = 1; method = $Method; params = $Params } | ConvertTo-Json -Compress -Depth 20
  $bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
  $client.SendAsync(
    [System.ArraySegment[byte]]::new($bytes),
    [System.Net.WebSockets.WebSocketMessageType]::Text,
    $true,
    [Threading.CancellationToken]::None
  ).GetAwaiter().GetResult()

  $buffer = [byte[]]::new(1048576)
  $builder = [System.Text.StringBuilder]::new()
  do {
    $segment = [System.ArraySegment[byte]]::new($buffer)
    $result = $client.ReceiveAsync($segment, [Threading.CancellationToken]::None).GetAwaiter().GetResult()
    if ($result.Count -gt 0) {
      [void]$builder.Append([System.Text.Encoding]::UTF8.GetString($buffer, 0, $result.Count))
    }
  } until ($result.EndOfMessage)

  $client.Dispose()
  return $builder.ToString() | ConvertFrom-Json
}

$targets = Get-CdpTargets -Port $Port -TimeoutSeconds $TimeoutSeconds
$target = @($targets | Where-Object { $_.type -eq "page" -and $_.webSocketDebuggerUrl -and $_.url -like "*worldquantbrain.com*" } | Select-Object -First 1)[0]
if ($null -eq $target) {
  $target = @($targets | Where-Object { $_.type -eq "page" -and $_.webSocketDebuggerUrl } | Select-Object -First 1)[0]
}
if ($null -eq $target) {
  throw "No debuggable page target found on Chrome DevTools port $Port."
}

$response = Invoke-Cdp -WebSocketUrl $target.webSocketDebuggerUrl -Method "Network.getAllCookies"
if ($null -ne $response.error) {
  $response = Invoke-Cdp -WebSocketUrl $target.webSocketDebuggerUrl -Method "Storage.getCookies"
}
if ($null -ne $response.error) {
  throw "Could not read cookies through Chrome DevTools: $($response.error.message)"
}

$cookies = @($response.result.cookies)
$parent = Split-Path -Parent $OutPath
if ($parent) {
  New-Item -ItemType Directory -Force -Path $parent | Out-Null
}
ConvertTo-Json -InputObject $cookies -Depth 32 -Compress | Set-Content -Path $OutPath -Encoding UTF8
if ($StoragePath) {
  $storageScript = @'
(() => {
  const dumpStorage = (storage) => {
    const out = {};
    for (let i = 0; i < storage.length; i += 1) {
      const key = storage.key(i);
      out[key] = storage.getItem(key);
    }
    return out;
  };
  return {
    href: location.href,
    origin: location.origin,
    localStorage: dumpStorage(localStorage),
    sessionStorage: dumpStorage(sessionStorage),
  };
})()
'@
  $storageResponse = Invoke-Cdp -WebSocketUrl $target.webSocketDebuggerUrl -Method "Runtime.evaluate" -Params @{
    expression = $storageScript
    returnByValue = $true
  }
  $storageParent = Split-Path -Parent $StoragePath
  if ($storageParent) {
    New-Item -ItemType Directory -Force -Path $storageParent | Out-Null
  }
  ConvertTo-Json -InputObject $storageResponse.result.result.value -Depth 32 -Compress | Set-Content -Path $StoragePath -Encoding UTF8
}
Write-Output "OK $($cookies.Count)"
'''
    script_path.write_text(script.strip() + "\n")


def _load_cookies(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        return [payload]
    return payload if isinstance(payload, list) else []


def _load_storage(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def _get_windows_cdp_state(
    port: int,
    cookies_path: Path,
    storage_path: Path,
    *,
    timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    script_path = Path("artifacts/auth/cdp_cookie_export.ps1")
    _write_windows_cookie_exporter(script_path)
    windows_script = _wsl_to_windows_path(script_path.resolve())
    windows_output = _wsl_to_windows_path(cookies_path.resolve())
    windows_storage = _wsl_to_windows_path(storage_path.resolve())

    completed = subprocess.run(
        [
            _windows_executable("powershell.exe"),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            windows_script,
            "-Port",
            str(port),
            "-OutPath",
            windows_output,
            "-StoragePath",
            windows_storage,
            "-TimeoutSeconds",
            str(int(max(timeout, 5))),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=max(timeout + 10, 20),
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"Windows CDP cookie export failed: {message}")
    return _load_cookies(cookies_path), _load_storage(storage_path)


def _worldquant_cookies(cookies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        cookie
        for cookie in cookies
        if "worldquantbrain.com" in str(cookie.get("domain", "")).lower()
    ]


def _save_cookies(path: Path, cookies: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cookies, indent=2, sort_keys=True) + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _save_storage(path: Path, storage: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(storage, indent=2, sort_keys=True) + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _session_from_cookies(cookies: list[dict[str, Any]]) -> requests.Session:
    session = requests.Session()
    for cookie in cookies:
        name = cookie.get("name")
        value = cookie.get("value")
        if not name or value is None:
            continue
        session.cookies.set(
            str(name),
            str(value),
            domain=cookie.get("domain") or None,
            path=cookie.get("path") or "/",
        )
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Referer": DEFAULT_PLATFORM_URL,
        }
    )
    return session


def _open_url(url: str) -> bool:
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


def _probe_cookie_session(cookies: list[dict[str, Any]], url: str, *, timeout: float) -> dict[str, Any]:
    session = _session_from_cookies(cookies)
    result: dict[str, Any] = {
        "url": url,
        "ok": False,
        "status_code": None,
        "content_type": None,
        "body_preview": "",
        "error": None,
        "cookie_count": len(cookies),
        "cookie_names": sorted({str(cookie.get("name", "")) for cookie in cookies if cookie.get("name")}),
    }
    try:
        response = session.get(url, timeout=timeout)
    except requests.RequestException as exc:
        result["error"] = str(exc)
        return result

    result["ok"] = response.ok
    result["status_code"] = response.status_code
    result["content_type"] = response.headers.get("Content-Type")
    result["body_preview"] = response.text[:500]
    return result


def _looks_like_jwt(value: str) -> bool:
    parts = value.split(".")
    return len(parts) == 3 and all(len(part) >= 10 for part in parts[:2])


def _candidate_bearer_tokens(storage: dict[str, Any]) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []

    def add(source: str, key: str, value: Any) -> None:
        if not isinstance(value, str) or not value:
            return
        lowered = key.lower()
        if "token" in lowered or "jwt" in lowered or _looks_like_jwt(value):
            candidates.append((f"{source}:{key}", value))
            return
        if value.startswith("{") and value.endswith("}"):
            try:
                payload = json.loads(value)
            except json.JSONDecodeError:
                return
            if isinstance(payload, dict):
                for inner_key, inner_value in payload.items():
                    add(source, f"{key}.{inner_key}", inner_value)

    for store_name in ("localStorage", "sessionStorage"):
        store = storage.get(store_name)
        if not isinstance(store, dict):
            continue
        for key, value in store.items():
            add(store_name, str(key), value)

    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source, token in candidates:
        if token not in seen:
            seen.add(token)
            unique.append((source, token))
    return unique


def _probe_bearer_tokens(
    cookies: list[dict[str, Any]],
    storage: dict[str, Any],
    url: str,
    *,
    timeout: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for source, token in _candidate_bearer_tokens(storage)[:8]:
        session = _session_from_cookies(cookies)
        session.headers["Authorization"] = f"Bearer {token}"
        result: dict[str, Any] = {
            "source": source,
            "ok": False,
            "status_code": None,
            "content_type": None,
            "body_preview": "",
            "error": None,
        }
        try:
            response = session.get(url, timeout=timeout)
        except requests.RequestException as exc:
            result["error"] = str(exc)
            results.append(result)
            continue
        result["ok"] = response.ok
        result["status_code"] = response.status_code
        result["content_type"] = response.headers.get("Content-Type")
        result["body_preview"] = response.text[:500]
        results.append(result)
        if response.ok:
            break
    return results


def _write_probe(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"logged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **result}, indent=2) + "\n")


def _storage_summary(storage: dict[str, Any]) -> dict[str, Any]:
    local = storage.get("localStorage") if isinstance(storage.get("localStorage"), dict) else {}
    session = storage.get("sessionStorage") if isinstance(storage.get("sessionStorage"), dict) else {}
    return {
        "href": storage.get("href"),
        "origin": storage.get("origin"),
        "localStorage_keys": sorted(local.keys()),
        "sessionStorage_keys": sorted(session.keys()),
    }


def _launch_browser(args: argparse.Namespace, url: str) -> None:
    browser_path = _find_browser_executable(args.browser_path)
    args.profile_dir.mkdir(parents=True, exist_ok=True)
    profile_arg = _wsl_to_windows_path(args.profile_dir.resolve())

    command = [
        browser_path,
        f"--remote-debugging-port={args.cdp_port}",
        f"--remote-debugging-address={args.cdp_bind_address}",
        "--remote-allow-origins=*",
        f"--user-data-dir={profile_arg}",
        "--no-first-run",
        "--new-window",
        url,
    ]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_browser_session(args: argparse.Namespace) -> int:
    open_url = args.open_url
    if args.open_inquiry:
        result = _basic_auth_probe(
            args.credentials,
            timeout=args.timeout,
            persona_template=args.persona_url_template,
        )
        _write_probe(args.probe_output, {"basic_auth": result})
        _print_basic_probe(result, args.persona_url_template)
        if result.get("biometric_url"):
            open_url = str(result["biometric_url"])
        elif result.get("persona_url"):
            open_url = str(result["persona_url"])
        elif result.get("inquiry"):
            open_url = _persona_inquiry_url(str(result["inquiry"]), args.persona_url_template)
        elif result.get("ok"):
            print("Direct Basic Auth already succeeded; opening platform page for cookie export.")
        else:
            print("No Persona inquiry was returned; opening platform page instead.")

    print(f"Opening controlled browser profile: {args.profile_dir}")
    print(f"URL: {open_url}")
    _launch_browser(args, open_url)

    cdp_host = _wait_for_cdp(args.cdp_port, timeout=args.timeout, hosts=_candidate_cdp_hosts(args.cdp_host))
    if cdp_host:
        print(f"DevTools reachable from WSL at {cdp_host}:{args.cdp_port}")
    elif _is_wsl():
        print(f"Will export cookies through Windows localhost:{args.cdp_port}")
    else:
        print("Could not reach Chrome DevTools; cannot export cookies.")
        return 2

    print("Complete login / face scan in the controlled browser if needed.")
    print("Before pressing Enter, make sure the controlled browser shows the Brain platform as logged in.")
    input("When the browser is authenticated, press Enter to export cookies/storage ... ")

    try:
        if cdp_host:
            all_cookies = _get_cdp_cookies(cdp_host, args.cdp_port)
            storage = _get_cdp_storage(cdp_host, args.cdp_port)
        else:
            all_cookies, storage = _get_windows_cdp_state(
                args.cdp_port,
                args.cookies_path,
                args.storage_path,
                timeout=args.timeout,
            )
    except Exception as exc:
        print(f"Browser export failed: {exc}")
        return 2

    cookies = _worldquant_cookies(all_cookies)
    _save_cookies(args.cookies_path, cookies)
    _save_storage(args.storage_path, storage)
    print(f"Saved {len(cookies)} WorldQuant cookies to {args.cookies_path}")
    summary = _storage_summary(storage)
    print(f"Saved browser storage to {args.storage_path}")
    print(f"Storage origin: {summary.get('origin')}  href: {summary.get('href')}")
    print(f"localStorage keys: {summary.get('localStorage_keys')}")
    print(f"sessionStorage keys: {summary.get('sessionStorage_keys')}")

    if args.api_probe_url:
        result = _probe_cookie_session(cookies, args.api_probe_url, timeout=args.timeout)
        bearer_results = _probe_bearer_tokens(
            cookies,
            storage,
            args.api_probe_url,
            timeout=args.timeout,
        )
        _write_probe(
            args.probe_output,
            {
                "cookie_probe": result,
                "bearer_token_probes": bearer_results,
                "storage_summary": summary,
            },
        )
        print(f"Cookie-session API probe status: {result.get('status_code')}")
        if result.get("error"):
            print(f"Cookie-session API probe error: {result['error']}")
        elif result.get("body_preview"):
            print(str(result["body_preview"]).replace("\n", " ")[:300])
        if bearer_results:
            for token_result in bearer_results:
                print(
                    "Bearer-token API probe "
                    f"{token_result.get('source')} status: {token_result.get('status_code')}"
                )
                if token_result.get("ok"):
                    print("Bearer-token API probe succeeded.")
                    return 0
        return 0 if result.get("ok") else 1

    return 0


def run_basic_probe(args: argparse.Namespace) -> int:
    result = _basic_auth_probe(
        args.credentials,
        timeout=args.timeout,
        persona_template=args.persona_url_template,
    )
    _write_probe(args.probe_output, {"basic_auth": result})
    _print_basic_probe(result, args.persona_url_template)
    return 0 if result.get("ok") else 1


def run_persona_flow(args: argparse.Namespace) -> int:
    username, password = load_credentials(args.credentials)
    session = requests.Session()
    basic_auth = HTTPBasicAuth(username, password)

    print("Starting one API auth POST. This should be the only POST in this flow.")
    result = _auth_post_probe(
        session,
        timeout=args.timeout,
        auth=basic_auth,
        persona_template=args.persona_url_template,
    )
    _write_probe(args.probe_output, {"initial_post": result})
    _print_basic_probe(result, args.persona_url_template)

    if result.get("ok"):
        print("API authentication succeeded on initial POST.")
        return 0

    if result.get("status_code") == 429:
        print("Authentication is currently throttled. Do not keep retrying.")
        return 1

    inquiry = result.get("inquiry")
    biometric_url = result.get("biometric_url")
    persona_url = result.get("persona_url")
    if not persona_url and inquiry:
        persona_url = _persona_inquiry_url(str(inquiry), args.persona_url_template)

    open_url = biometric_url or persona_url
    if not open_url:
        print("API did not return a Brain biometric URL or Persona inquiry.")
        return 1

    if biometric_url:
        print(f"Opening Brain biometric URL: {biometric_url}")
    else:
        print(f"Opening Persona URL: {persona_url}")
    if not _open_url(str(open_url)):
        print(f"Could not open automatically. Open manually: {open_url}")
    input("After face scan succeeds, press Enter. Do not rerun the script yet ... ")

    if args.settle_seconds > 0:
        print(f"Waiting {args.settle_seconds:.0f}s for WorldQuant/Persona status propagation ...")
        time.sleep(args.settle_seconds)

    checks: list[dict[str, Any]] = []
    poll = max(args.status_poll_seconds, 1.0)
    if not biometric_url:
        for finalize_post in range(1, args.finalize_max_posts + 1):
            status = _auth_post_probe(
                session,
                timeout=args.timeout,
                auth=basic_auth,
                kind="auth_after_biometric",
                persona_template=args.persona_url_template,
            )
            checks.append(status)
            print(f"POST Brain auth finalization: {status.get('status_code')}")
            if status.get("error"):
                print(f"Brain auth finalization error: {status['error']}")
                break
            if status.get("ok"):
                _write_probe(
                    args.probe_output,
                    {"initial_post": result, "status_checks": checks, "authenticated": True},
                )
                print("Authentication completed through Brain after Persona scan.")
                return 0
            if status.get("status_code") == 429:
                print("Brain auth finalization is rate-limited; stopping.")
                break
            next_inquiry = status.get("inquiry")
            if next_inquiry and next_inquiry != inquiry:
                print("Brain returned a new Persona inquiry; stopping to avoid an inquiry loop.")
                if status.get("persona_url"):
                    print(f"New Persona URL: {status['persona_url']}")
                break
            if finalize_post < args.finalize_max_posts:
                time.sleep(poll)

        _write_probe(
            args.probe_output,
            {"initial_post": result, "status_checks": checks, "authenticated": False},
        )
        print("Brain auth finalization did not show an authenticated API session.")
        return 1

    deadline = time.monotonic() + args.status_timeout
    status_posts = 0
    while True:
        if status_posts >= args.status_max_posts:
            print(f"Stopped after {args.status_max_posts} biometric status POSTs.")
            break
        status_posts += 1
        status = _post_existing_auth_url(session, str(biometric_url), timeout=args.timeout)
        checks.append(status)
        print(f"POST biometric status: {status.get('status_code')}")
        if status.get("error"):
            print(f"POST biometric error: {status['error']}")
            break
        if status.get("ok"):
            _write_probe(
                args.probe_output,
                {"initial_post": result, "status_checks": checks, "authenticated": True},
            )
            print("Authentication completed through the existing biometric URL.")
            return 0
        if status.get("status_code") == 429:
            print("Biometric status check is rate-limited; stopping.")
            _write_probe(
                args.probe_output,
                {"initial_post": result, "status_checks": checks, "authenticated": False},
            )
            return 1
        next_url = status.get("biometric_url")
        if next_url and next_url != biometric_url:
            biometric_url = str(next_url)
            print(f"Brain returned a follow-up biometric URL: {biometric_url}")

        if time.monotonic() >= deadline:
            break
        time.sleep(poll)

    _write_probe(
        args.probe_output,
        {"initial_post": result, "status_checks": checks, "authenticated": False},
    )
    print("Status checks did not show an authenticated API session.")
    print("No additional POST was sent to /authentication, so this flow should not create fresh inquiries.")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Browser-session auth helper for WorldQuant Brain.")
    parser.add_argument(
        "--mode",
        choices=("browser", "basic-probe", "persona"),
        default="browser",
        help=(
            "browser exports/test browser cookies; basic-probe checks direct API Basic Auth once; "
            "persona does one /authentication POST, opens face scan, then polls the returned biometric URL."
        ),
    )
    parser.add_argument("--credentials", default="brain.txt")
    parser.add_argument("--browser-path", default="")
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--cookies-path", type=Path, default=DEFAULT_COOKIES_PATH)
    parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    parser.add_argument("--probe-output", type=Path, default=DEFAULT_PROBE_PATH)
    parser.add_argument("--open-url", default=DEFAULT_PLATFORM_URL)
    parser.add_argument(
        "--open-inquiry",
        action="store_true",
        help="Call Basic Auth once and open the returned Persona inquiry in the controlled browser.",
    )
    parser.add_argument("--persona-url-template", default=DEFAULT_PERSONA_URL_TEMPLATE)
    parser.add_argument("--api-probe-url", default=DEFAULT_API_PROBE_URL)
    parser.add_argument("--cdp-port", type=int, default=9223)
    parser.add_argument("--cdp-host", default="")
    parser.add_argument("--cdp-bind-address", default="0.0.0.0")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--settle-seconds", type=float, default=60.0)
    parser.add_argument("--status-timeout", type=float, default=180.0)
    parser.add_argument("--status-poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--status-max-posts",
        type=int,
        default=3,
        help="Maximum POSTs to the existing biometric URL after face scan.",
    )
    parser.add_argument(
        "--finalize-max-posts",
        type=int,
        default=1,
        help="Maximum Brain /authentication POSTs after completing an inquiry-only Persona scan.",
    )
    parser.add_argument(
        "--show-budget",
        action="store_true",
        help="Show the local daily auth-attempt budget and exit without API calls.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.show_budget:
        print(json.dumps(auth_attempt_budget_status(), indent=2, sort_keys=True))
        return 0
    if args.mode == "basic-probe":
        return run_basic_probe(args)
    if args.mode == "persona":
        return run_persona_flow(args)
    return run_browser_session(args)


if __name__ == "__main__":
    raise SystemExit(main())
