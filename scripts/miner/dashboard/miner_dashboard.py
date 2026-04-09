#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import re
import subprocess
from urllib.parse import parse_qs, urlparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PM2_APP_NAME = os.getenv("PM2_APP_NAME", "poker44_miner")
PM2_LOG_DIR = Path(os.getenv("PM2_LOG_DIR", str(Path.home() / ".pm2" / "logs")))

def _pm2_log_paths(app_name: str) -> tuple[Path, Path]:
    # PM2 log filenames normalize app names (underscores become hyphens).
    normalized = (app_name or "").replace("_", "-")
    out_log = PM2_LOG_DIR / f"{normalized}-out.log"
    err_log = PM2_LOG_DIR / f"{normalized}-error.log"
    return out_log, err_log

REQUEST_PATTERNS = [
    re.compile(r"DetectionSynapse", re.IGNORECASE),
    re.compile(r"UnknownSynapseError", re.IGNORECASE),
    re.compile(r"Validator request received", re.IGNORECASE),
    re.compile(r"Scored\s+\d+\s+chunks", re.IGNORECASE),
    re.compile(r"blacklist", re.IGNORECASE),
    re.compile(r"validator", re.IGNORECASE),
]

OUTPUT_PATTERNS = [
    re.compile(r"risk_scores?", re.IGNORECASE),
    re.compile(r"predictions?", re.IGNORECASE),
    re.compile(r"predctions", re.IGNORECASE),  # current miner log typo: "Miner Predctions"
    re.compile(r"Validator response sent", re.IGNORECASE),
    re.compile(r"Scored\s+\d+\s+chunks", re.IGNORECASE),
    re.compile(r"forward", re.IGNORECASE),
    re.compile(r"Miner UID", re.IGNORECASE),
    re.compile(r"Axon served", re.IGNORECASE),
]

LOG_PARSE_RE = re.compile(
    r"^\s*(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+\|\s+"
    r"(?P<level>[A-Z]+)\s+\|\s+"
    r"(?P<src>[^|]+)\|\s+"
    r"(?P<msg>.*)$"
)
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _read_tail(path: Path, max_lines: int = 3000) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return list(deque(f, maxlen=max_lines))


def _match_lines(lines: list[str], patterns: list[re.Pattern], limit: int = 50) -> list[str]:
    matched: list[str] = []
    for line in reversed(lines):
        stripped = _strip_ansi(line)
        if any(p.search(stripped) for p in patterns):
            matched.append(stripped.rstrip())
        if len(matched) >= limit:
            break
    return list(reversed(matched))


def _count_matches(lines: list[str], patterns: list[re.Pattern]) -> int:
    count = 0
    for line in lines:
        stripped = _strip_ansi(line)
        if any(p.search(stripped) for p in patterns):
            count += 1
    return count


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _is_noise_line(text: str) -> bool:
    low = text.lower()
    noisy_tokens = (
        "allowed_validator_hotkeys:",
        "project_name:",
        "num_concurrent_forwards:",
        "force_validator_permit:",
        "blacklist:",
    )
    return any(token in low for token in noisy_tokens)


def _normalize_message(text: str) -> str:
    msg = text.strip()
    msg = re.sub(r"UnknownSynapseError#[a-f0-9-]+", "UnknownSynapseError", msg)
    msg = re.sub(r"\s+", " ", msg)
    return msg


def _collapse_entries(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    if not entries:
        return entries
    out: list[dict[str, str]] = []
    for item in entries:
        if (
            out
            and out[-1]["level"] == item["level"]
            and out[-1]["source"] == item["source"]
            and out[-1]["message"] == item["message"]
        ):
            count = int(out[-1].get("count", "1")) + 1
            out[-1]["count"] = str(count)
            out[-1]["timestamp"] = item["timestamp"]
        else:
            clone = dict(item)
            clone["count"] = "1"
            out.append(clone)
    return out


def _to_human_entries(lines: list[str], limit: int = 20) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for line in reversed(lines):
        text = _strip_ansi(line).strip()
        if not text:
            continue
        if _is_noise_line(text):
            continue
        m = LOG_PARSE_RE.match(text)
        if m:
            message = _normalize_message(m.group("msg"))
            entries.append(
                {
                    "timestamp": m.group("ts"),
                    "level": m.group("level"),
                    "source": m.group("src").strip(),
                    "message": message,
                }
            )
        else:
            entries.append(
                {"timestamp": "-", "level": "INFO", "source": "-", "message": _normalize_message(text)}
            )
        if len(entries) >= limit:
            break
    return _collapse_entries(list(reversed(entries)))


def _pm2_status(app_name: str) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["pm2", "jlist"],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout or "[]")
    except Exception as e:
        return {
            "ok": False,
            "error": f"pm2 query failed: {e}",
            "status": "unknown",
            "pid": None,
            "restarts": None,
            "uptime_seconds": None,
            "memory_mb": None,
            "cpu_percent": None,
        }

    target = None
    for item in data:
        if item.get("name") == app_name:
            target = item
            break

    if not target:
        return {
            "ok": False,
            "error": f"pm2 app '{app_name}' not found",
            "status": "not_found",
            "pid": None,
            "restarts": None,
            "uptime_seconds": None,
            "memory_mb": None,
            "cpu_percent": None,
        }

    env = target.get("pm2_env", {})
    monit = target.get("monit", {})
    status = env.get("status", "unknown")
    pm_uptime = env.get("pm_uptime")
    now_ms = int(dt.datetime.now().timestamp() * 1000)
    uptime_seconds = None
    if isinstance(pm_uptime, (int, float)):
        uptime_seconds = max(0, int((now_ms - pm_uptime) / 1000))

    mem = monit.get("memory")
    memory_mb = round(mem / (1024 * 1024), 2) if isinstance(mem, (int, float)) else None

    return {
        "ok": True,
        "error": None,
        "status": status,
        "pid": target.get("pid"),
        "restarts": env.get("restart_time"),
        "uptime_seconds": uptime_seconds,
        "uptime_hms": _fmt_hms(uptime_seconds),
        "memory_mb": memory_mb,
        "cpu_percent": monit.get("cpu"),
    }

def _pm2_app_names() -> list[str]:
    """Return all PM2 process names."""
    try:
        proc = subprocess.run(
            ["pm2", "jlist"],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout or "[]")
    except Exception:
        return []
    out: list[str] = []
    for item in data:
        name = item.get("name")
        if isinstance(name, str) and name:
            out.append(name)
    return out


def _default_miner_choices() -> list[str]:
    """Heuristic: show miners that look like Poker44 miners."""
    names = _pm2_app_names()
    miners = [n for n in names if "poker44" in n.lower() and "dashboard" not in n.lower()]
    # Prefer canonical names first if present.
    preferred = []
    for n in ("poker44_miner", "poker44-kevin"):
        if n in miners:
            preferred.append(n)
    for n in miners:
        if n not in preferred:
            preferred.append(n)
    return preferred or [PM2_APP_NAME]


def _fmt_hms(seconds: int | None) -> str:
    if seconds is None:
        return "-"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_status(app_name: str) -> dict[str, Any]:
    pm2 = _pm2_status(app_name)
    out_log, err_log = _pm2_log_paths(app_name)
    out_lines = _read_tail(out_log)
    err_lines = _read_tail(err_log)
    merged_lines = out_lines + err_lines

    validator_requests = _match_lines(merged_lines, REQUEST_PATTERNS, limit=120)
    miner_outputs = _match_lines(out_lines, OUTPUT_PATTERNS, limit=120)
    recent_errors = [l.rstrip() for l in err_lines[-30:]]
    validator_request_count = _count_matches(merged_lines, REQUEST_PATTERNS)
    miner_output_count = _count_matches(out_lines, OUTPUT_PATTERNS)

    health = "healthy" if pm2.get("status") == "online" else "unhealthy"
    if pm2.get("restarts") and pm2.get("restarts", 0) > 5:
        health = "degraded"

    return {
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "app_name": app_name,
        "health": health,
        "pm2": pm2,
        "validator_request_count": validator_request_count,
        "validator_requests": validator_requests[-20:],
        "validator_request_entries": _to_human_entries(validator_requests, limit=20),
        "miner_output_count": miner_output_count,
        "miner_outputs": miner_outputs[-20:],
        "miner_output_entries": _to_human_entries(miner_outputs, limit=20),
        "recent_errors": recent_errors[-20:],
        "log_paths": {"out": str(out_log), "err": str(err_log)},
    }


INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Poker44 Miner Dashboard</title>
  <style>
    :root { --bg:#0b1020; --card:#131a2e; --fg:#e7ecff; --muted:#8ea0d4; --ok:#39d98a; --bad:#ff5c7c; --warn:#ffbf5f; }
    body { margin:0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background:var(--bg); color:var(--fg); }
    .wrap { max-width: 1180px; margin: 0 auto; padding: 16px; }
    .grid { display:grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap:12px; }
    .card { background:var(--card); border:1px solid #2a355b; border-radius:10px; padding:12px; }
    .k { color:var(--muted); font-size:12px; }
    .v { font-size:24px; margin-top:4px; }
    .ok { color:var(--ok); } .bad { color:var(--bad); } .warn { color:var(--warn); }
    h2 { margin:18px 0 8px 0; font-size:15px; color:#b7c6f7; }
    pre { background:#0d1326; border:1px solid #2a355b; border-radius:10px; padding:10px; overflow:auto; max-height:280px; white-space:pre-wrap; }
    .row { display:grid; grid-template-columns: 1fr 1fr; gap:12px; }
    .logbox {
      background:#0d1326;
      border:1px solid #2a355b;
      border-radius:10px;
      padding:8px;
      height:340px;
      max-height:340px;
      overflow-y:scroll; /* Always show vertical scrollbar for discoverability. */
      overflow-x:hidden;
      scrollbar-width: thin;
      scrollbar-color: #4a5c92 #0d1326;
    }
    .logbox::-webkit-scrollbar { width: 10px; }
    .logbox::-webkit-scrollbar-track { background: #0d1326; border-radius: 8px; }
    .logbox::-webkit-scrollbar-thumb { background: #4a5c92; border-radius: 8px; border: 2px solid #0d1326; }
    .logbox::-webkit-scrollbar-thumb:hover { background: #6f84c8; }
    .logrow { border-bottom:1px solid #1f2a4a; padding:7px 4px; }
    .logrow:last-child { border-bottom:none; }
    .meta { color:var(--muted); font-size:11px; margin-bottom:3px; }
    .msg { font-size:12px; line-height:1.35; white-space: pre-wrap; word-break: break-word; }
    .lvl-ERROR { color:var(--bad); }
    .lvl-WARNING { color:var(--warn); }
    .lvl-DEBUG { color:#8db8ff; }
    .lvl-INFO { color:#d9e5ff; }
    @media (max-width: 980px) { .grid { grid-template-columns: 1fr 1fr; } .row { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Poker44 Miner Dashboard</h1>
    <div class="card" style="margin-bottom:12px">
      <div class="k">Select miner</div>
      <div style="margin-top:6px; display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
        <select id="minerSelect" style="background:#0d1326;color:var(--fg);border:1px solid #2a355b;border-radius:8px;padding:8px 10px;min-width:260px;"></select>
        <div class="k">Logs: <span id="logPaths"></span></div>
      </div>
    </div>
    <div class="grid">
      <div class="card"><div class="k">Health</div><div id="health" class="v">-</div></div>
      <div class="card"><div class="k">PM2 Status</div><div id="status" class="v">-</div></div>
      <div class="card"><div class="k">Validator Requests</div><div id="reqCount" class="v">-</div></div>
      <div class="card"><div class="k">Miner Output Events</div><div id="outCount" class="v">-</div></div>
    </div>
    <div class="grid" style="margin-top:12px">
      <div class="card"><div class="k">PID</div><div id="pid" class="v" style="font-size:18px">-</div></div>
      <div class="card"><div class="k">Uptime (hh:mm:ss)</div><div id="uptime" class="v" style="font-size:18px">-</div></div>
      <div class="card"><div class="k">Restarts</div><div id="restarts" class="v" style="font-size:18px">-</div></div>
      <div class="card"><div class="k">Mem / CPU</div><div id="resources" class="v" style="font-size:18px">-</div></div>
    </div>
    <div class="row">
      <div>
        <h2>Validator Requests (latest)</h2>
        <div id="requests" class="logbox"></div>
      </div>
      <div>
        <h2>Miner Output (latest)</h2>
        <div id="outputs" class="logbox"></div>
      </div>
    </div>
    <h2>Recent Errors</h2>
    <pre id="errors"></pre>
    <div class="k" id="ts"></div>
  </div>
  <script>
    const setText = (id, v) => document.getElementById(id).textContent = v;
    const setClass = (id, cls) => document.getElementById(id).className = 'v ' + cls;
    const esc = (s) => String(s ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
    let currentApp = null;

    async function loadApps() {
      const r = await fetch('/api/apps');
      const d = await r.json();
      const apps = d.apps || [];
      const sel = document.getElementById('minerSelect');
      sel.innerHTML = apps.map(a => `<option value=\"${esc(a)}\">${esc(a)}</option>`).join('');
      currentApp = d.default_app || apps[0] || null;
      if (currentApp) sel.value = currentApp;
      sel.addEventListener('change', () => { currentApp = sel.value; refresh(); });
    }

    function renderEntries(id, entries, emptyText) {
      const root = document.getElementById(id);
      if (!entries || entries.length === 0) {
        root.innerHTML = `<div class="logrow"><div class="meta">-</div><div class="msg">${esc(emptyText)}</div></div>`;
        return;
      }
      root.innerHTML = entries.map(e => {
        const lvl = esc(e.level || 'INFO');
        const countBadge = Number(e.count || 1) > 1 ? ` <span class="lvl-INFO">x${esc(e.count)}</span>` : '';
        return `<div class="logrow">
          <div class="meta">${esc(e.timestamp)} | <span class="lvl-${lvl}">${lvl}</span> | ${esc(e.source)}</div>
          <div class="msg">${esc(e.message)}${countBadge}</div>
        </div>`;
      }).join('');
    }
    async function refresh() {
      try {
        const app = currentApp ? `?app=${encodeURIComponent(currentApp)}` : '';
        const r = await fetch('/api/status' + app);
        const d = await r.json();
        setText('health', d.health);
        setClass('health', d.health === 'healthy' ? 'ok' : (d.health === 'degraded' ? 'warn' : 'bad'));
        setText('status', d.pm2.status);
        setClass('status', d.pm2.status === 'online' ? 'ok' : 'bad');
        setText('reqCount', String(d.validator_request_count));
        setText('outCount', String(d.miner_output_count));
        setText('pid', String(d.pm2.pid ?? '-'));
        setText('uptime', String(d.pm2.uptime_hms ?? '-'));
        setText('restarts', String(d.pm2.restarts ?? '-'));
        setText('resources', `${d.pm2.memory_mb ?? '-'} MB / ${d.pm2.cpu_percent ?? '-'}%`);
        renderEntries('requests', d.validator_request_entries || [], '(no matching request logs yet)');
        renderEntries('outputs', d.miner_output_entries || [], '(no output events yet)');
        setText('errors', (d.recent_errors || []).join('\\n') || '(no recent errors)');
        setText('ts', `last update: ${d.timestamp}`);
        if (d.log_paths) {
          setText('logPaths', `${d.log_paths.out} | ${d.log_paths.err}`);
        }
      } catch (e) {
        setText('ts', 'dashboard fetch error: ' + e);
      }
    }
    loadApps().then(() => { refresh(); setInterval(refresh, 2000); });
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query or "")
        if self.path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/apps":
            apps = _default_miner_choices()
            payload = json.dumps(
                {"apps": apps, "default_app": apps[0] if apps else PM2_APP_NAME}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if path == "/api/status":
            app = (qs.get("app") or [PM2_APP_NAME])[0]
            payload = json.dumps(build_status(app)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Poker44 miner live dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10298)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Dashboard listening on http://{args.host}:{args.port}")
    print(f"Watching PM2 app: {PM2_APP_NAME}")
    out_log, err_log = _pm2_log_paths(PM2_APP_NAME)
    print(f"Logs (default selection): {out_log} | {err_log}")
    server.serve_forever()


if __name__ == "__main__":
    main()
