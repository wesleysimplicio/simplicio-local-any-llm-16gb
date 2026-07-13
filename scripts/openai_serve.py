"""US4 V6 Apple Edition - local OpenAI-compatible server.

Runs a chat backend (MLX or Ollama) and an embeddings backend behind a single
HTTP endpoint so any OpenAI client (simplicio-cli, langchain, raw curl, etc.)
can hit one base URL:

    /v1/chat/completions    -> chat backend (mlx-lm subprocess OR ollama proxy)
    /v1/completions         -> chat backend (proxied)
    /v1/models              -> aggregated list (chat + embedding)
    /v1/embeddings          -> EmbeddingGemma via mlx-embeddings (in-process)
    /health                 -> liveness probe

Design intent: stdlib only (no FastAPI, no uvicorn). One file.

Chat backend selection (`US4_SERVE_CHAT_BACKEND`):

  - `mlx` (default): delegates to the official `mlx_lm.server` shipped with
    mlx-lm >= 0.31.0. We boot it as a child process on PORT + 1 and reverse-
    proxy /v1/chat/completions and /v1/completions to it.
  - `ollama`: assumes a running Ollama daemon (default 127.0.0.1:11434) and
    reverse-proxies the same routes to its native OpenAI-compat endpoint at
    `/v1/...`. No subprocess is spawned.
  - any other value: treated as a generic OpenAI-compatible HTTP upstream;
    set `US4_SERVE_CHAT_UPSTREAM` to the base URL (no trailing /v1).

The embeddings path is always served locally by mlx-embeddings because it
does not ship a server.

Environment knobs (all optional):

    US4_SERVE_HOST            default 127.0.0.1
    US4_SERVE_PORT            default 8080
    US4_SERVE_CHAT_BACKEND    default mlx; valid: mlx, ollama, custom
    US4_SERVE_CHAT_UPSTREAM   override upstream base URL (e.g.
                              http://127.0.0.1:11434). Defaults derived from
                              the selected backend.
    US4_SERVE_CHAT_MODEL      default mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
                              (or `openbmb/minicpm5:latest` when backend=ollama).
    US4_SERVE_EMBED_MODEL     default mlx-community/embeddinggemma-300m-bf16
    US4_SERVE_DISABLE_CHAT    truthy -> skip chat backend entirely
    US4_SERVE_DISABLE_EMBED   truthy -> skip embeddings handler
    US4_SERVE_LOG_LEVEL       default INFO

    RAM-tuning passthroughs (forwarded to mlx_lm.server when backend=mlx;
    silently ignored for backend=ollama or custom upstreams):

    US4_SERVE_PROMPT_CACHE_BYTES  cap KV cache memory (e.g. 536870912 = 512 MiB).
                                  Validated as a positive integer at startup;
                                  non-numeric values are logged and skipped.
    US4_SERVE_MLX_EXTRA_ARGS      raw extra args appended to the mlx_lm.server
                                  command line (shell-style split). Escape
                                  hatch for any flag not exposed individually
                                  (e.g. "--max-tokens 256 --prefill-step-size 512").
                                  Tokens starting with --host, --port, or
                                  --cors* (and their values when in
                                  "--flag VALUE" form) are stripped before
                                  forwarding to prevent accidental override of
                                  the loopback bind. Network binding stays
                                  fixed at 127.0.0.1 (or US4_SERVE_HOST,
                                  loopback-validated).

Exit codes:

    0  graceful shutdown
    1  fatal startup error
    2  mlx-lm not installed and chat backend=mlx and chat not disabled
    3  mlx-embeddings not installed and embeddings not disabled
"""

from __future__ import annotations

import atexit
import ipaddress
import json
import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

LOG = logging.getLogger("us4.serve")

DEFAULT_CHAT_MODEL_MLX = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
DEFAULT_CHAT_MODEL_OLLAMA = "openbmb/minicpm5:latest"
DEFAULT_EMBED_MODEL = "mlx-community/embeddinggemma-300m-bf16"
DEFAULT_OLLAMA_UPSTREAM = "http://127.0.0.1:11434"
SUPPORTED_CHAT_BACKENDS = ("mlx", "ollama", "custom")
MAX_BODY_BYTES = 8 * 1024 * 1024
UPSTREAM_TIMEOUT_S = 120


def _sanitize_path(path: str) -> str:
    return path.replace("\n", "\\n").replace("\r", "\\r")[:120]


def _truthy(value: Optional[str]) -> bool:
    if not value:
        return False
    return value.lower() not in {"0", "false", "no", "off", ""}


class Settings:
    host: str = os.environ.get("US4_SERVE_HOST", "127.0.0.1")
    port: int = int(os.environ.get("US4_SERVE_PORT", "8080"))
    chat_backend: str = os.environ.get(
        "US4_SERVE_CHAT_BACKEND", "mlx"
    ).strip().lower() or "mlx"
    chat_upstream_override: str = os.environ.get(
        "US4_SERVE_CHAT_UPSTREAM", ""
    ).strip()
    embed_model: str = os.environ.get("US4_SERVE_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    disable_chat: bool = _truthy(os.environ.get("US4_SERVE_DISABLE_CHAT"))
    disable_embed: bool = _truthy(os.environ.get("US4_SERVE_DISABLE_EMBED"))
    log_level: str = os.environ.get("US4_SERVE_LOG_LEVEL", "INFO").upper()
    # RAM-tuning knobs passed straight through to mlx_lm.server when
    # `chat_backend == "mlx"`. Empty values are skipped so the upstream
    # uses its own defaults.
    prompt_cache_bytes: str = os.environ.get(
        "US4_SERVE_PROMPT_CACHE_BYTES", ""
    ).strip()
    mlx_extra_args: str = os.environ.get(
        "US4_SERVE_MLX_EXTRA_ARGS", ""
    ).strip()

    @property
    def chat_model(self) -> str:
        explicit = os.environ.get("US4_SERVE_CHAT_MODEL")
        if explicit:
            return explicit
        if self.chat_backend == "ollama":
            return DEFAULT_CHAT_MODEL_OLLAMA
        return DEFAULT_CHAT_MODEL_MLX

    @property
    def upstream_port(self) -> int:
        return self.port + 1

    @property
    def upstream_url(self) -> str:
        if self.chat_upstream_override:
            return self.chat_upstream_override.rstrip("/")
        if self.chat_backend == "ollama":
            return DEFAULT_OLLAMA_UPSTREAM
        # mlx and unknown backends without override: spawn-and-proxy on +1
        return f"http://127.0.0.1:{self.upstream_port}"

    @property
    def upstream_health_url(self) -> str:
        # mlx_lm.server exposes /health; Ollama exposes /api/tags as a cheap
        # liveness check (/health is not standardized there).
        if self.chat_backend == "ollama":
            return f"{self.upstream_url}/api/tags"
        return f"{self.upstream_url}/health"

    @property
    def spawns_chat_subprocess(self) -> bool:
        # Only the mlx backend boots a child process; ollama and `custom`
        # expect an already-running upstream.
        return (not self.disable_chat) and self.chat_backend == "mlx"


SETTINGS = Settings()


class EmbeddingsBackend:
    """In-process EmbeddingGemma via mlx-embeddings. Lazy-loaded."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None

    def ensure_loaded(self) -> None:
        # No fast path outside the lock: with ThreadingHTTPServer two requests
        # can race past the None check and trigger duplicate model loads
        # (catastrophic on unified memory). The lock cost is negligible after
        # the model is loaded once.
        with self._lock:
            if self._model is not None:
                return
            try:
                from mlx_embeddings import load as _load
            except ImportError as exc:
                raise RuntimeError(
                    "mlx-embeddings is not installed. "
                    "Run: pip install -r scripts/requirements-serve.txt"
                ) from exc
            LOG.info("loading embedding model: %s", self._model_id)
            t0 = time.monotonic()
            self._model, self._tokenizer = _load(self._model_id)
            LOG.info("embedding model loaded in %.2fs", time.monotonic() - t0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.ensure_loaded()
        # mlx-embeddings 0.1.0 ships a `generate()` helper that calls
        # `model(**inputs)` with keys `input_ids` / `attention_mask`. Some model
        # classes (gemma3_text, others) expect `inputs=` instead, raising
        # TypeError. Tokenize here and dispatch with the signature the loaded
        # model accepts so the backend stays model-agnostic.
        batch = self._tokenizer(
            texts,
            return_tensors="mlx",
            padding=True,
            truncation=True,
            max_length=512,
        )
        ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        try:
            output = self._model(inputs=ids, attention_mask=attention_mask)
        except TypeError:
            output = self._model(input_ids=ids, attention_mask=attention_mask)
        return self._extract_vectors(output)

    @staticmethod
    def _extract_vectors(output: Any) -> list[list[float]]:
        candidate = output
        for attr in ("text_embeds", "sentence_embeddings", "embeddings", "pooler_output"):
            if hasattr(candidate, attr):
                candidate = getattr(candidate, attr)
                break
        if hasattr(candidate, "tolist"):
            data = candidate.tolist()
        else:
            data = list(candidate)
        if data and not isinstance(data[0], list):
            return [list(map(float, data))]
        return [[float(x) for x in row] for row in data]


EMBED_BACKEND: Optional[EmbeddingsBackend] = None


def _apply_cors(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")


def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    _apply_cors(handler)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _send_error(handler: BaseHTTPRequestHandler, status: int, message: str, code: str = "invalid_request_error") -> None:
    _send_json(handler, status, {"error": {"message": message, "type": code}})


def _proxy_upstream(handler: BaseHTTPRequestHandler, path: str, raw_body: bytes) -> None:
    if SETTINGS.disable_chat:
        _send_error(handler, 503, "chat backend disabled (US4_SERVE_DISABLE_CHAT)", "service_unavailable")
        return
    url = f"{SETTINGS.upstream_url}{path}"
    method = handler.command
    headers = {"Content-Type": handler.headers.get("Content-Type", "application/json")}
    accept = handler.headers.get("Accept")
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, data=raw_body if raw_body else None, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT_S) as resp:
            handler.send_response(resp.status)
            content_type = resp.headers.get("Content-Type", "application/json")
            handler.send_header("Content-Type", content_type)
            _apply_cors(handler)
            transfer = resp.headers.get("Transfer-Encoding")
            content_length = resp.headers.get("Content-Length")
            if transfer:
                handler.send_header("Transfer-Encoding", transfer)
            elif content_length:
                handler.send_header("Content-Length", content_length)
            handler.end_headers()
            try:
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    handler.wfile.write(chunk)
                    handler.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                LOG.info("client disconnected mid-stream")
                return
    except urllib.error.HTTPError as exc:
        body = exc.read()
        handler.send_response(exc.code)
        handler.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
        _apply_cors(handler)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        try:
            handler.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return
    except urllib.error.URLError as exc:
        _send_error(handler, 502, f"chat upstream unreachable: {exc.reason}", "bad_gateway")


def _build_models_payload() -> dict:
    now = int(time.time())
    data: list[dict] = []
    if not SETTINGS.disable_chat:
        data.append({"id": SETTINGS.chat_model, "object": "model", "created": now, "owned_by": "us4-local"})
    if not SETTINGS.disable_embed:
        data.append({"id": SETTINGS.embed_model, "object": "model", "created": now, "owned_by": "us4-local"})
    return {"object": "list", "data": data}


def _handle_embeddings(handler: BaseHTTPRequestHandler, body: Optional[dict]) -> None:
    if SETTINGS.disable_embed:
        _send_error(handler, 503, "embedding backend disabled (US4_SERVE_DISABLE_EMBED)", "service_unavailable")
        return
    if not body:
        _send_error(handler, 400, "request body must be JSON with 'input' field")
        return
    raw_input = body.get("input")
    if raw_input is None:
        _send_error(handler, 400, "missing 'input' field")
        return
    if isinstance(raw_input, str):
        texts = [raw_input]
    elif isinstance(raw_input, list) and all(isinstance(item, str) for item in raw_input):
        texts = list(raw_input)
    else:
        _send_error(handler, 400, "'input' must be string or list of strings")
        return
    requested_model = body.get("model") or SETTINGS.embed_model
    if EMBED_BACKEND is None:
        _send_error(handler, 503, "embedding backend not initialized", "service_unavailable")
        return
    try:
        vectors = EMBED_BACKEND.embed(texts)
    except Exception:
        LOG.exception("embedding failure")
        _send_error(handler, 500, "embedding failed", "internal_error")
        return
    data = [
        {"object": "embedding", "index": idx, "embedding": vec}
        for idx, vec in enumerate(vectors)
    ]
    total_tokens = sum(len(t.split()) for t in texts)
    _send_json(
        handler,
        200,
        {
            "object": "list",
            "data": data,
            "model": requested_model,
            "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        },
    )


class Us4Handler(BaseHTTPRequestHandler):
    server_version = "us4-serve/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        LOG.info("%s %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/v1/health"):
            _send_json(self, 200, {"status": "ok", "chat": not SETTINGS.disable_chat, "embed": not SETTINGS.disable_embed})
            return
        if path in ("/v1/models", "/models"):
            _send_json(self, 200, _build_models_payload())
            return
        _send_error(self, 404, f"not found: {_sanitize_path(path)}", "not_found")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        _apply_cors(self)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header) if length_header else 0
        except ValueError:
            length = 0
        if length < 0:
            _send_error(self, 400, "invalid Content-Length")
            return
        if length > MAX_BODY_BYTES:
            _send_error(self, 413, "request body too large", "payload_too_large")
            return
        raw_body = self.rfile.read(length) if length > 0 else b""
        body: Optional[dict] = None
        if raw_body:
            try:
                body = json.loads(raw_body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = None
        if path in ("/v1/embeddings", "/embeddings"):
            _handle_embeddings(self, body)
            return
        if path in ("/v1/chat/completions", "/chat/completions", "/v1/completions", "/completions"):
            _proxy_upstream(self, path, raw_body)
            return
        _send_error(self, 404, f"not found: {_sanitize_path(path)}", "not_found")


def _spawn_upstream() -> Optional[subprocess.Popen]:
    if SETTINGS.disable_chat:
        LOG.info("chat backend disabled; skipping subprocess")
        return None
    if SETTINGS.chat_backend != "mlx":
        LOG.info(
            "chat backend=%s; expecting already-running upstream at %s",
            SETTINGS.chat_backend,
            SETTINGS.upstream_url,
        )
        return None
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        LOG.error("mlx-lm is not installed. Run: pip install -r scripts/requirements-serve.txt")
        sys.exit(2)
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "server",
        "--model",
        SETTINGS.chat_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(SETTINGS.upstream_port),
        "--log-level",
        SETTINGS.log_level,
    ]
    # Optional RAM-tuning passthroughs. These are appended only when set so
    # the default behaviour stays identical for callers that do not opt in.
    if SETTINGS.prompt_cache_bytes:
        if SETTINGS.prompt_cache_bytes.isdigit():
            cmd.extend(["--prompt-cache-bytes", SETTINGS.prompt_cache_bytes])
        else:
            LOG.error(
                "ignoring US4_SERVE_PROMPT_CACHE_BYTES=%r: expected a positive"
                " integer number of bytes (e.g. 268435456 for 256 MiB)",
                SETTINGS.prompt_cache_bytes,
            )
    if SETTINGS.mlx_extra_args:
        try:
            extra = shlex.split(SETTINGS.mlx_extra_args)
        except ValueError as exc:
            LOG.error(
                "invalid US4_SERVE_MLX_EXTRA_ARGS (%s); ignoring: %s",
                SETTINGS.mlx_extra_args,
                exc,
            )
        else:
            # Reject network-binding flags. mlx_lm.server uses argparse, which
            # honours the *last* occurrence of a flag, so a user copy-pasting
            # the recipe from the README onto a cloud VM with --host 0.0.0.0
            # in MLX_EXTRA_ARGS would silently expose the unauthenticated LLM
            # endpoint to the network. Same applies to --port and --cors* —
            # serve-shape is fixed by the facade, not by the upstream.
            #
            # The loop peeks one token ahead so the *value* that follows a
            # blocked flag in `--flag VALUE` form is dropped together with
            # the flag itself; otherwise the orphan value would land as a
            # positional argument and crash upstream argparse.
            blocked_prefixes = ("--host", "--port", "--cors")
            safe: list[str] = []
            rejected: list[str] = []
            i = 0
            while i < len(extra):
                token = extra[i]
                # Case-insensitive match so --HOST / --Port variants cannot
                # smuggle past the filter even if the upstream argparse later
                # gains case-folded flags.
                if token.lower().startswith(blocked_prefixes):
                    rejected.append(token)
                    has_inline_value = "=" in token
                    next_token = extra[i + 1] if i + 1 < len(extra) else None
                    if (
                        not has_inline_value
                        and next_token is not None
                        and not next_token.startswith("-")
                    ):
                        rejected.append(next_token)
                        i += 2
                        continue
                    i += 1
                    continue
                safe.append(token)
                i += 1
            if rejected:
                LOG.error(
                    "ignoring blocked tokens in US4_SERVE_MLX_EXTRA_ARGS"
                    " (network binding is fixed to 127.0.0.1 by us4-v6): %s",
                    " ".join(rejected),
                )
            cmd.extend(safe)
    LOG.info("spawning mlx-lm chat backend on 127.0.0.1:%d", SETTINGS.upstream_port)
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, close_fds=True)
    atexit.register(_terminate_proc, proc)
    return proc


def _terminate_proc(proc: Optional[subprocess.Popen]) -> None:
    if not proc or proc.poll() is not None:
        return
    LOG.info("terminating mlx-lm subprocess (pid=%s)", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _wait_upstream_ready(timeout_s: float = 120.0) -> bool:
    if SETTINGS.disable_chat:
        return True
    deadline = time.monotonic() + timeout_s
    health_url = SETTINGS.upstream_health_url
    label = SETTINGS.chat_backend
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    LOG.info("%s chat backend ready at %s", label, SETTINGS.upstream_url)
                    return True
        except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError):
            time.sleep(1.0)
    LOG.warning(
        "%s chat backend did not become ready at %s within %.0fs",
        label, health_url, timeout_s,
    )
    return False


def _init_embeddings() -> None:
    global EMBED_BACKEND
    if SETTINGS.disable_embed:
        LOG.info("embeddings backend disabled; skipping mlx-embeddings init")
        return
    try:
        import mlx_embeddings  # noqa: F401
    except ImportError:
        LOG.error("mlx-embeddings is not installed. Run: pip install -r scripts/requirements-serve.txt")
        sys.exit(3)
    EMBED_BACKEND = EmbeddingsBackend(SETTINGS.embed_model)


def _install_signal_handlers(server: ThreadingHTTPServer, upstream: Optional[subprocess.Popen]) -> None:
    def _shutdown(_signum: int, _frame: Any) -> None:
        LOG.info("signal received; shutting down")
        threading.Thread(target=server.shutdown, daemon=True).start()
        _terminate_proc(upstream)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)


def _warn_if_non_loopback(host: str) -> None:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if host not in ("localhost", ""):
            LOG.warning("binding to non-numeric host %s; ensure access is restricted", host)
        return
    if not ip.is_loopback:
        LOG.warning(
            "binding to non-loopback address %s: this exposes the API to the network "
            "without auth. Use only on trusted networks.",
            host,
        )


def main() -> int:
    logging.basicConfig(level=SETTINGS.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if SETTINGS.chat_backend not in SUPPORTED_CHAT_BACKENDS:
        LOG.warning(
            "unknown chat backend %r; treating as custom upstream. Valid: %s",
            SETTINGS.chat_backend,
            ", ".join(SUPPORTED_CHAT_BACKENDS),
        )
    chat_descr = (
        "off"
        if SETTINGS.disable_chat
        else f"{SETTINGS.chat_backend}:{SETTINGS.chat_model} via {SETTINGS.upstream_url}"
    )
    LOG.info(
        "us4 serve starting (host=%s port=%d chat=%s embed=%s)",
        SETTINGS.host,
        SETTINGS.port,
        chat_descr,
        "off" if SETTINGS.disable_embed else SETTINGS.embed_model,
    )
    _warn_if_non_loopback(SETTINGS.host)
    _init_embeddings()
    upstream = _spawn_upstream()
    _wait_upstream_ready()
    server = ThreadingHTTPServer((SETTINGS.host, SETTINGS.port), Us4Handler)
    _install_signal_handlers(server, upstream)
    LOG.info("listening on http://%s:%d", SETTINGS.host, SETTINGS.port)
    try:
        server.serve_forever()
    finally:
        _terminate_proc(upstream)
    return 0


if __name__ == "__main__":
    sys.exit(main())
