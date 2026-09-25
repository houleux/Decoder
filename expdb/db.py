"""
Connection handling for the central experiment database.

The database is PostgreSQL hosted on Neon, shared by every machine that runs
experiments. The connection string comes from the EXPDB_URL environment
variable, or from EXPDB_URL in the repo-root .env file (a real environment
variable wins over .env). There is deliberately no fallback to a local
database: if EXPDB_URL is missing, every call fails loudly rather than writing
results somewhere nobody else can see.

Why this module speaks HTTPS instead of using a Postgres driver: the Ada
cluster's firewall only allows outbound traffic on web ports (22/80/443/8080/
8443). The Postgres wire protocol (5432) is blocked, so psycopg cannot reach
Neon from a compute node. Neon also serves SQL over HTTPS on 443 -- the
transport its official serverless driver (@neondatabase/serverless) uses in
HTTP mode -- and this module is a minimal client for that transport:

    POST https://<endpoint-host>/sql
    Neon-Connection-String: <EXPDB_URL>
    {"query": "...$1...", "params": [...]}            one statement
    {"queries": [{"query": ..., "params": ...}, ...]} one transaction

Consequences for callers:
  * Placeholders are Postgres-native `$1, $2, ...`, not `?` or `%s`.
  * There are no interactive transactions. `execute_batch()` runs a list of
    statements atomically; a read-then-write sequence cannot be atomic.
  * Values come back as text and are parsed here by column type OID. An
    unknown type raises instead of guessing.

The schema is NOT created on connect. Create it once per database with
`python3 cli/exp.py init-db`; until then every query fails with
"relation ... does not exist".
"""
import datetime as dt
import decimal
import http.client
import json
import numbers
import os
import re
import select
import threading
import time
import urllib.parse
import uuid
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
ENV_VAR = "EXPDB_URL"

REQUEST_TIMEOUT_S = 60.0   # generous: a suspended Neon compute takes seconds to wake
CONNECT_ATTEMPTS = 5       # opening a TCP/TLS connection: nothing sent yet, always safe to retry
READ_ATTEMPTS = 3          # SELECT/SHOW: safe to resend
RETRY_BACKOFF_S = 2.0
IDLE_REOPEN_S = 30.0       # don't reuse a keep-alive connection idle longer than this

# Writes are never resent. If a write's request fails after it was sent, the
# statement may or may not have run, and resending an increment (commit_chunk)
# could double-count frames. The caller crashes instead, and run_experiments.py
# resumes from whatever the database actually holds.


class DatabaseError(Exception):
    """Postgres rejected the statement (syntax error, constraint violation, ...)."""

    def __init__(self, message: str, code: str | None):
        super().__init__(f"{message} (SQLSTATE {code})")
        self.code = code


class TransportError(Exception):
    """The request could not be completed. For a write, the outcome is unknown."""


def get_url() -> str:
    load_dotenv(ENV_FILE, override=False)
    url = os.environ.get(ENV_VAR)
    if not url:
        raise RuntimeError(
            f"{ENV_VAR} is not set. Put the Neon *pooled* connection string in "
            f"{ENV_FILE} as {ENV_VAR}=postgresql://... (or export it). "
            f"There is no local-database fallback."
        )
    return url


# ---------------------------------------------------------------------------
# Value encoding (Python -> text params) and decoding (text -> Python by OID)
# ---------------------------------------------------------------------------

def _encode(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, numbers.Integral):          # includes numpy integers
        return str(int(v))
    if isinstance(v, numbers.Real):              # includes numpy floats
        return repr(float(v))                    # shortest exact round-trip repr
    if isinstance(v, (str, decimal.Decimal)):
        return str(v)
    if isinstance(v, dt.datetime):
        if v.tzinfo is None:
            raise ValueError(f"refusing to send naive datetime {v!r}; attach a timezone")
        return v.isoformat()
    raise TypeError(f"expdb: cannot send parameter of type {type(v).__name__}")


_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})(?:\.(\d{1,6}))?(?:([+-])(\d{2})(?::?(\d{2}))?)?$")


def _parse_timestamp(s: str, aware: bool) -> dt.datetime:
    m = _TS_RE.match(s)
    if not m or (m.group(4) is not None) != aware:
        raise ValueError(f"expdb: unparseable {'timestamptz' if aware else 'timestamp'} {s!r}")
    date, clock, frac, sign, hh, mm = m.groups()
    iso = f"{date}T{clock}.{(frac or '0').ljust(6, '0')}"
    if aware:
        iso += f"{sign}{hh}:{mm or '00'}"
    return dt.datetime.fromisoformat(iso)


_PARSERS = {
    16: lambda s: {"t": True, "f": False}[s],                 # bool
    20: int, 21: int, 23: int, 26: int,                       # int8, int2, int4, oid
    700: float, 701: float,                                   # float4, float8
    1700: decimal.Decimal,                                    # numeric (e.g. SUM(bigint))
    19: str, 25: str, 1042: str, 1043: str,                   # name, text, bpchar, varchar
    1114: lambda s: _parse_timestamp(s, aware=False),         # timestamp
    1184: lambda s: _parse_timestamp(s, aware=True),          # timestamptz
}


class Result:
    """Rows of one statement, mimicking the DB-API cursor subset expdb uses."""

    def __init__(self, raw: dict):
        fields = raw.get("fields") or []
        self.columns = [f["name"] for f in fields]
        self.rowcount = raw.get("rowCount")
        parsers = []
        for f in fields:
            oid = f["dataTypeID"]
            if oid not in _PARSERS:
                raise TypeError(f"expdb: no parser for Postgres type OID {oid} "
                                f"(column {f['name']!r}); add one to expdb/db.py::_PARSERS")
            parsers.append(_PARSERS[oid])
        self._rows = [
            tuple(None if v is None else p(v) for p, v in zip(parsers, row))
            for row in raw.get("rows") or []
        ]
        self._pos = 0

    def fetchone(self):
        if self._pos >= len(self._rows):
            return None
        self._pos += 1
        return self._rows[self._pos - 1]

    def fetchall(self):
        rows = self._rows[self._pos:]
        self._pos = len(self._rows)
        return rows


def _is_read_only(sql: str) -> bool:
    return sql.lstrip().split(None, 1)[0].upper() in ("SELECT", "SHOW")


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

class Connection:
    """
    One keep-alive HTTPS connection to Neon's SQL endpoint. Not thread-safe:
    use one per thread (get_conn() does this).
    """

    def __init__(self, url: str):
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("postgres", "postgresql") or not parts.hostname:
            raise RuntimeError(f"{ENV_VAR} must be a postgresql:// connection string")
        self._url = url
        self._host = parts.hostname
        self._http: http.client.HTTPSConnection | None = None
        self._last_used = 0.0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        self._drop_http()
        self.closed = True

    def execute(self, sql: str, params=()) -> Result:
        body = {"query": sql, "params": [_encode(p) for p in params]}
        return Result(self._post(body, read_only=_is_read_only(sql)))

    def execute_batch(self, statements: list[tuple[str, tuple]]) -> list[Result]:
        """Runs [(sql, params), ...] in ONE transaction: all commit or none do."""
        body = {"queries": [{"query": sql, "params": [_encode(p) for p in params]}
                            for sql, params in statements]}
        raw = self._post(body, read_only=all(_is_read_only(sql) for sql, _ in statements))
        return [Result(r) for r in raw["results"]]

    # -- transport ----------------------------------------------------------

    def _drop_http(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    def _connection_dropped(self) -> bool:
        # A keep-alive socket the server has closed polls readable (EOF).
        sock = self._http.sock
        return sock is None or bool(select.select([sock], [], [], 0)[0])

    def _ensure_http(self) -> http.client.HTTPSConnection:
        if self.closed:
            raise RuntimeError("expdb connection is closed")
        if self._http is not None and (
                time.monotonic() - self._last_used > IDLE_REOPEN_S or self._connection_dropped()):
            self._drop_http()
        if self._http is None:
            for attempt in range(1, CONNECT_ATTEMPTS + 1):
                conn = http.client.HTTPSConnection(self._host, 443, timeout=REQUEST_TIMEOUT_S)
                try:
                    conn.connect()
                    break
                except OSError as e:
                    conn.close()
                    if attempt == CONNECT_ATTEMPTS:
                        raise TransportError(
                            f"could not connect to {self._host}:443 after "
                            f"{CONNECT_ATTEMPTS} attempts: {e}") from e
                    time.sleep(RETRY_BACKOFF_S * 2 ** (attempt - 1))
            self._http = conn
        return self._http

    def _post(self, body: dict, read_only: bool) -> dict:
        payload = json.dumps(body)
        headers = {
            "Content-Type": "application/json",
            "Neon-Connection-String": self._url,
            "Neon-Raw-Text-Output": "true",
            "Neon-Array-Mode": "true",
        }
        attempts = READ_ATTEMPTS if read_only else 1
        for attempt in range(1, attempts + 1):
            conn = self._ensure_http()
            try:
                conn.request("POST", "/sql", payload, headers)
                resp = conn.getresponse()
                data = resp.read()
            except (OSError, http.client.HTTPException) as e:
                self._drop_http()
                if attempt < attempts:
                    time.sleep(RETRY_BACKOFF_S * 2 ** (attempt - 1))
                    continue
                raise TransportError(
                    f"request to Neon failed ({type(e).__name__}: {e}). "
                    + ("" if read_only else "This was a write: it may or may not have been applied. ")
                ) from e
            self._last_used = time.monotonic()
            if resp.status == 200:
                return json.loads(data)
            try:
                err = json.loads(data)
            except ValueError:
                err = None
            if isinstance(err, dict) and err.get("code"):
                raise DatabaseError(err.get("message", ""), err["code"])
            raise TransportError(f"Neon returned HTTP {resp.status}: {data[:300]!r}")


_local = threading.local()


def connect() -> Connection:
    """
    Opens a new, uncached connection. Use as a context manager
    (`with connect() as conn:`) where it should not outlive the block, e.g.
    once per web request.
    """
    return Connection(get_url())


def get_conn() -> Connection:
    """
    Returns this thread's cached connection, opening one if needed. Cached so a
    sweep reuses one keep-alive HTTPS connection instead of paying a TLS
    handshake per expdb call.
    """
    conn = getattr(_local, "conn", None)
    if conn is None or conn.closed or _local.pid != os.getpid():
        conn = connect()
        _local.conn = conn
        _local.pid = os.getpid()
    return conn


def close_conn() -> None:
    """Closes this thread's cached connection, if any."""
    conn = getattr(_local, "conn", None)
    if conn is not None and _local.pid == os.getpid():
        conn.close()
    _local.conn = None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_schema(conn: Connection) -> None:
    """
    Creates the tables if they don't exist. Run explicitly (cli/exp.py init-db),
    not on every connect.

    Counters are BIGINT: `messages` already reaches ~6.5e8 on WRAN at 100k
    frames, a third of the INTEGER limit.
    """
    conn.execute_batch([
        ("""
        CREATE TABLE IF NOT EXISTS configs (
            config_id    TEXT PRIMARY KEY,
            config_json  TEXT NOT NULL,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
            description  TEXT,
            tags         TEXT
        )
        """, ()),
        ("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id             TEXT PRIMARY KEY,
            config_id          TEXT NOT NULL REFERENCES configs(config_id),
            run_type           TEXT NOT NULL,   -- 'train' | 'eval_only' | 'train+eval'
            status             TEXT NOT NULL,   -- 'running' | 'completed' | 'failed' | 'interrupted'
            full_config_json   TEXT NOT NULL,
            checkpoint_path    TEXT,            -- NULL for eval_only
            episodes_done      INTEGER,         -- NULL for eval_only
            intermediate_checkpoints TEXT,      -- JSON list of paths (if opt-in)
            training_stats_csv TEXT,            -- path to CSV (if opt-in)
            started_at         TIMESTAMPTZ DEFAULT current_timestamp,
            completed_at       TIMESTAMPTZ,
            error_message      TEXT
        )
        """, ()),
        ("""
        CREATE TABLE IF NOT EXISTS eval_results (
            config_id           TEXT NOT NULL REFERENCES configs(config_id),
            snr_db              DOUBLE PRECISION NOT NULL,
            target_frame_errors INTEGER NOT NULL,
            max_frames          INTEGER NOT NULL,

            frames_done         BIGINT NOT NULL DEFAULT 0,
            completed           BOOLEAN NOT NULL DEFAULT FALSE,

            bit_errors          BIGINT NOT NULL DEFAULT 0,
            total_bits          BIGINT NOT NULL DEFAULT 0,
            frame_errors        BIGINT NOT NULL DEFAULT 0,
            messages            BIGINT NOT NULL DEFAULT 0,

            last_updated        TIMESTAMPTZ DEFAULT current_timestamp,
            PRIMARY KEY (config_id, snr_db, target_frame_errors, max_frames)
        )
        """, ()),
    ])


# ---------------------------------------------------------------------------
# Throwaway databases for tests
# ---------------------------------------------------------------------------

def _with_dbname(url: str, dbname: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit(parts._replace(path=f"/{dbname}"))


@contextmanager
def temporary_database(prefix: str = "expdb_test_"):
    """
    Tests only. Creates a throwaway database on the same Neon branch as
    EXPDB_URL, initialises the schema, and points EXPDB_URL (in os.environ, so
    subprocesses inherit it) at it for the duration of the block. Drops it
    afterwards. Only ever drops the database it created.
    """
    prod_url = get_url()
    name = f"{prefix}{uuid.uuid4().hex[:12]}"
    if urllib.parse.urlsplit(prod_url).path.lstrip("/") == name:
        raise RuntimeError("refusing to use the production database as a test database")

    with connect() as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    saved = os.environ.get(ENV_VAR)
    close_conn()
    os.environ[ENV_VAR] = _with_dbname(prod_url, name)
    try:
        init_schema(get_conn())
        yield name
    finally:
        close_conn()
        if saved is None:
            del os.environ[ENV_VAR]
        else:
            os.environ[ENV_VAR] = saved
        with Connection(prod_url) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
