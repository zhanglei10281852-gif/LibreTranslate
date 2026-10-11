"""Lifecycle management for file translation products.

Each translation request owns a job identified by the same UUID that used to
be embedded in the uploaded file name. Job state is persisted on the
filesystem next to the files it describes, so it is shared by every worker
process running on the host and survives worker restarts.

A job moves through the following states::

    uploading -> translating -> ready          (success, result published)
                          \\
                           \\-> discarded       (failure / cancellation)

Layout inside the upload directory::

    <upload_dir>/
        <uuid>.<name>_<target><ext>      # ready (published) results only
        .jobs/
            .gc.lock/                    # single-reaper election lock
            <uuid>/
                state.json               # atomic job state
                input<ext>               # uploaded source
                candidate<ext>           # output being written (never public)
                dl/<token>               # active download reservations
                lock/                    # per-job mutex

Only ``ready`` results are ever reachable through the download endpoint, and
a result becomes ready with a single atomic rename. The periodic sweeper is
elected with an exclusive lock so that at most one worker reclaims expired
results; translating jobs are kept alive by a heartbeat and jobs whose owner
worker died (heartbeat gone stale) are eventually reaped.
"""

import contextlib
import json
import os
import re
import shutil
import threading
import time
import uuid

# Published results are available for 30 minutes (matches the historical
# retention), measured from the moment the result is published.
RESULT_TTL = 30 * 60

# How often the translating worker renews the liveness of its job.
HEARTBEAT_INTERVAL = 20

# A translating job whose heartbeat has been silent for longer than this is
# considered orphaned (its worker died) and reclaimed. Generous on purpose:
# a live translation must never be reaped.
TRANSLATING_STALE = 15 * 60

# An "uploading" job with no progress past this age is a crashed request.
UPLOADING_STALE = 30 * 60

# Download reservations older than this belong to a downloader that died.
DOWNLOAD_STALE = 10 * 60

# A sweeper lock older than this means the elected reaper died mid-sweep.
GC_LOCK_STALE = 10 * 60

STATUS_UPLOADING = "uploading"
STATUS_TRANSLATING = "translating"
STATUS_READY = "ready"

JOBS_DIRNAME = ".jobs"
_GC_LOCK_DIRNAME = ".gc.lock"
_JOB_LOCK_DIRNAME = "lock"
_STATE_FILENAME = "state.json"
_DL_DIRNAME = "dl"

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


class FileNotAvailable(Exception):
    """Raised when a product is missing, not published yet or expired."""


class TranslationJob:
    """Handle to a single in-flight file translation job."""

    def __init__(self, upload_dir, job_id, safe_name, source, target, ext):
        self.upload_dir = upload_dir
        self.id = job_id
        self.safe_name = safe_name
        self.source = source
        self.target = target
        self.ext = ext
        self.status = STATUS_UPLOADING
        self._heartbeat_stop = None

    @property
    def dir(self):
        return os.path.join(_jobs_root(self.upload_dir), self.id)

    @property
    def input_path(self):
        return os.path.join(self.dir, "input" + self.ext)

    @property
    def candidate_path(self):
        return os.path.join(self.dir, "candidate" + self.ext)

    @property
    def public_name(self):
        base = os.path.splitext(self.safe_name)[0]
        # Mirrors the historical argostranslatefiles output naming so that
        # download URLs and the original filename download semantics are
        # unchanged.
        return "%s.%s_%s%s" % (self.id, base, self.target, self.ext)

    @property
    def public_path(self):
        return os.path.join(self.upload_dir, self.public_name)


def _jobs_root(upload_dir):
    return os.path.join(upload_dir, JOBS_DIRNAME)


def init_store(upload_dir):
    """Make sure the on-disk job store exists."""
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(_jobs_root(upload_dir), exist_ok=True)


# ---------------------------------------------------------------------------
# Locking primitives (atomic mkdir works on POSIX and Windows/NTFS).
# ---------------------------------------------------------------------------

def _write_owner(lock_dir, token):
    try:
        with open(os.path.join(lock_dir, "owner"), "w") as f:
            f.write("%d:%s" % (os.getpid(), token))
    except OSError:
        pass


def _read_owner(lock_dir):
    try:
        with open(os.path.join(lock_dir, "owner"), "r") as f:
            return f.read().strip()
    except OSError:
        return None


def _lock_is_stale(lock_dir, stale_timeout, now):
    owner_file = os.path.join(lock_dir, "owner")
    try:
        return (now - os.path.getmtime(owner_file)) > stale_timeout
    except OSError:
        try:
            return (now - os.path.getmtime(lock_dir)) > stale_timeout
        except OSError:
            return False


def _try_mkdir_lock(lock_dir, stale_timeout=None, now=None):
    """Try to acquire a mkdir-based lock.

    Returns an opaque owner token when acquired, ``None`` otherwise. When
    ``stale_timeout`` is given, a lock left behind by a dead owner is taken
    over.
    """
    now = time.time() if now is None else now
    token = uuid.uuid4().hex
    try:
        os.mkdir(lock_dir)
        _write_owner(lock_dir, token)
        return token
    except FileExistsError:
        if stale_timeout is None or not _lock_is_stale(lock_dir, stale_timeout, now):
            return None
        # Take over the stale lock, but only if nobody beat us to it.
        expected = _read_owner(lock_dir)
        if expected is None:
            return None
        shutil.rmtree(lock_dir, ignore_errors=True)
        try:
            os.mkdir(lock_dir)
        except FileExistsError:
            return None
        _write_owner(lock_dir, token)
        return token


def _release_mkdir_lock(lock_dir, token):
    # Never release a lock that has been taken over by another owner.
    if token is not None and _read_owner(lock_dir) != "%d:%s" % (
        os.getpid(),
        token,
    ):
        return
    shutil.rmtree(lock_dir, ignore_errors=True)


@contextlib.contextmanager
def _job_lock(job_dir, *, wait=2.0, stale_timeout=GC_LOCK_STALE, now=None):
    """Acquire the per-job lock, briefly waiting for contention."""
    lock_dir = os.path.join(job_dir, _JOB_LOCK_DIRNAME)
    deadline = time.time() + wait
    token = None
    while True:
        token = _try_mkdir_lock(lock_dir, stale_timeout=stale_timeout, now=now)
        if token is not None or time.time() >= deadline:
            break
        time.sleep(0.05)
    if token is None:
        raise FileNotAvailable("job is busy")
    try:
        yield
    finally:
        _release_mkdir_lock(lock_dir, token)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _state_path(job_dir):
    return os.path.join(job_dir, _STATE_FILENAME)


def _read_state(job_dir):
    try:
        with open(_state_path(job_dir), "r") as f:
            state = json.load(f)
        if isinstance(state, dict) and "id" in state and "status" in state:
            return state
    except (OSError, ValueError):
        pass
    return None


def _write_state(job_dir, state):
    state_path = _state_path(job_dir)
    tmp_path = "%s.tmp.%d.%s" % (state_path, os.getpid(), uuid.uuid4().hex)
    with open(tmp_path, "w") as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, state_path)


def _snapshot(job, status, **extra):
    now = time.time()
    state = {
        "id": job.id,
        "safe_name": job.safe_name,
        "source": job.source,
        "target": job.target,
        "ext": job.ext,
        "status": status,
        "created_at": getattr(job, "created_at", now),
        "updated_at": now,
    }
    state.update(extra)
    if status == STATUS_READY:
        state.setdefault("published_at", now)
        state["public_name"] = job.public_name
    return state


# ---------------------------------------------------------------------------
# Job lifecycle (request side)
# ---------------------------------------------------------------------------

def create_job(upload_dir, safe_name, source, target, ext):
    """Register a new job for a freshly validated upload."""
    init_store(upload_dir)
    now = time.time()
    for _ in range(3):
        job_id = str(uuid.uuid4())
        job_dir = os.path.join(_jobs_root(upload_dir), job_id)
        try:
            os.mkdir(job_dir)
        except FileExistsError:
            continue
        job = TranslationJob(upload_dir, job_id, safe_name, source, target, ext)
        job.created_at = now
        state = _snapshot(job, STATUS_UPLOADING)
        _write_state(job_dir, state)
        return job
    raise RuntimeError("Could not allocate a file translation job")


def begin_translation(job):
    """Move the job to the translating state and start a heartbeat."""
    job.status = STATUS_TRANSLATING
    _write_state(job.dir, _snapshot(job, STATUS_TRANSLATING))

    stop = threading.Event()

    def _heartbeat():
        while not stop.wait(HEARTBEAT_INTERVAL):
            try:
                os.utime(_state_path(job.dir), None)
            except OSError:
                return

    thread = threading.Thread(
        target=_heartbeat, name="file-job-heartbeat-%s" % job.id, daemon=True
    )
    thread.start()
    job._heartbeat_stop = (stop, thread)


def _stop_heartbeat(job):
    handle = job.__dict__.pop("_heartbeat_stop", None)
    if handle is not None:
        stop, thread = handle
        stop.set()
        thread.join(timeout=1)


def candidate_output_path(job):
    """Callable meant to be passed as ``get_output_path`` to
    ``argostranslatefiles.translate_file`` so that candidate output is never
    written under its public name."""

    def _resolve(_underlying_translation, _input_path):
        return job.candidate_path

    return _resolve


def publish(job, output_path):
    """Atomically publish a complete candidate as the downloadable result.

    Returns the public file name. Raises if the translation did not produce a
    complete candidate.
    """
    _stop_heartbeat(job)

    if not output_path or os.path.abspath(output_path) != os.path.abspath(
        job.candidate_path
    ):
        raise FileNotFoundError("The translated file was not produced")
    if not os.path.isfile(job.candidate_path):
        raise FileNotFoundError("The translated file is missing")

    with _job_lock(job.dir):
        if not os.path.isfile(job.candidate_path):
            raise FileNotFoundError("The translated file is missing")

        # Atomic publish: the public name only ever points at a complete file.
        os.replace(job.candidate_path, job.public_path)
        try:
            os.remove(job.input_path)
        except OSError:
            pass

        job.status = STATUS_READY
        state = _snapshot(job, STATUS_READY)
        _write_state(job.dir, state)

    return job.public_name


def discard(job):
    """Remove every unpublished artifact of a failed/cancelled job."""
    _stop_heartbeat(job)
    try:
        with _job_lock(job.dir, wait=0.5):
            for path in (job.candidate_path, job.input_path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            shutil.rmtree(job.dir, ignore_errors=True)
    except (FileNotAvailable, OSError):
        shutil.rmtree(job.dir, ignore_errors=True)

    # A publish never reported success, so nothing public should exist; make
    # sure a crash between rename and state commit cannot leak the result.
    try:
        os.remove(job.public_path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

def _job_id_from_filename(filename):
    job_id = filename.split(".", 1)[0]
    return job_id if _UUID_RE.match(job_id) else None


@contextlib.contextmanager
def reserve_download(upload_dir, filename, now=None):
    """Reserve a published product for reading.

    Yields the absolute path of the file to stream. Raises
    :class:`FileNotAvailable` for products that are missing, not yet
    published or expired. A reservation marker keeps the sweeper away while
    the file is being read.
    """
    now = time.time() if now is None else now
    job_id = _job_id_from_filename(filename)
    if job_id is None:
        raise FileNotAvailable("Invalid file identifier")

    job_dir = os.path.join(_jobs_root(upload_dir), job_id)
    if not os.path.isdir(job_dir):
        raise FileNotAvailable("The translated file is not available")

    marker_path = None

    try:
        with _job_lock(job_dir, wait=2.0):
            state = _read_state(job_dir)
            if (
                state is None
                or state.get("status") != STATUS_READY
                or state.get("public_name") != filename
            ):
                raise FileNotAvailable("The translated file is not available")

            published_at = state.get("published_at")
            if published_at is None or (now - float(published_at)) > RESULT_TTL:
                raise FileNotAvailable("The translated file has expired")

            public_path = os.path.join(upload_dir, filename)
            if not os.path.isfile(public_path):
                raise FileNotAvailable("The translated file is not available")

            dl_dir = os.path.join(job_dir, _DL_DIRNAME)
            os.makedirs(dl_dir, exist_ok=True)
            marker_path = os.path.join(dl_dir, uuid.uuid4().hex)
            with open(marker_path, "w") as f:
                f.write("%d" % os.getpid())
    except FileNotFoundError:
        # The job directory vanished between the lookup and the lock.
        raise FileNotAvailable("The translated file is not available")

    try:
        # Re-check after releasing the lock: a sweeper cannot have removed
        # the file while a fresh marker exists, but a plain open still guards
        # against filesystem surprises.
        if not os.path.isfile(public_path):
            raise FileNotAvailable("The translated file is not available")
        yield public_path
    finally:
        if marker_path is not None:
            try:
                os.remove(marker_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Reclamation
# ---------------------------------------------------------------------------

def _fresh_download_markers(job_dir, now):
    """Return the number of active (not stale) download reservations."""
    dl_dir = os.path.join(job_dir, _DL_DIRNAME)
    active = 0
    try:
        entries = os.listdir(dl_dir)
    except FileNotFoundError:
        return 0
    for name in entries:
        marker = os.path.join(dl_dir, name)
        try:
            if (now - os.path.getmtime(marker)) <= DOWNLOAD_STALE and os.path.isfile(
                marker
            ):
                active += 1
            else:
                # Stale marker left behind by a dead downloader.
                os.remove(marker)
        except OSError:
            pass
    return active


def _public_name_from_state(state):
    try:
        base = os.path.splitext(str(state["safe_name"]))[0]
        return "%s.%s_%s%s" % (
            state["id"],
            base,
            state["target"],
            state.get("ext", ""),
        )
    except (KeyError, TypeError):
        return state.get("public_name")


def _remove_job(upload_dir, job_id, state=None):
    public_name = None
    if state is not None:
        public_name = state.get("public_name") or _public_name_from_state(state)
    if public_name:
        try:
            os.remove(os.path.join(upload_dir, public_name))
        except OSError:
            pass
    else:
        # Defensive fallback: remove any top-level file owned by this job.
        try:
            for name in os.listdir(upload_dir):
                if os.path.isfile(os.path.join(upload_dir, name)) and \
                        name.split(".", 1)[0] == job_id:
                    os.remove(os.path.join(upload_dir, name))
        except OSError:
            pass
    shutil.rmtree(os.path.join(_jobs_root(upload_dir), job_id), ignore_errors=True)


def _newest_mtime(path):
    """Newest mtime of job content, ignoring the lock directory.

    The lock is touched by the sweeper itself and must never make a
    half-initialized job look fresh.
    """
    newest = 0
    try:
        for root, dirs, files in os.walk(path):
            if _JOB_LOCK_DIRNAME in dirs:
                dirs.remove(_JOB_LOCK_DIRNAME)
            for name in files:
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(root, name)))
                except OSError:
                    pass
    except OSError:
        pass
    return newest


def _sweep_jobs(upload_dir, now):
    root = _jobs_root(upload_dir)
    try:
        entries = os.listdir(root)
    except FileNotFoundError:
        return

    for job_id in entries:
        if job_id == _GC_LOCK_DIRNAME:
            continue
        job_dir = os.path.join(root, job_id)
        if not os.path.isdir(job_dir):
            continue

        try:
            lock_token = _try_mkdir_lock(
                os.path.join(job_dir, _JOB_LOCK_DIRNAME),
                stale_timeout=GC_LOCK_STALE,
                now=now,
            )
        except OSError:
            # The job directory may have been removed concurrently.
            continue
        if lock_token is None:
            # Owned by a request or another reaper; try again next cycle.
            continue

        try:
            state = _read_state(job_dir)

            if state is None:
                # Corrupted or half-initialized state: only reclaim once no
                # file in the directory has been touched recently, so a job
                # being created is never touched.
                if now - _newest_mtime(job_dir) > UPLOADING_STALE:
                    _remove_job(upload_dir, job_id)
                continue

            status = state.get("status")

            if status == STATUS_READY:
                published_at = state.get("published_at")
                expired = (
                    published_at is not None
                    and (now - float(published_at)) > RESULT_TTL
                )
                public_path = os.path.join(
                    upload_dir, state.get("public_name", "")
                )
                public_missing = state.get("public_name") and not os.path.isfile(
                    public_path
                )
                if public_missing or (
                    expired and _fresh_download_markers(job_dir, now) == 0
                ):
                    _remove_job(upload_dir, job_id, state)

            elif status == STATUS_TRANSLATING:
                # Kept alive by the heartbeat of the translating worker.
                try:
                    last_seen = os.path.getmtime(_state_path(job_dir))
                except OSError:
                    last_seen = 0
                if now - last_seen > TRANSLATING_STALE:
                    _remove_job(upload_dir, job_id, state)

            elif status == STATUS_UPLOADING:
                if now - float(state.get("created_at", 0)) > UPLOADING_STALE:
                    _remove_job(upload_dir, job_id, state)

            else:
                _remove_job(upload_dir, job_id, state)
        except (OSError, ValueError):
            # Never let a single bad job stop the sweep.
            continue
        finally:
            _release_mkdir_lock(
                os.path.join(job_dir, _JOB_LOCK_DIRNAME), lock_token
            )


def _sweep_legacy_files(upload_dir, now):
    """Reclaim unmanaged top-level files (e.g. from older releases).

    Files belonging to a live job directory are never touched; anything else
    older than the result TTL is removed, preserving the historical behavior.
    """
    root = _jobs_root(upload_dir)
    try:
        names = os.listdir(upload_dir)
    except FileNotFoundError:
        return

    for name in names:
        path = os.path.join(upload_dir, name)
        if not os.path.isfile(path):
            continue
        job_id = _job_id_from_filename(name)
        if job_id is not None and os.path.isdir(os.path.join(root, job_id)):
            continue
        try:
            if (now - os.path.getmtime(path)) > RESULT_TTL:
                os.remove(path)
        except OSError:
            pass


def sweep(upload_dir, now=None):
    """Run one reclamation pass.

    Only one worker at a time performs the sweep (elected through an exclusive
    lock). Returns True when this caller was the elected reaper.
    """
    now = time.time() if now is None else now
    init_store(upload_dir)

    gc_lock = os.path.join(_jobs_root(upload_dir), _GC_LOCK_DIRNAME)
    lock_token = _try_mkdir_lock(
        gc_lock, stale_timeout=GC_LOCK_STALE, now=now
    )
    if lock_token is None:
        return False

    try:
        _sweep_jobs(upload_dir, now)
        _sweep_legacy_files(upload_dir, now)
    finally:
        _release_mkdir_lock(gc_lock, lock_token)

    return True
