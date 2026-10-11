import json
import os
import shutil
import tempfile
import time

import pytest

from libretranslate import file_jobs


@pytest.fixture()
def upload_dir():
    path = tempfile.mkdtemp(prefix="lt-jobs-test-")
    try:
        file_jobs.init_store(path)
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _state(upload_dir, job):
    with open(os.path.join(job.dir, "state.json"), "r") as f:
        return json.load(f)


def _make_published_job(upload_dir, name="report.txt", content=b"translated"):
    """Drive a job through its full successful lifecycle."""
    safe_name = name
    ext = os.path.splitext(safe_name)[1]
    job = file_jobs.create_job(upload_dir, safe_name, "en", "es", ext)
    with open(job.input_path, "wb") as f:
        f.write(b"source")
    file_jobs.begin_translation(job)
    with open(job.candidate_path, "wb") as f:
        f.write(content)
    public_name = file_jobs.publish(job, job.candidate_path)
    return job, public_name


def test_create_job_layout_and_identity(upload_dir):
    job = file_jobs.create_job(upload_dir, "My_report.txt", "en", "es", ".txt")

    assert os.path.isdir(job.dir)
    assert job.input_path.endswith(os.path.join(".jobs", job.id, "input.txt"))
    assert job.candidate_path.endswith(
        os.path.join(".jobs", job.id, "candidate.txt")
    )
    state = _state(upload_dir, job)
    assert state["status"] == file_jobs.STATUS_UPLOADING

    # Public naming preserves the historical uuid + original name semantics.
    assert job.public_name == "%s.My_report_es.txt" % job.id


def test_publish_is_atomic_and_hides_candidate(upload_dir):
    job, public_name = _make_published_job(upload_dir)

    # The candidate is now exclusively available under its public name.
    assert not os.path.exists(job.candidate_path)
    assert not os.path.exists(job.input_path)
    assert os.path.isfile(os.path.join(upload_dir, public_name))
    assert _state(upload_dir, job)["status"] == file_jobs.STATUS_READY
    assert _state(upload_dir, job)["public_name"] == public_name


def test_publish_requires_complete_candidate(upload_dir):
    job = file_jobs.create_job(upload_dir, "report.txt", "en", "es", ".txt")
    file_jobs.begin_translation(job)

    # translate_file reporting no output / a foreign path must not publish.
    with pytest.raises(FileNotFoundError):
        file_jobs.publish(job, False)
    with pytest.raises(FileNotFoundError):
        file_jobs.publish(job, os.path.join(upload_dir, "elsewhere.txt"))

    # Nothing leaked under a public name.
    assert [n for n in os.listdir(upload_dir) if os.path.isfile(
        os.path.join(upload_dir, n))] == []

    file_jobs.discard(job)


def test_discard_cleans_unpublished_artifacts(upload_dir):
    job = file_jobs.create_job(upload_dir, "report.txt", "en", "es", ".txt")
    with open(job.input_path, "w") as f:
        f.write("source")
    file_jobs.begin_translation(job)
    with open(job.candidate_path, "w") as f:
        f.write("half")

    file_jobs.discard(job)

    assert not os.path.exists(job.dir)
    assert not os.path.exists(job.public_path)


def test_discard_removes_result_if_publish_did_not_commit(upload_dir):
    # Simulate a crash after the atomic rename but before the state commit.
    job = file_jobs.create_job(upload_dir, "report.txt", "en", "es", ".txt")
    with open(job.input_path, "w") as f:
        f.write("source")
    with open(job.candidate_path, "w") as f:
        f.write("done")
    os.replace(job.candidate_path, job.public_path)

    file_jobs.discard(job)

    assert not os.path.exists(job.public_path)
    assert not os.path.exists(job.dir)


def test_reserve_download_happy_path_and_filename_semantics(upload_dir):
    job, public_name = _make_published_job(upload_dir, name="report.txt")

    with file_jobs.reserve_download(upload_dir, public_name) as path:
        assert path == os.path.join(upload_dir, public_name)
        with open(path, "rb") as f:
            assert f.read() == b"translated"
        # A reservation marker exists while reading.
        markers = os.listdir(os.path.join(job.dir, "dl"))
        assert len(markers) == 1

    # Marker is released once the download is done.
    assert not os.path.isdir(os.path.join(job.dir, "dl")) or \
        os.listdir(os.path.join(job.dir, "dl")) == []


def test_download_unpublished_translating_job_is_rejected(upload_dir):
    job = file_jobs.create_job(upload_dir, "report.txt", "en", "es", ".txt")
    with open(job.input_path, "w") as f:
        f.write("source")
    file_jobs.begin_translation(job)
    with open(job.candidate_path, "w") as f:
        f.write("half")

    with pytest.raises(file_jobs.FileNotAvailable):
        with file_jobs.reserve_download(upload_dir, job.public_name):
            pass

    # A candidate written under the public name would still be unreachable.
    assert not os.path.exists(job.public_path)


def test_download_unknown_invalid_and_expired(upload_dir):
    job, public_name = _make_published_job(upload_dir)

    with pytest.raises(file_jobs.FileNotAvailable):
        with file_jobs.reserve_download(upload_dir, "not-a-uuid.report_es.txt"):
            pass

    missing_id = "00000000-0000-0000-0000-000000000000.report_es.txt"
    with pytest.raises(file_jobs.FileNotAvailable):
        with file_jobs.reserve_download(upload_dir, missing_id):
            pass

    future = time.time() + file_jobs.RESULT_TTL + 1
    with pytest.raises(file_jobs.FileNotAvailable):
        with file_jobs.reserve_download(upload_dir, public_name, now=future):
            pass


def test_sweep_keeps_fresh_ready_and_translating_jobs(upload_dir):
    ready_job, ready_name = _make_published_job(upload_dir, name="a.txt")
    translating_job = file_jobs.create_job(
        upload_dir, "b.txt", "en", "es", ".txt"
    )
    with open(translating_job.input_path, "w") as f:
        f.write("source")
    file_jobs.begin_translation(translating_job)

    assert file_jobs.sweep(upload_dir) is True

    assert os.path.isfile(os.path.join(upload_dir, ready_name))
    assert os.path.isdir(ready_job.dir)
    assert os.path.isfile(translating_job.input_path)
    assert os.path.isdir(translating_job.dir)


def test_sweep_reclaims_expired_results(upload_dir):
    job, public_name = _make_published_job(upload_dir)
    past = time.time() - file_jobs.RESULT_TTL - 1

    state = _state(upload_dir, job)
    state["published_at"] = past
    with open(os.path.join(job.dir, "state.json"), "w") as f:
        json.dump(state, f)

    assert file_jobs.sweep(upload_dir, now=time.time())

    assert not os.path.exists(job.public_path)
    assert not os.path.exists(job.dir)
    assert public_name not in os.listdir(upload_dir)


def test_sweep_never_deletes_a_result_being_downloaded(upload_dir):
    job, public_name = _make_published_job(upload_dir)
    past = time.time() - file_jobs.RESULT_TTL - 1

    state = _state(upload_dir, job)
    state["published_at"] = past
    with open(os.path.join(job.dir, "state.json"), "w") as f:
        json.dump(state, f)

    # Expire the state timestamps but hold a fresh download reservation.
    os.utime(os.path.join(job.dir, "state.json"), (past, past))
    dl_dir = os.path.join(job.dir, "dl")
    os.makedirs(dl_dir)
    marker = os.path.join(dl_dir, "active")
    with open(marker, "w") as f:
        f.write("1")

    assert file_jobs.sweep(upload_dir, now=time.time())

    assert os.path.isfile(job.public_path)
    assert os.path.isdir(job.dir)

    # Once the downloader is gone the result is reclaimed.
    os.remove(marker)
    assert file_jobs.sweep(upload_dir, now=time.time())
    assert not os.path.exists(job.public_path)
    assert not os.path.exists(job.dir)


def test_sweep_reclaims_after_stale_download_marker(upload_dir):
    job, public_name = _make_published_job(upload_dir)
    now = time.time()
    past = now - file_jobs.RESULT_TTL - 1

    state = _state(upload_dir, job)
    state["published_at"] = past
    with open(os.path.join(job.dir, "state.json"), "w") as f:
        json.dump(state, f)

    dl_dir = os.path.join(job.dir, "dl")
    os.makedirs(dl_dir)
    stale_marker = os.path.join(dl_dir, "dead")
    with open(stale_marker, "w") as f:
        f.write("1")
    old = now - file_jobs.DOWNLOAD_STALE - 1
    os.utime(stale_marker, (old, old))

    assert file_jobs.sweep(upload_dir, now=now)

    assert not os.path.exists(job.public_path)
    assert not os.path.exists(job.dir)


def test_sweep_reaps_orphaned_translating_job_after_restart(upload_dir):
    # A worker died mid-translation: state says translating, no heartbeat.
    job = file_jobs.create_job(upload_dir, "big.docx", "en", "es", ".docx")
    with open(job.input_path, "wb") as f:
        f.write(b"source")
    file_jobs.begin_translation(job)
    with open(job.candidate_path, "wb") as f:
        f.write(b"half output")
    file_jobs._stop_heartbeat(job)  # simulate the worker going away

    old = time.time() - file_jobs.TRANSLATING_STALE - 1
    os.utime(os.path.join(job.dir, "state.json"), (old, old))

    # A fresh sweep (as another worker would run) reaps it.
    assert file_jobs.sweep(upload_dir, now=time.time())

    assert not os.path.exists(job.dir)
    assert not os.path.exists(job.input_path)
    assert not os.path.exists(job.candidate_path)


def test_heartbeat_keeps_long_translation_alive(upload_dir):
    job = file_jobs.create_job(upload_dir, "huge.txt", "en", "es", ".txt")
    file_jobs.begin_translation(job)
    with open(job.input_path, "w") as f:
        f.write("source")

    # Pretend the upload happened ages ago (well beyond any retention).
    ancient = time.time() - file_jobs.RESULT_TTL * 10
    os.utime(job.input_path, (ancient, ancient))

    assert file_jobs.sweep(upload_dir, now=time.time())
    assert os.path.isfile(job.input_path)

    file_jobs.discard(job)


def test_sweep_reaps_stale_uploading_and_unknown_states(upload_dir):
    job = file_jobs.create_job(upload_dir, "lost.txt", "en", "es", ".txt")
    old = time.time() - file_jobs.UPLOADING_STALE - 1
    state = _state(upload_dir, job)
    state["created_at"] = old
    with open(os.path.join(job.dir, "state.json"), "w") as f:
        json.dump(state, f)

    assert file_jobs.sweep(upload_dir, now=time.time())
    assert not os.path.exists(job.dir)


def test_sweep_reaps_corrupted_state_only_after_grace(upload_dir):
    job = file_jobs.create_job(upload_dir, "weird.txt", "en", "es", ".txt")
    with open(os.path.join(job.dir, "state.json"), "w") as f:
        f.write("{broken")

    # Recently touched: leave alone (job may be initializing).
    assert file_jobs.sweep(upload_dir, now=time.time())
    assert os.path.isdir(job.dir)

    old = time.time() - file_jobs.UPLOADING_STALE - 1
    for root, _dirs, files in os.walk(job.dir):
        for name in files:
            os.utime(os.path.join(root, name), (old, old))

    assert file_jobs.sweep(upload_dir, now=time.time())
    assert not os.path.exists(job.dir)


def test_sweep_reclaims_legacy_files(upload_dir):
    # Files produced by older releases have no job state.
    old_file = os.path.join(
        upload_dir, "11111111-1111-1111-1111-111111111111.old.txt"
    )
    with open(old_file, "w") as f:
        f.write("old")
    old = time.time() - file_jobs.RESULT_TTL - 1
    os.utime(old_file, (old, old))

    fresh_file = os.path.join(
        upload_dir, "22222222-2222-2222-2222-222222222222.new.txt"
    )
    with open(fresh_file, "w") as f:
        f.write("new")

    assert file_jobs.sweep(upload_dir, now=time.time())

    assert not os.path.exists(old_file)
    assert os.path.isfile(fresh_file)


def test_single_reaper_per_cycle(upload_dir):
    gc_lock = os.path.join(upload_dir, ".jobs", ".gc.lock")
    token = file_jobs._try_mkdir_lock(gc_lock)
    assert token is not None

    # Another worker's concurrent scan must not perform any deletion.
    job, public_name = _make_published_job(upload_dir)
    assert file_jobs.sweep(upload_dir) is False
    assert os.path.isfile(os.path.join(upload_dir, public_name))

    file_jobs._release_mkdir_lock(gc_lock, token)
    assert file_jobs.sweep(upload_dir) is True


def test_stale_reaper_lock_is_taken_over(upload_dir):
    gc_lock = os.path.join(upload_dir, ".jobs", ".gc.lock")
    dead_token = file_jobs._try_mkdir_lock(gc_lock)
    old = time.time() - file_jobs.GC_LOCK_STALE - 1
    os.utime(os.path.join(gc_lock, "owner"), (old, old))

    # A live reaper takes over the stale lock.
    now = time.time()
    new_token = file_jobs._try_mkdir_lock(
        gc_lock, stale_timeout=file_jobs.GC_LOCK_STALE, now=now
    )
    assert new_token is not None
    assert new_token != dead_token

    # The dead owner releasing later must not free the new owner's lock.
    file_jobs._release_mkdir_lock(gc_lock, dead_token)
    assert os.path.isdir(gc_lock)

    file_jobs._release_mkdir_lock(gc_lock, new_token)
    assert not os.path.isdir(gc_lock)


def test_ready_result_survives_restart_within_ttl(upload_dir):
    job, public_name = _make_published_job(upload_dir)

    # "Restart": a fresh sweep with no in-memory job handle.
    assert file_jobs.sweep(upload_dir, now=time.time())

    with file_jobs.reserve_download(upload_dir, public_name) as path:
        with open(path, "rb") as f:
            assert f.read() == b"translated"


def test_marker_released_when_download_raises(upload_dir):
    job, public_name = _make_published_job(upload_dir)

    with pytest.raises(RuntimeError):
        with file_jobs.reserve_download(upload_dir, public_name):
            raise RuntimeError("boom")

    dl_dir = os.path.join(job.dir, "dl")
    assert not os.path.exists(dl_dir) or os.listdir(dl_dir) == []
