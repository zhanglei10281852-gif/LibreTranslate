import io
import json
import os
from urllib.parse import urlsplit

import argostranslatefiles
import pytest


@pytest.fixture()
def fake_file_translation(monkeypatch):
    """Drive file formats without running an actual translation model."""

    def fake_translate_file(
        underlying_translation, file_path, get_output_path=None
    ):
        if get_output_path is not None:
            out_path = get_output_path(underlying_translation, file_path)
        else:
            base, ext = os.path.splitext(file_path)
            out_path = base + "_es" + ext

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\xa1Hola mundo!")
        return out_path

    def fail_translate_file(
        underlying_translation, file_path, get_output_path=None
    ):
        if get_output_path is not None:
            out_path = get_output_path(underlying_translation, file_path)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("half output")
        raise RuntimeError("translation failed")

    fake_file_translation.translate_file = fake_translate_file
    fake_file_translation.fail_translate_file = fail_translate_file

    monkeypatch.setattr(
        argostranslatefiles, "translate_file", fake_translate_file
    )
    monkeypatch.setattr(
        argostranslatefiles, "get_texts", lambda file_path: "Hello world"
    )
    return fake_file_translation


def _upload(client, filename="hello.txt", source="en", target="es",
           content=b"Hello world"):
    return client.post(
        "/translate_file",
        data={
            "source": source,
            "target": target,
            "file": (io.BytesIO(content), filename),
        },
        content_type="multipart/form-data",
    )


def test_translate_file_publishes_and_downloads(client, fake_file_translation):
    from libretranslate.app import get_upload_dir
    from libretranslate import file_jobs

    upload_dir = get_upload_dir()

    response = _upload(client)
    assert response.status_code == 200

    body = json.loads(response.data)
    assert "translatedFileUrl" in body

    download_path = urlsplit(body["translatedFileUrl"]).path
    assert download_path.endswith("hello_es.txt")

    # Only the published result is visible at the top level; the input and
    # candidate live (briefly) under .jobs/<uuid>/ and are never public.
    top_level_files = [
        n for n in os.listdir(upload_dir)
        if os.path.isfile(os.path.join(upload_dir, n))
    ]
    public_name = os.path.basename(download_path)
    assert public_name in top_level_files
    assert not any("input" in n or "candidate" in n for n in top_level_files)

    job_id = public_name.split(".", 1)[0]
    job_dir = os.path.join(upload_dir, ".jobs", job_id)
    with open(os.path.join(job_dir, "state.json")) as f:
        state = json.load(f)
    assert state["status"] == file_jobs.STATUS_READY
    # The source is removed once the result is published.
    assert not os.path.exists(
        os.path.join(job_dir, "input" + state["ext"])
    )

    download = client.get(download_path)
    assert download.status_code == 200
    assert download.data.decode("utf-8") == "\xa1Hola mundo!"
    disposition = download.headers["Content-Disposition"]
    assert "attachment" in disposition
    # Original filename download semantics: UUID prefix stripped.
    assert "hello_es.txt" in disposition


def test_translate_file_auto_detection(client, fake_file_translation):
    response = _upload(client, source="auto")
    assert response.status_code == 200

    download_path = urlsplit(
        json.loads(response.data)["translatedFileUrl"]
    ).path
    download = client.get(download_path)
    assert download.status_code == 200
    assert download.data.decode("utf-8") == "\xa1Hola mundo!"


def test_translate_file_failure_cleans_unpublished_artifacts(
    client, fake_file_translation, monkeypatch
):
    from libretranslate.app import get_upload_dir

    upload_dir = get_upload_dir()
    jobs_dir = os.path.join(upload_dir, ".jobs")

    # The sweeper's election lock briefly appears under .jobs; ignore it.
    jobs_before = {n for n in os.listdir(jobs_dir) if n != ".gc.lock"}
    files_before = {
        n for n in os.listdir(upload_dir)
        if os.path.isfile(os.path.join(upload_dir, n))
    }

    monkeypatch.setattr(
        argostranslatefiles,
        "translate_file",
        fake_file_translation.fail_translate_file,
    )

    response = _upload(client)
    assert response.status_code == 500
    assert "error" in json.loads(response.data)

    # No orphaned job directory, source or half-written candidate remains.
    jobs_after = {n for n in os.listdir(jobs_dir) if n != ".gc.lock"}
    files_after = {
        n for n in os.listdir(upload_dir)
        if os.path.isfile(os.path.join(upload_dir, n))
    }
    assert jobs_after == jobs_before
    assert files_after == files_before


def test_download_missing_returns_stable_404(client):
    response = client.get(
        "/download_file/00000000-0000-0000-0000-000000000000.hello_es.txt"
    )
    assert response.status_code == 404
    body = json.loads(response.data)
    assert "error" in body


def test_download_invalid_identifier_returns_404(client):
    response = client.get("/download_file/not-a-uuid.hello_es.txt")
    assert response.status_code == 404
    assert "error" in json.loads(response.data)


def test_translate_file_unsupported_format(client, fake_file_translation):
    response = _upload(client, filename="virus.exe")
    assert response.status_code == 400
    assert "error" in json.loads(response.data)


def test_translate_file_missing_parameters(client, fake_file_translation):
    response = client.post(
        "/translate_file",
        data={"target": "es"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400

    response = client.post(
        "/translate_file",
        data={"source": "en", "target": "es"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_translate_file_unsupported_language(client, fake_file_translation):
    response = _upload(client, target="zz")
    assert response.status_code == 400
    assert "error" in json.loads(response.data)
