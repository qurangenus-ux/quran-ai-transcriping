"""
Unit tests for the FastAPI API routes.

Uses TestClient with a real JobQueue backed by a temporary database
so no real pipeline or ML model is needed during the test run.
"""

import io
import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.api.routes import create_app


@pytest.fixture
def client(tmp_path):
    """Return a TestClient with a test database and mocked worker."""
    import app.queue.job_queue as jq_module
    from app.database import Database
    from app.queue.job_queue import JobQueue

    db = Database(db_path=str(tmp_path / "api_test.db"))

    mock_worker = MagicMock()
    mock_worker.is_running = True
    mock_worker.is_processing = False

    with patch.object(jq_module, "database", db):
        q = JobQueue()
        with patch("app.api.routes.job_queue", q), \
             patch("app.api.routes.background_worker", mock_worker):
            app = create_app()
            with TestClient(app) as c:
                yield c, q


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    c, _ = client
    response = c.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "worker_running" in data
    assert "queue_size" in data


# ---------------------------------------------------------------------------
# /api/info
# ---------------------------------------------------------------------------

def test_api_info(client):
    c, _ = client
    response = c.get("/api/info")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Quran AI Transcription API"
    assert "endpoints" in data


# ---------------------------------------------------------------------------
# /transcribe/async
# ---------------------------------------------------------------------------

def test_transcribe_async_success(client):
    c, q = client
    audio_content = b"fake audio data"
    response = c.post(
        "/transcribe/async",
        files={"audio_file": ("test.mp3", io.BytesIO(audio_content), "audio/mpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    assert "/jobs/" in data["status_url"]


def test_transcribe_async_no_filename(client):
    """Uploading a file without a filename should return 400 or 422."""
    c, _ = client
    response = c.post(
        "/transcribe/async",
        files={"audio_file": ("", io.BytesIO(b"data"), "audio/mpeg")},
    )
    assert response.status_code in (400, 422)


# ---------------------------------------------------------------------------
# /jobs/{job_id}/status
# ---------------------------------------------------------------------------

def test_job_status_queued(client):
    c, q = client
    job_id = q.create_job("/tmp/audio.mp3", "audio.mp3")
    response = c.get(f"/jobs/{job_id}/status")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "queued"


def test_job_status_not_found(client):
    c, _ = client
    response = c.get("/jobs/non-existent-id/status")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# /jobs/{job_id}/metadata
# ---------------------------------------------------------------------------

def test_job_metadata_not_completed(client):
    c, q = client
    job_id = q.create_job("/tmp/audio.mp3", "audio.mp3")
    response = c.get(f"/jobs/{job_id}/metadata")
    assert response.status_code == 400


def test_job_metadata_completed(client):
    c, q = client
    from app.database import JobStatus

    job_id = q.create_job("/tmp/audio.mp3", "audio.mp3")
    metadata = {"surah_number": 2, "total_ayahs": 5}
    q.update_job_status(
        job_id,
        JobStatus.COMPLETED,
        metadata_json=json.dumps(metadata),
    )
    response = c.get(f"/jobs/{job_id}/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["surah_number"] == 2


def test_job_metadata_not_found(client):
    c, _ = client
    response = c.get("/jobs/bad-id/metadata")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# /jobs/{job_id}/download
# ---------------------------------------------------------------------------

def test_download_not_completed(client):
    c, q = client
    job_id = q.create_job("/tmp/audio.mp3", "audio.mp3")
    response = c.get(f"/jobs/{job_id}/download")
    assert response.status_code == 400


def test_download_result_file_missing(client):
    c, q = client
    from app.database import JobStatus

    job_id = q.create_job("/tmp/audio.mp3", "audio.mp3")
    q.update_job_status(
        job_id,
        JobStatus.COMPLETED,
        result_zip_path="/non/existent/file.zip",
    )
    response = c.get(f"/jobs/{job_id}/download")
    assert response.status_code == 404


def test_download_result_success(client, tmp_path):
    c, q = client
    from app.database import JobStatus

    zip_path = tmp_path / "result.zip"
    zip_path.write_bytes(b"PK fake zip content")

    job_id = q.create_job("/tmp/audio.mp3", "audio.mp3")
    q.update_job_status(
        job_id,
        JobStatus.COMPLETED,
        result_zip_path=str(zip_path),
    )
    response = c.get(f"/jobs/{job_id}/download")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"


# ---------------------------------------------------------------------------
# /jobs (list)
# ---------------------------------------------------------------------------

def test_list_jobs_empty(client):
    c, _ = client
    response = c.get("/jobs")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["jobs"] == []


def test_list_jobs_with_status_filter(client):
    c, q = client
    from app.database import JobStatus

    q.create_job("/tmp/a.mp3", "a.mp3")
    job_id2 = q.create_job("/tmp/b.mp3", "b.mp3")
    q.update_job_status(job_id2, JobStatus.COMPLETED)

    response = c.get("/jobs?status=completed")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1


# ---------------------------------------------------------------------------
# /jobs/resume
# ---------------------------------------------------------------------------

def test_resume_queue(client):
    c, _ = client
    response = c.post("/jobs/resume")
    assert response.status_code == 200
    data = response.json()
    assert "resumed" in data["message"].lower()


# ---------------------------------------------------------------------------
# DELETE /jobs/finished
# ---------------------------------------------------------------------------

def test_clear_finished_jobs(client):
    c, q = client
    from app.database import JobStatus

    job_id = q.create_job("/tmp/done.mp3", "done.mp3")
    q.update_job_status(job_id, JobStatus.COMPLETED)

    response = c.delete("/jobs/finished")
    assert response.status_code == 200
    data = response.json()
    assert data["deleted_count"] >= 1


# ---------------------------------------------------------------------------
# DELETE /jobs/{job_id}
# ---------------------------------------------------------------------------

def test_delete_specific_job(client):
    c, q = client
    job_id = q.create_job("/tmp/todel.mp3", "todel.mp3")
    response = c.delete(f"/jobs/{job_id}")
    assert response.status_code == 200


def test_delete_specific_job_not_found(client):
    c, _ = client
    response = c.delete("/jobs/does-not-exist")
    assert response.status_code == 404

