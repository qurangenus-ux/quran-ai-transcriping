"""
Unit tests for the Database class.
"""

import pytest
import tempfile
import os
from app.database import Database, JobStatus


@pytest.fixture
def db(tmp_path):
    """Create a temporary in-memory database for each test."""
    db_file = str(tmp_path / "test_jobs.db")
    return Database(db_path=db_file)


def test_create_job(db):
    """Test that a job can be created and returns a UUID."""
    job_id = db.create_job("recitation.mp3", "/tmp/recitation.mp3")
    assert job_id is not None
    assert len(job_id) == 36  # UUID format


def test_get_job(db):
    """Test retrieving a job by ID."""
    job_id = db.create_job("test.wav", "/tmp/test.wav")
    job = db.get_job(job_id)
    assert job is not None
    assert job["id"] == job_id
    assert job["original_filename"] == "test.wav"
    assert job["audio_file_path"] == "/tmp/test.wav"
    assert job["status"] == JobStatus.QUEUED


def test_get_job_not_found(db):
    """Test that None is returned for a missing job."""
    result = db.get_job("non-existent-id")
    assert result is None


def test_update_job_status_processing(db):
    """Test updating job status to processing."""
    job_id = db.create_job("test.mp3", "/tmp/test.mp3")
    db.update_job_status(job_id, JobStatus.PROCESSING)
    job = db.get_job(job_id)
    assert job["status"] == JobStatus.PROCESSING
    assert job["started_at"] is not None


def test_update_job_status_completed(db):
    """Test updating job status to completed with result path."""
    job_id = db.create_job("test.mp3", "/tmp/test.mp3")
    db.update_job_status(
        job_id,
        JobStatus.COMPLETED,
        result_zip_path="/tmp/result.zip",
        transcription_text="بسم الله",
        metadata_json='{"surah": 1}'
    )
    job = db.get_job(job_id)
    assert job["status"] == JobStatus.COMPLETED
    assert job["result_zip_path"] == "/tmp/result.zip"
    assert job["transcription_text"] == "بسم الله"
    assert job["metadata_json"] == '{"surah": 1}'
    assert job["completed_at"] is not None


def test_update_job_status_failed(db):
    """Test updating job status to failed with error message."""
    job_id = db.create_job("test.mp3", "/tmp/test.mp3")
    db.update_job_status(job_id, JobStatus.FAILED, error_message="Transcription error")
    job = db.get_job(job_id)
    assert job["status"] == JobStatus.FAILED
    assert job["error_message"] == "Transcription error"
    assert job["completed_at"] is not None


def test_get_next_queued_job_fifo(db):
    """Test that get_next_queued_job returns jobs in FIFO order."""
    id1 = db.create_job("first.mp3", "/tmp/first.mp3")
    id2 = db.create_job("second.mp3", "/tmp/second.mp3")
    next_job = db.get_next_queued_job()
    assert next_job["id"] == id1


def test_get_next_queued_job_empty(db):
    """Test that None is returned when queue is empty."""
    result = db.get_next_queued_job()
    assert result is None


def test_get_all_jobs(db):
    """Test retrieving all jobs."""
    db.create_job("a.mp3", "/tmp/a.mp3")
    db.create_job("b.mp3", "/tmp/b.mp3")
    jobs = db.get_all_jobs()
    assert len(jobs) == 2


def test_delete_job(db):
    """Test deleting a job."""
    job_id = db.create_job("del.mp3", "/tmp/del.mp3")
    db.delete_job(job_id)
    assert db.get_job(job_id) is None


def test_get_finished_jobs(db):
    """Test retrieving completed and failed jobs."""
    id1 = db.create_job("c.mp3", "/tmp/c.mp3")
    id2 = db.create_job("f.mp3", "/tmp/f.mp3")
    id3 = db.create_job("q.mp3", "/tmp/q.mp3")
    db.update_job_status(id1, JobStatus.COMPLETED)
    db.update_job_status(id2, JobStatus.FAILED)
    finished = db.get_finished_jobs()
    ids = [j["id"] for j in finished]
    assert id1 in ids
    assert id2 in ids
    assert id3 not in ids


def test_reset_processing_jobs_to_queued(db):
    """Test resetting processing jobs back to queued."""
    job_id = db.create_job("r.mp3", "/tmp/r.mp3")
    db.update_job_status(job_id, JobStatus.PROCESSING)
    count = db.reset_processing_jobs_to_queued()
    assert count == 1
    job = db.get_job(job_id)
    assert job["status"] == JobStatus.QUEUED
    assert job["started_at"] is None
