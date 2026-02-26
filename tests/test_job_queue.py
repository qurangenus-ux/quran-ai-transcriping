"""
Unit tests for the JobQueue class.
"""

import pytest
import json
from unittest.mock import patch
from app.database import Database, JobStatus


@pytest.fixture
def queue(tmp_path):
    """Create a JobQueue backed by a temporary database."""
    import app.queue.job_queue as jq_module
    from app.queue.job_queue import JobQueue

    db = Database(db_path=str(tmp_path / "test_queue.db"))
    with patch.object(jq_module, "database", db):
        yield JobQueue()


def test_create_job_returns_id(queue):
    """Test that create_job returns a valid job ID."""
    job_id = queue.create_job("/tmp/audio.mp3", "audio.mp3")
    assert job_id is not None
    assert len(job_id) == 36


def test_get_job_maps_id_field(queue):
    """Test that get_job maps 'id' field to 'job_id'."""
    job_id = queue.create_job("/tmp/audio.mp3", "audio.mp3")
    job = queue.get_job(job_id)
    assert job is not None
    assert "job_id" in job
    assert job["job_id"] == job_id
    assert "id" not in job


def test_get_job_not_found(queue):
    """Test that None is returned for a missing job."""
    result = queue.get_job("missing-id")
    assert result is None


def test_get_all_jobs(queue):
    """Test retrieving all jobs with mapped fields."""
    queue.create_job("/tmp/a.mp3", "a.mp3")
    queue.create_job("/tmp/b.mp3", "b.mp3")
    jobs = queue.get_all_jobs()
    assert len(jobs) == 2
    for job in jobs:
        assert "job_id" in job
        assert "id" not in job


def test_get_queue_size(queue):
    """Test that queue size counts only queued jobs."""
    queue.create_job("/tmp/a.mp3", "a.mp3")
    job_id = queue.create_job("/tmp/b.mp3", "b.mp3")
    queue.update_job_status(job_id, JobStatus.PROCESSING)
    assert queue.get_queue_size() == 1


def test_delete_job(queue):
    """Test deleting a job."""
    job_id = queue.create_job("/tmp/del.mp3", "del.mp3")
    queue.delete_job(job_id)
    assert queue.get_job(job_id) is None


def test_get_job_metadata_parses_json(queue):
    """Test that get_job_metadata parses metadata JSON."""
    job_id = queue.create_job("/tmp/m.mp3", "m.mp3")
    metadata = {"surah_number": 1, "total_ayahs": 7}
    queue.update_job_status(
        job_id,
        JobStatus.COMPLETED,
        metadata_json=json.dumps(metadata)
    )
    result = queue.get_job_metadata(job_id)
    assert result is not None
    assert result["surah_number"] == 1
    assert result["total_ayahs"] == 7


def test_get_job_metadata_no_metadata(queue):
    """Test that None is returned when no metadata exists."""
    job_id = queue.create_job("/tmp/nm.mp3", "nm.mp3")
    result = queue.get_job_metadata(job_id)
    assert result is None


def test_is_job_complete(queue):
    """Test is_job_complete returns True for completed jobs."""
    job_id = queue.create_job("/tmp/c.mp3", "c.mp3")
    queue.update_job_status(job_id, JobStatus.COMPLETED)
    assert queue.is_job_complete(job_id) is True


def test_is_job_failed(queue):
    """Test is_job_failed returns True for failed jobs."""
    job_id = queue.create_job("/tmp/f.mp3", "f.mp3")
    queue.update_job_status(job_id, JobStatus.FAILED)
    assert queue.is_job_failed(job_id) is True


def test_is_job_processing(queue):
    """Test is_job_processing returns True for processing jobs."""
    job_id = queue.create_job("/tmp/p.mp3", "p.mp3")
    queue.update_job_status(job_id, JobStatus.PROCESSING)
    assert queue.is_job_processing(job_id) is True


def test_reset_processing_jobs(queue):
    """Test that processing jobs are reset to queued."""
    job_id = queue.create_job("/tmp/r.mp3", "r.mp3")
    queue.update_job_status(job_id, JobStatus.PROCESSING)
    queue.reset_processing_jobs()
    job = queue.get_job(job_id)
    assert job["status"] == JobStatus.QUEUED

