"""
Shared helper functions for StreamWise app HTTP endpoint tests.

Each helper encapsulates a common assertion so that individual app test files
can delegate to a single implementation rather than repeating the same code.
"""

import os

from http import HTTPStatus

from quart import Quart


async def check_app_root(test_app: Quart, app_title: str) -> None:
    """Check that GET / returns 200 with the correct HTML and app title."""
    client = test_app.test_client()
    response = await client.get("/")
    assert response is not None
    assert response.status_code == HTTPStatus.OK
    assert "text/html; charset=utf-8" == response.content_type
    response_html = await response.get_data(as_text=True)
    assert response_html.startswith("<!DOCTYPE html>\n<html lang=\"en\">")
    assert app_title in response_html


async def check_health(test_app: Quart) -> None:
    """Check that /health returns the expected status dict with no jobs."""
    client = test_app.test_client()
    response = await client.get("/health")
    assert response is not None
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert response_json == {
        "host": None,
        "jobs": {},
        "k8s_cluster": None,
        "port": None,
        "services": {},
        "status": "ok"
    }


async def check_files(test_app: Quart, app_name: str) -> None:
    """Check the /files, /file, /file_stream and /file_view endpoints."""
    client = test_app.test_client()

    # Ensure the tmp directory exists so listing always works
    os.makedirs(f"/tmp/{app_name}", exist_ok=True)

    response = await client.get("/files")
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert "files" in response_json

    response = await client.get("/file/testfile.txt")
    assert response.status_code == HTTPStatus.NOT_FOUND
    response_json = await response.get_json()
    assert response_json == {"error": f"File '/tmp/{app_name}/testfile.txt' not found"}

    response = await client.get("/file_stream/job_id/testfile2.txt")
    assert response.status_code == HTTPStatus.NOT_FOUND
    response_json = await response.get_json()
    assert response_json == {"error": "File not found"}

    response = await client.get("/file_view/job_id/testfile3.txt")
    assert response.status_code == HTTPStatus.OK
    assert response.content_type == "text/html; charset=utf-8"
    response_text = await response.get_data(as_text=True)
    assert response_text.startswith("<!DOCTYPE html>\n<html>")
    assert "<title>File viewer: testfile3.txt</title>" in response_text


async def check_unknown_route(test_app: Quart) -> None:
    """Check that an unknown route returns 404."""
    client = test_app.test_client()
    response = await client.get("/does-not-exist")
    assert response.status_code == HTTPStatus.NOT_FOUND


async def check_job_submit_page(test_app: Quart) -> None:
    """Check that GET /job returns the job submission page."""
    client = test_app.test_client()
    response = await client.get("/job")
    assert response.status_code == HTTPStatus.OK
    text = await response.get_data(as_text=True)
    assert len(text) > 0


async def check_job_status_page(test_app: Quart) -> None:
    """Check that GET /job/<job_id> returns the job status page."""
    client = test_app.test_client()
    response = await client.get("/job/testjobid")
    assert response.status_code == HTTPStatus.OK
    text = await response.get_data(as_text=True)
    assert len(text) > 0


async def check_api_job_status(test_app: Quart) -> None:
    """Check that the API job status endpoint returns a status dict."""
    client = test_app.test_client()
    response = await client.get("/api/job/nonexistent_job_id/status")
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert "status" in response_json


async def check_api_job_requests(test_app: Quart) -> None:
    """Check that the API job requests endpoint returns 200."""
    client = test_app.test_client()
    response = await client.get("/api/job/nonexistent_job_id/requests")
    assert response.status_code == HTTPStatus.OK
