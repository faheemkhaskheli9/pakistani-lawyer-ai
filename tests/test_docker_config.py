"""Static checks for the container setup (issue #13)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_runs_fastapi_and_exposes_healthcheck():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in text
    assert "pip install --no-cache-dir -r requirements.txt" in text
    assert "legal_core.api:app" in text
    assert "/health" in text
    assert "EXPOSE 8000" in text


def test_compose_builds_api_and_maps_port():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "build:" in text
    assert '"8000:8000"' in text
    assert "PAKISTANI_LAWYER_API_KEY" in text
    assert "./data/corpus:/app/data/corpus:ro" in text
    assert "legal_index:/app/data/index" in text


def test_dockerignore_excludes_local_secrets():
    entries = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())
    assert ".env" in entries
    assert ".git" in entries


def test_container_runs_as_non_root_and_has_no_default_api_key():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER appuser" in dockerfile
    assert "legal_core.ingest" in dockerfile
    assert "local-development-key" not in (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

def test_container_rebuilds_index_after_mount_and_uses_configured_path():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert 'CMD ["sh", "-c", "python -m legal_core.ingest' in dockerfile
    assert "--index-dir ${INDEX_DIR:-data/index}" in dockerfile
    assert 'INDEX_DIR:' in compose
