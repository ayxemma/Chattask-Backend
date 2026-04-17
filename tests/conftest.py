"""
Shared test configuration and fixtures.

We set a fake OPENAI_API_KEY in the environment *before* importing the app so
that config.py picks it up at module load time.  Individual tests that want to
test the missing-key path patch the key back to None themselves.
"""
import os

# Must happen before any app module is imported.
os.environ.setdefault("OPENAI_API_KEY", "test-fake-key-do-not-call-openai")

import pytest
from starlette.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """A synchronous TestClient that handles async route handlers automatically."""
    with TestClient(app) as c:
        yield c
