"""
tests/conftest.py

Shared fixtures for all test files.

browser_tools.py now uses playwright.sync_api throughout, so the fixture
simply injects a sync Page into the module globals. No event loop conflicts.
"""

import functools
import http.server
import socket
import threading
import time
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

MOCK_SITE_DIR = Path(__file__).parent.parent / "mock_site"


def _get_free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def mock_server():
    """Serve mock_site/ over HTTP for the whole test session."""
    port = _get_free_port()
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(MOCK_SITE_DIR),
    )
    handler.log_message = lambda *args: None

    server = http.server.HTTPServer(("localhost", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)

    url = f"http://localhost:{port}"
    print(f"\n[conftest] Mock server → {url}")
    yield url
    server.shutdown()


@pytest.fixture
def browser_page(mock_server):
    """
    Fresh sync Playwright page per test.
    Injects it into browser_tools module globals so every @tool picks it up.
    Cleans up after the test completes.
    """
    import tools.browser_tools as bt

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()

        # Inject into the module so _get_page() returns this page immediately
        bt._playwright = pw
        bt._browser = browser
        bt._page = page

        yield page

        browser.close()

    # Reset module state so the next test gets a clean slate
    bt._playwright = None
    bt._browser = None
    bt._page = None
