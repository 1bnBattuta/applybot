"""
tools/browser_tools.py

Playwright browser actions wrapped as LangChain @tool functions.
Uses playwright.sync_api — no async, no event loop juggling.

LangChain @tool is synchronous by default, and the agent loop calls
llm.invoke() (sync), so there is no reason for async here.
The sync_api approach also makes testing trivial: the conftest fixture
injects a sync Page and the tools use it directly.
"""

import json
import time
from playwright.sync_api import sync_playwright, Page, Browser, Playwright
from langchain_core.tools import tool

# ─────────────────────────────────────────────────────────────────────────────
# Global browser state — one instance shared across all tool calls in a run
# ─────────────────────────────────────────────────────────────────────────────

_playwright: Playwright | None = None
_browser: Browser | None = None
_page: Page | None = None


def _get_page() -> Page:
    """Return the active sync page, launching a browser if needed."""
    global _playwright, _browser, _page
    if _page is None:
        _playwright = sync_playwright().start()
        _browser = _playwright.chromium.launch(headless=True)
        _page = _browser.new_page()
    return _page


def close_browser():
    """Teardown — call once at the end of an agent run."""
    global _playwright, _browser, _page
    if _browser:
        _browser.close()
    if _playwright:
        _playwright.stop()
    _playwright = _browser = _page = None


# ─────────────────────────────────────────────────────────────────────────────
# LangChain tools — all synchronous
# ─────────────────────────────────────────────────────────────────────────────

@tool
def navigate(url: str) -> str:
    """
    Navigate the browser to a URL.
    Always call this first before any other tool.
    Returns the page title to confirm navigation succeeded.
    """
    page = _get_page()
    try:
        page.goto(url, wait_until="domcontentloaded")
        title = page.title()
        return json.dumps({"status": "navigated", "url": url, "page_title": title})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_page_text() -> str:
    """
    Return the visible text content of the current page, truncated to 4000 characters.
    Use this to understand page structure or read job listings.
    No arguments needed.
    """
    page = _get_page()
    try:
        text = page.inner_text("body").strip()
        return json.dumps({"text": text[:4000], "truncated": len(text) > 4000})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_page_html(selector: str) -> str:
    """
    Return the inner HTML of all elements matching a CSS selector.
    Use this to extract structured data from job cards or detail sections.
    Returns up to 10 matching elements, each truncated to 800 characters.
    Example selectors: '.job-card', '#results', '.job-title', 'h1'.
    """
    page = _get_page()
    try:
        elements = page.query_selector_all(selector)
        if not elements:
            return json.dumps({"error": f"No elements found for selector: {selector}"})
        results = [el.inner_html().strip()[:800] for el in elements[:10]]
        return json.dumps({"selector": selector, "count": len(elements), "elements": results})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def click(selector: str) -> str:
    """
    Click the first element matching a CSS selector.
    Use this to open job detail pages, press buttons, or trigger searches.
    Example selectors: '.job-card:first-child', '#search-btn', 'button#apply-btn'.
    """
    page = _get_page()
    try:
        page.locator(selector).first.click(timeout=5000)
        page.wait_for_load_state("domcontentloaded")
        return json.dumps({"status": "clicked", "selector": selector})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def fill(selector: str, text: str) -> str:
    """
    Clear and type text into an input field identified by a CSS selector.
    Use this to enter keywords in search boxes or fill form fields.
    Example: selector='#search-input', text='Python backend'.
    """
    page = _get_page()
    try:
        page.fill(selector, text)
        return json.dumps({"status": "filled", "selector": selector, "value": text})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def select_option(selector: str, value: str) -> str:
    """
    Select an option in a <select> dropdown by its value attribute.
    Use this to apply filters (remote policy, sector, experience level).
    Example: selector='#filter-remote', value='remote'.
    """
    page = _get_page()
    try:
        page.select_option(selector, value=value)
        return json.dumps({"status": "selected", "selector": selector, "value": value})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def scroll_down(pixels: int = 600) -> str:
    """
    Scroll down the page by a number of pixels to reveal more content.
    Use when results are lazy-loaded or a load more button is needed.
    Default is 600 pixels.
    """
    page = _get_page()
    try:
        page.evaluate(f"window.scrollBy(0, {pixels})")
        time.sleep(0.5)
        return json.dumps({"status": "scrolled", "pixels": pixels})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def wait(milliseconds: int = 1000) -> str:
    """
    Pause execution for a number of milliseconds.
    Use after dynamic actions (search submission, page transitions) to let content load.
    Default is 1000ms (1 second).
    """
    time.sleep(milliseconds / 1000)
    return json.dumps({"status": "waited", "milliseconds": milliseconds})


@tool
def get_current_url() -> str:
    """
    Return the URL of the currently open page.
    Use this to verify navigation state or log which page is being scraped.
    No arguments needed.
    """
    page = _get_page()
    return json.dumps({"url": page.url})


@tool
def get_element_attribute(selector: str, attribute: str) -> str:
    """
    Get an HTML attribute value from the first element matching a CSS selector.
    Use this to extract href links, data-id, data-title, etc.
    Example: selector='.job-card', attribute='data-id'.
    """
    page = _get_page()
    try:
        value = page.get_attribute(selector, attribute)
        return json.dumps({"selector": selector, "attribute": attribute, "value": value})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# Exported list — pass this to llm.bind_tools() in the agent
# ─────────────────────────────────────────────────────────────────────────────

BROWSER_TOOLS = [
    navigate,
    get_page_text,
    get_page_html,
    click,
    fill,
    select_option,
    scroll_down,
    wait,
    get_current_url,
    get_element_attribute,
]
