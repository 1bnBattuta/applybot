"""
tools/browser_tools.py

Playwright browser actions wrapped as LangChain @tool functions.
The agent in agents/browser_agent.py binds these tools to the LLM.

Each function:
  - Has a clear docstring (LangChain uses it as the tool description)
  - Accepts only simple types (str, int) so LangChain can auto-generate the schema
  - Returns a plain string (the agent reads this as the tool observation)
"""

import asyncio
import json
from playwright.async_api import async_playwright, Page, Browser
from langchain_core.tools import tool

# ─────────────────────────────────────────────────────────────────────────────
# Global browser state — one instance shared across all tool calls in a run
# ─────────────────────────────────────────────────────────────────────────────

_playwright = None
_browser: Browser | None = None
_page: Page | None = None


def _get_event_loop():
    """Return the running event loop, or create one if needed."""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def _run(coro):
    """Run an async coroutine from a synchronous context."""
    loop = _get_event_loop()
    return loop.run_until_complete(coro)


async def _get_page() -> Page:
    """Return the active page, launching a browser if needed."""
    global _playwright, _browser, _page
    if _page is None:
        _playwright = await async_playwright().start()
        _browser = await _playwright.chromium.launch(headless=True)
        _page = await _browser.new_page()
    return _page


async def _close_browser_async():
    global _playwright, _browser, _page
    if _browser:
        await _browser.close()
    if _playwright:
        await _playwright.stop()
    _playwright = _browser = _page = None


def close_browser():
    """Teardown — call once at the end of an agent run."""
    _run(_close_browser_async())


# ─────────────────────────────────────────────────────────────────────────────
# LangChain tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def navigate(url: str) -> str:
    """
    Navigate the browser to a URL.
    Always call this first before any other tool.
    Returns the page title to confirm navigation succeeded.
    """
    async def _navigate():
        page = await _get_page()
        await page.goto(url, wait_until="domcontentloaded")
        title = await page.title()
        return json.dumps({"status": "navigated", "url": url, "page_title": title})
    return _run(_navigate())


@tool
def get_page_text() -> str:
    """
    Return the visible text content of the current page, truncated to 4000 characters.
    Use this to understand page structure or read job listings.
    No arguments needed.
    """
    async def _get():
        page = await _get_page()
        text = await page.inner_text("body")
        truncated = text.strip()[:4000]
        return json.dumps({"text": truncated, "truncated": len(text) > 4000})
    return _run(_get())


@tool
def get_page_html(selector: str) -> str:
    """
    Return the inner HTML of all elements matching a CSS selector.
    Use this to extract structured data from job cards or detail sections.
    Returns up to 10 matching elements, each truncated to 800 characters.
    Example selectors: '.job-card', '#results', '.job-title', 'h1'.
    """
    async def _get():
        page = await _get_page()
        try:
            elements = await page.query_selector_all(selector)
            if not elements:
                return json.dumps({"error": f"No elements found for selector: {selector}"})
            results = []
            for el in elements[:10]:
                html = await el.inner_html()
                results.append(html.strip()[:800])
            return json.dumps({"selector": selector, "count": len(elements), "elements": results})
        except Exception as e:
            return json.dumps({"error": str(e)})
    return _run(_get())


@tool
def click(selector: str) -> str:
    """
    Click the first element matching a CSS selector.
    Use this to open job detail pages, press buttons, or trigger searches.
    Example selectors: '.job-card:first-child', '#search-btn', 'button#apply-btn'.
    """
    async def _click():
        page = await _get_page()
        try:
            el = page.locator(selector).first
            await el.click(timeout=5000)
            await page.wait_for_load_state("domcontentloaded")
            return json.dumps({"status": "clicked", "selector": selector})
        except Exception as e:
            return json.dumps({"error": str(e)})
    return _run(_click())


@tool
def fill(selector: str, text: str) -> str:
    """
    Clear and type text into an input field identified by a CSS selector.
    Use this to enter keywords in search boxes or fill form fields.
    Example: selector='#search-input', text='Python backend'.
    """
    async def _fill():
        page = await _get_page()
        try:
            await page.fill(selector, text)
            return json.dumps({"status": "filled", "selector": selector, "value": text})
        except Exception as e:
            return json.dumps({"error": str(e)})
    return _run(_fill())


@tool
def select_option(selector: str, value: str) -> str:
    """
    Select an option in a <select> dropdown by its value attribute.
    Use this to apply filters (remote policy, sector, experience level).
    Example: selector='#filter-remote', value='remote'.
    """
    async def _select():
        page = await _get_page()
        try:
            await page.select_option(selector, value=value)
            return json.dumps({"status": "selected", "selector": selector, "value": value})
        except Exception as e:
            return json.dumps({"error": str(e)})
    return _run(_select())


@tool
def scroll_down(pixels: int = 600) -> str:
    """
    Scroll down the page by a number of pixels to reveal more content.
    Use when results are lazy-loaded or a 'load more' button is needed.
    Default is 600 pixels.
    """
    async def _scroll():
        page = await _get_page()
        await page.evaluate(f"window.scrollBy(0, {pixels})")
        await asyncio.sleep(0.5)
        return json.dumps({"status": "scrolled", "pixels": pixels})
    return _run(_scroll())


@tool
def wait(milliseconds: int = 1000) -> str:
    """
    Pause execution for a number of milliseconds.
    Use after dynamic actions (search submission, page transitions) to let content load.
    Default is 1000ms (1 second).
    """
    async def _wait():
        await asyncio.sleep(milliseconds / 1000)
        return json.dumps({"status": "waited", "milliseconds": milliseconds})
    return _run(_wait())


@tool
def get_current_url() -> str:
    """
    Return the URL of the currently open page.
    Use this to verify navigation state or log which page is being scraped.
    No arguments needed.
    """
    async def _get():
        page = await _get_page()
        return json.dumps({"url": page.url})
    return _run(_get())


@tool
def get_element_attribute(selector: str, attribute: str) -> str:
    """
    Get an HTML attribute value from the first element matching a CSS selector.
    Use this to extract href links, data-id, data-title, etc.
    Example: selector='.job-card', attribute='data-id'.
    """
    async def _get():
        page = await _get_page()
        try:
            value = await page.get_attribute(selector, attribute)
            return json.dumps({"selector": selector, "attribute": attribute, "value": value})
        except Exception as e:
            return json.dumps({"error": str(e)})
    return _run(_get())


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
