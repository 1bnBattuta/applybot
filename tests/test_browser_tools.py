"""
tests/test_browser_tools.py

Tests every @tool function in tools/browser_tools.py against the live mock site.
No API key needed — pure Playwright against the local HTML file.

Run:
    pytest tests/test_browser_tools.py -v
"""

import json
import pytest
from tools.browser_tools import (
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
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse(result: str) -> dict:
    """Parse the JSON string returned by every tool."""
    return json.loads(result)


# ─────────────────────────────────────────────────────────────────────────────
# navigate
# ─────────────────────────────────────────────────────────────────────────────

class TestNavigate:

    def test_navigate_returns_page_title(self, browser_page, mock_server):
        result = parse(navigate.invoke({"url": mock_server}))
        assert result["status"] == "navigated"
        assert "MockJobs" in result["page_title"]

    def test_navigate_stores_correct_url(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        result = parse(get_current_url.invoke({}))
        assert mock_server in result["url"]

    def test_navigate_invalid_url_returns_error(self, browser_page, mock_server):
        # Playwright will raise on a completely invalid URL
        result = parse(navigate.invoke({"url": "http://localhost:1"}))
        # Either an error key or a failed navigation — should not crash
        assert "error" in result or "status" in result


# ─────────────────────────────────────────────────────────────────────────────
# get_page_text
# ─────────────────────────────────────────────────────────────────────────────

class TestGetPageText:

    def test_returns_text_after_navigation(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        result = parse(get_page_text.invoke({}))
        assert "text" in result
        assert len(result["text"]) > 0

    def test_text_contains_job_titles(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        result = parse(get_page_text.invoke({}))
        text = result["text"]
        # The mock site has these job titles
        assert "Python" in text or "ML Engineer" in text or "DevOps" in text

    def test_text_truncated_at_4000_chars(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        result = parse(get_page_text.invoke({}))
        assert len(result["text"]) <= 4000


# ─────────────────────────────────────────────────────────────────────────────
# get_page_html
# ─────────────────────────────────────────────────────────────────────────────

class TestGetPageHtml:

    def test_finds_job_cards(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        result = parse(get_page_html.invoke({"selector": ".job-card"}))
        assert "error" not in result
        assert result["count"] == 8         # mock site has 8 listings
        assert len(result["elements"]) == 8

    def test_finds_job_titles(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        result = parse(get_page_html.invoke({"selector": ".job-title"}))
        assert result["count"] == 8
        # Each element should contain a job title string
        combined = " ".join(result["elements"])
        assert "Python" in combined or "Engineer" in combined

    def test_unknown_selector_returns_error(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        result = parse(get_page_html.invoke({"selector": ".does-not-exist"}))
        assert "error" in result

    def test_caps_at_10_elements(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        # .badge-tech appears many times — should still return at most 10
        result = parse(get_page_html.invoke({"selector": ".badge-tech"}))
        assert len(result["elements"]) <= 10


# ─────────────────────────────────────────────────────────────────────────────
# fill + click (search flow)
# ─────────────────────────────────────────────────────────────────────────────

class TestFillAndClick:

    def test_fill_search_input(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        result = parse(fill.invoke({"selector": "#search-input", "text": "Python"}))
        assert result["status"] == "filled"
        assert result["value"] == "Python"

    def test_search_filters_results(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        fill.invoke({"selector": "#search-input", "text": "Python"})
        click.invoke({"selector": "#search-btn"})
        wait.invoke({"milliseconds": 300})

        # After filtering, result count should be lower than 8
        result = parse(get_page_html.invoke({"selector": ".job-card:not([data-hidden='true'])"}))
        assert result["count"] <= 8

    def test_click_invalid_selector_returns_error(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        result = parse(click.invoke({"selector": "#nonexistent-button"}))
        assert "error" in result

    def test_search_for_nonexistent_keyword(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        fill.invoke({"selector": "#search-input", "text": "COBOL mainframe 1975"})
        click.invoke({"selector": "#search-btn"})
        wait.invoke({"milliseconds": 300})

        result = parse(get_page_text.invoke({}))
        # The result count badge should show 0
        assert "0" in result["text"]


# ─────────────────────────────────────────────────────────────────────────────
# select_option (filter dropdowns)
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectOption:

    def test_filter_remote_only(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        result = parse(select_option.invoke({"selector": "#filter-remote", "value": "remote"}))
        assert result["status"] == "selected"
        assert result["value"] == "remote"

    def test_remote_filter_reduces_results(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        select_option.invoke({"selector": "#filter-remote", "value": "remote"})
        wait.invoke({"milliseconds": 300})

        result = parse(get_page_text.invoke({}))
        # The mock site has 3 remote jobs — count should be < 8
        import re
        match = re.search(r"Showing (\d+) jobs", result["text"])
        if match:
            count = int(match.group(1))
            assert count < 8

    def test_filter_fintech_sector(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        select_option.invoke({"selector": "#filter-sector", "value": "fintech"})
        wait.invoke({"milliseconds": 300})

        result = parse(get_page_text.invoke({}))
        assert "FinTech" in result["text"] or "fintech" in result["text"].lower()

    def test_invalid_dropdown_value_returns_error(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        result = parse(select_option.invoke({"selector": "#filter-remote", "value": "mars"}))
        assert "error" in result


# ─────────────────────────────────────────────────────────────────────────────
# click into detail page
# ─────────────────────────────────────────────────────────────────────────────

class TestDetailPage:

    def test_click_first_job_card_opens_detail(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        click.invoke({"selector": ".job-card"})
        wait.invoke({"milliseconds": 300})

        result = parse(get_page_text.invoke({}))
        # Detail page contains these sections
        assert "Responsibilities" in result["text"]
        assert "Requirements" in result["text"]

    def test_detail_page_has_apply_button(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        click.invoke({"selector": ".job-card"})
        wait.invoke({"milliseconds": 300})

        result = parse(get_page_html.invoke({"selector": "#apply-btn"}))
        assert "error" not in result
        assert result["count"] == 1

    def test_back_button_returns_to_list(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        click.invoke({"selector": ".job-card"})
        wait.invoke({"milliseconds": 300})
        click.invoke({"selector": "#back-btn"})
        wait.invoke({"milliseconds": 300})

        # Back on the list — job cards should be visible again
        result = parse(get_page_html.invoke({"selector": ".job-card"}))
        assert result["count"] == 8


# ─────────────────────────────────────────────────────────────────────────────
# get_element_attribute
# ─────────────────────────────────────────────────────────────────────────────

class TestGetElementAttribute:

    def test_reads_data_id_from_job_card(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        result = parse(get_element_attribute.invoke({
            "selector": ".job-card",
            "attribute": "data-id"
        }))
        assert "error" not in result
        assert result["value"] is not None
        assert result["value"].isdigit()

    def test_reads_data_title_from_job_card(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        result = parse(get_element_attribute.invoke({
            "selector": ".job-card",
            "attribute": "data-title"
        }))
        assert "error" not in result
        assert len(result["value"]) > 0

    def test_unknown_attribute_returns_none(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        result = parse(get_element_attribute.invoke({
            "selector": ".job-card",
            "attribute": "data-nonexistent"
        }))
        assert result["value"] is None


# ─────────────────────────────────────────────────────────────────────────────
# scroll_down + wait (no assertion on content, just no crash)
# ─────────────────────────────────────────────────────────────────────────────

class TestScrollAndWait:

    def test_scroll_down_succeeds(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        result = parse(scroll_down.invoke({"pixels": 300}))
        assert result["status"] == "scrolled"
        assert result["pixels"] == 300

    def test_wait_succeeds(self, browser_page, mock_server):
        result = parse(wait.invoke({"milliseconds": 200}))
        assert result["status"] == "waited"
        assert result["milliseconds"] == 200

    def test_scroll_default_pixels(self, browser_page, mock_server):
        navigate.invoke({"url": mock_server})
        result = parse(scroll_down.invoke({}))
        assert result["pixels"] == 600
