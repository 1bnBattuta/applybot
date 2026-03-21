"""
test_scraper.py — Tests for the ScraperAgent.

Run with:
    pytest tests/test_scraper.py -v
"""

import sys
import importlib.util
import pytest
from pathlib import Path


# ── Direct file imports ───────────────────────────────────────────────────────

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

ROOT = Path(__file__).parent.parent

scraper_mod = load_module("scraper_agent", ROOT / "agents" / "scraper_agent.py")
ScraperAgent = scraper_mod.ScraperAgent
JobOffer = scraper_mod.JobOffer  # same class the scraper uses — instanceof will work


# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_HTML_PATH = ROOT / "mock_site" / "mock_job.html"

@pytest.fixture(scope="module")
def mock_html() -> str:
    return MOCK_HTML_PATH.read_text(encoding="utf-8")

@pytest.fixture(scope="module")
def agent() -> ScraperAgent:
    return ScraperAgent()

@pytest.fixture(scope="module")
def extracted_offer(agent, mock_html) -> JobOffer:
    """Run extraction once, reuse across all tests in the module."""
    return agent.extract(mock_html, url="http://localhost/mock_job.html", platform="mock")


# ── Basic extraction tests ────────────────────────────────────────────────────

def test_returns_job_offer(extracted_offer):
    assert isinstance(extracted_offer, JobOffer)

def test_extraction_succeeded(extracted_offer):
    assert extracted_offer.extraction_success is True
    assert extracted_offer.extraction_notes == ""

def test_title_extracted(extracted_offer):
    assert "backend engineer" in extracted_offer.title.lower()

def test_company_extracted(extracted_offer):
    assert "acme" in extracted_offer.company.lower()

def test_location_extracted(extracted_offer):
    assert "paris" in extracted_offer.location.lower()

def test_remote_policy_extracted(extracted_offer):
    assert extracted_offer.remote_policy in ("hybrid", "on-site", "remote", "unknown")
    assert extracted_offer.remote_policy == "hybrid"

def test_salary_extracted(extracted_offer):
    assert extracted_offer.salary_range != ""
    assert "45" in extracted_offer.salary_range or "60" in extracted_offer.salary_range

def test_tech_stack_extracted(extracted_offer):
    stack = [t.lower() for t in extracted_offer.tech_stack]
    assert any("python" in t for t in stack)
    assert any("fastapi" in t or "fast" in t for t in stack)
    assert any("docker" in t for t in stack)

def test_description_is_plain_text(extracted_offer):
    assert "<" not in extracted_offer.description_raw
    assert ">" not in extracted_offer.description_raw
    assert len(extracted_offer.description_raw) > 50

def test_metadata_attached(extracted_offer):
    assert extracted_offer.url == "http://localhost/mock_job.html"
    assert extracted_offer.platform == "mock"

def test_id_and_timestamp_set(extracted_offer):
    assert extracted_offer.id != ""
    assert extracted_offer.scraped_at != ""


# ── Robustness tests ──────────────────────────────────────────────────────────

def test_empty_html_does_not_raise(agent):
    offer = agent.extract("", url="http://test", platform="mock")
    assert isinstance(offer, JobOffer)
    if not offer.extraction_success:
        assert offer.extraction_notes != ""

def test_html_without_salary(agent):
    html = "<h1>Data Engineer</h1><div>OpenAI, London, on-site</div><p>Python, Spark</p>"
    offer = agent.extract(html, url="http://test", platform="mock")
    assert isinstance(offer, JobOffer)
    assert offer.salary_range == "" or offer.salary_range is not None

def test_french_remote_policy(agent):
    html = """
    <h1>Développeur Backend</h1>
    <div>Startup Paris</div>
    <p>Poste en télétravail complet. Stack: Python, Django, PostgreSQL.</p>
    """
    offer = agent.extract(html, url="http://test", platform="wttj")
    assert offer.remote_policy == "remote"

def test_dom_change_resilience(agent):
    restructured_html = """
    <article>
      <section data-qa="header">
        <span data-v-abc123>Senior Python Developer</span>
        <a data-v-xyz789>TechCorp</a>
      </section>
      <section data-qa="details">
        <span>Lyon, France</span>
        <span>Remote</span>
        <span>50k–70k EUR</span>
      </section>
      <div data-qa="body">
        We need a Python expert. Required: Python, FastAPI, AWS, Terraform, PostgreSQL.
      </div>
    </article>
    """
    offer = agent.extract(restructured_html, url="http://test", platform="indeed")
    assert "python" in offer.title.lower() or "developer" in offer.title.lower()
    assert offer.remote_policy == "remote"
    stack = [t.lower() for t in offer.tech_stack]
    assert any("python" in t for t in stack)


# ── Serialisation test ────────────────────────────────────────────────────────

def test_to_dict_roundtrip(extracted_offer):
    d = extracted_offer.to_dict()
    restored = JobOffer.from_dict(d)
    assert restored.title == extracted_offer.title
    assert restored.company == extracted_offer.company
    assert restored.tech_stack == extracted_offer.tech_stack