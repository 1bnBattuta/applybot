"""
scraper_agent.py — Extracts structured JobOffer objects from raw HTML using LangChain + GitHub Models.

Design principle: instead of brittle CSS selectors, raw HTML is passed directly
to the LLM, which fills the JobOffer schema. This makes extraction resilient to
minor DOM changes across platforms.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from models.job_offer import JobOffer

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)


# ── Pydantic schema (used by LangChain's JsonOutputParser) ───────────────────

class JobOfferSchema(BaseModel):
    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    location: str = Field(description="City, Country")
    remote_policy: str = Field(description="one of: on-site | hybrid | remote | unknown")
    tech_stack: list[str] = Field(description="All mentioned technologies, frameworks, languages, tools")
    salary_range: str = Field(description="e.g. 45k–60k EUR, or empty string if not mentioned")
    description_raw: str = Field(description="Full job description text, stripped of HTML tags")


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert at extracting structured data from job listing HTML pages.
Your task is to extract job offer information and return it as a valid JSON object.
Respond ONLY with a JSON object — no preamble, no markdown, no explanation."""

EXTRACTION_PROMPT = """Extract the job offer details from the following HTML and return a JSON object
matching this exact schema:

{{
  "title": "Job title (string)",
  "company": "Company name (string)",
  "location": "City, Country (string)",
  "remote_policy": "one of: on-site | hybrid | remote | unknown",
  "tech_stack": ["list", "of", "technologies"],
  "salary_range": "e.g. 45k–60k EUR, or empty string if not mentioned",
  "description_raw": "Full job description text, cleaned of HTML tags"
}}

Rules:
- If a field is not found, use an empty string (or empty list for tech_stack).
- For remote_policy, infer from keywords like "télétravail", "remote", "hybrid", "on-site", "présentiel".
- For tech_stack, extract ALL mentioned technologies, frameworks, languages, and tools.
- description_raw should be plain text only — strip all HTML tags.

HTML to extract from:
---
{html}
---"""


# ── Main agent ────────────────────────────────────────────────────────────────

class ScraperAgent:
    """
    Extracts a JobOffer from raw HTML using LangChain + GitHub Models (Azure).

    Usage:
        agent = ScraperAgent()
        offer = agent.extract(html, url="https://...", platform="wttj")
    """

    def __init__(self, model: str = "gpt-4.1-mini", max_html_chars: int = 40_000):
        self.max_html_chars = max_html_chars

        llm = ChatOpenAI(
            model=model,
            openai_api_key=os.environ["AI_API_KEY"],
            openai_api_base=os.environ["AI_ENDPOINT"],
        )
        parser = JsonOutputParser(pydantic_object=JobOfferSchema)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", EXTRACTION_PROMPT),
        ])

        # LangChain LCEL chain: prompt | llm | parser
        self.chain = prompt | llm | parser

    def extract(self, html: str, url: str = "", platform: str = "") -> JobOffer:
        """
        Main entry point. Returns a JobOffer — always, even on failure.
        On failure, extraction_success=False and extraction_notes explains why.
        """
        truncated_html = html[: self.max_html_chars]

        try:
            data = self.chain.invoke({"html": truncated_html})
            offer = JobOffer(
                title=data.get("title", ""),
                company=data.get("company", ""),
                location=data.get("location", ""),
                remote_policy=data.get("remote_policy", "unknown"),
                tech_stack=data.get("tech_stack", []),
                salary_range=data.get("salary_range", ""),
                description_raw=data.get("description_raw", ""),
                extraction_success=True,
            )
        except Exception as e:
            logger.warning(f"Extraction failed for {url}: {e}")
            offer = JobOffer(
                extraction_success=False,
                extraction_notes=str(e),
            )

        # Attach metadata that the LLM doesn't know about
        offer.url = url
        offer.platform = platform
        return offer