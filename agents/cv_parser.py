"""
CV Parser Agent
---------------
Takes a path to a CV PDF, extracts structured information using pdfplumber + LangChain
(Claude via ChatAnthropic) with structured output, and returns a UserProfile dict.

Architecture ref: Section 3.1 — CV parser agent
"""

import json
import logging
from pathlib import Path

import pdfplumber
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UserProfile schema — Pydantic model for LangChain structured output
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    """Structured profile extracted from a candidate's CV."""

    job_titles: list[str] = Field(
        description="Job titles the candidate has held or is targeting."
    )
    skills: list[str] = Field(
        description="Technical and soft skills listed in the CV."
    )
    years_of_experience: float = Field(
        description="Total years of professional experience (estimate if not explicit)."
    )
    education: str = Field(
        description="Highest or most relevant degree / diploma."
    )
    languages: list[str] = Field(
        description="Human languages spoken (e.g. French, English)."
    )
    preferred_sectors: list[str] = Field(
        description="Industry sectors the candidate prefers or has experience in."
    )
    notes: str = Field(
        description=(
            "Free-text summary: location preferences, remote policy, "
            "salary expectations, anything not covered by the other fields."
        )
    )


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extract raw text from all pages of a PDF using pdfplumber.
    Raises FileNotFoundError if the path does not exist.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"CV file not found: {pdf_path}")

    pages_text: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())

    full_text = "\n\n".join(pages_text)
    logger.debug("Extracted %d characters from %s", len(full_text), pdf_path.name)
    return full_text


# ---------------------------------------------------------------------------
# LangChain structured extraction
# ---------------------------------------------------------------------------

def _build_chain(llm: ChatAnthropic):
    """
    Build a LangChain chain: prompt | llm.with_structured_output(UserProfile).
    """
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are an expert HR assistant. "
                "Extract structured profile information from the CV text provided by the user. "
                "Infer missing fields (e.g. years_of_experience) from context when possible. "
                "Always fill every field."
            ),
        ),
        (
            "human",
            "Here is the full text extracted from a candidate's CV:\n\n"
            "<cv_text>\n{cv_text}\n</cv_text>\n\n"
            "Extract the structured profile.",
        ),
    ])

    structured_llm = llm.with_structured_output(UserProfile)
    return prompt | structured_llm


def _call_langchain_parse(raw_text: str, llm: ChatAnthropic) -> dict:
    """
    Run the LangChain chain and return the UserProfile as a plain dict.
    """
    chain = _build_chain(llm)
    profile: UserProfile = chain.invoke({"cv_text": raw_text})
    logger.info("UserProfile successfully extracted via LangChain.")
    return profile.dict()


# ---------------------------------------------------------------------------
# Fallback: raw-text-based profile
# ---------------------------------------------------------------------------

def _fallback_profile(raw_text: str) -> dict:
    """
    Return a minimal UserProfile dict populated only with the raw text in `notes`.
    Used when the LLM call fails so the pipeline can continue gracefully.
    """
    logger.warning("Using fallback profile (raw text only).")
    return {
        "job_titles": [],
        "skills": [],
        "years_of_experience": 0,
        "education": "",
        "languages": [],
        "preferred_sectors": [],
        "notes": raw_text[:2000],
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_cv(
    cv_path: str | Path,
    extra_preferences: dict | None = None,
    llm: ChatAnthropic | None = None,
) -> dict:
    """
    Parse a CV PDF and return a UserProfile dict.

    Parameters
    ----------
    cv_path : str | Path
        Path to the CV PDF file.
    extra_preferences : dict, optional
        User-supplied overrides to merge into the profile, e.g.:
        {
            "preferred_location": "Paris",
            "remote_policy": "full-remote",
            "salary_range": "50k-65k EUR",
        }
        These are appended to the `notes` field.
    llm : ChatAnthropic, optional
        Pre-instantiated LangChain ChatAnthropic model.
        If None, a default instance is created (reads ANTHROPIC_API_KEY from env).

    Returns
    -------
    dict
        A UserProfile-compatible dict ready to be passed to downstream agents.

    Example
    -------
    >>> profile = parse_cv("data/john_doe_cv.pdf")
    >>> print(profile["skills"])
    ['Python', 'FastAPI', 'Docker', 'PostgreSQL']
    """
    if llm is None:
        llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=1024)

    # Step 1 — extract raw text
    logger.info("Extracting text from %s", cv_path)
    raw_text = extract_text_from_pdf(cv_path)

    if not raw_text.strip():
        logger.error("PDF appears to be empty or image-only: %s", cv_path)
        profile = _fallback_profile("")
    else:
        # Step 2 — structured extraction via LangChain
        try:
            profile = _call_langchain_parse(raw_text, llm)
        except Exception as exc:
            logger.error("LangChain call failed (%s). Falling back to raw text.", exc)
            profile = _fallback_profile(raw_text)

    # Step 3 — merge extra preferences supplied by the user
    if extra_preferences:
        notes_parts = [profile.get("notes", "")]
        for key, value in extra_preferences.items():
            notes_parts.append(f"{key}: {value}")
        profile["notes"] = " | ".join(filter(None, notes_parts))

    return profile


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    parser = argparse.ArgumentParser(description="Parse a CV PDF into a UserProfile.")
    parser.add_argument("cv_path", help="Path to the CV PDF file.")
    parser.add_argument(
        "--prefs",
        type=str,
        default=None,
        help='Optional JSON string of extra preferences, e.g. \'{"remote_policy":"full-remote"}\'',
    )
    args = parser.parse_args()

    extra = json.loads(args.prefs) if args.prefs else None

    try:
        result = parse_cv(args.cv_path, extra_preferences=extra)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
