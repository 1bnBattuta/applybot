"""
agents/scorer_agent.py

Scores a list of JobOffer objects against a UserProfile.
Uses LangChain .with_structured_output(JobScore) so the LLM returns
a validated, typed object with no manual JSON parsing.

Usage:
    from agents.scorer_agent import score_offers
    results = score_offers(profile, offers)
    for r in results:
        print(r.offer.title, r.score.summary())
"""

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from models.job_offer import JobOffer
from models.score import JobScore
from models.user_profile import UserProfile

load_dotenv()
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

AI_API_KEY  = os.getenv("AI_API_KEY")
AI_ENDPOINT = os.getenv("AI_ENDPOINT", "https://models.inference.ai.azure.com")
AI_MODEL    = os.getenv("AI_MODEL", "gpt-4.1-mini")
MAX_TOKENS  = int(os.getenv("MAX_TOKENS", "2048"))


# ─────────────────────────────────────────────────────────────────────────────
# Output container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoredOffer:
    """A JobOffer paired with its JobScore. Sortable by score.total."""
    offer: JobOffer
    score: JobScore

    def __lt__(self, other: "ScoredOffer") -> bool:
        return self.score.total < other.score.total


# ─────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ─────────────────────────────────────────────────────────────────────────────

_REMOTE_ALIASES = {
    "on-site":     "onsite",
    "on site":     "onsite",
    "présentiel":  "onsite",
    "full remote": "remote",
    "fully remote":"remote",
    "télétravail": "remote",
    "teletravail": "remote",
    "hybride":     "hybrid",
}

def _normalise_remote(value: str) -> str:
    return _REMOTE_ALIASES.get(value.lower().strip(), value.lower().strip() or "unknown")


def _offer_to_prompt(offer: JobOffer) -> str:
    """Format a JobOffer as a compact string for the scoring prompt."""
    lines = [
        f"Title: {offer.title}",
        f"Company: {offer.company}",
        f"Location: {offer.location}",
        f"Remote policy: {_normalise_remote(offer.remote_policy)}",
    ]
    if offer.tech_stack:
        lines.append(f"Tech stack: {', '.join(offer.tech_stack)}")
    if offer.salary_range:
        lines.append(f"Salary: {offer.salary_range}")
    if offer.description_raw:
        # Cap description to keep prompt size reasonable
        lines.append(f"Description (truncated):\n{offer.description_raw[:1500]}")
    return "\n".join(lines)


def _profile_to_prompt(profile: UserProfile) -> str:
    """Format a UserProfile as a compact string for the scoring prompt."""
    lines = [
        f"Titles: {', '.join(profile.job_titles)}",
        f"Skills: {', '.join(profile.skills)}",
        f"Experience: {profile.years_of_experience} years ({profile.seniority})",
        f"Education: {profile.education}",
        f"Languages: {', '.join(profile.languages)}",
        f"Preferred sectors: {', '.join(profile.preferred_sectors)}",
        f"Remote preference: {profile.remote_preference}",
        f"Preferred locations: {', '.join(profile.preferred_locations)}",
    ]
    if profile.salary_min:
        # TODO: Add a way to support multiple currencies
        lines.append(f"Minimum salary: {profile.salary_min:,} EUR/year")
    if profile.notes:
        lines.append(f"Notes: {profile.notes}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Salary flag helper (no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

def _check_salary_flag(profile: UserProfile, offer: JobOffer) -> bool:
    """
    Return True if the offer's salary range appears below the candidate's minimum.
    Parsed heuristically from strings like "45k–60k EUR" or "45 000 – 60 000 €".
    Returns False if either salary is not stated.
    """
    if not profile.salary_min or not offer.salary_range:
        return False

    import re
    nums = re.findall(r"[\d\s]+", offer.salary_range.replace(",", "").replace(".", ""))
    values = []
    for n in nums:
        n = n.replace(" ", "")
        if n.isdigit():
            v = int(n)
            # Handle "45k" → 45000
            if v < 1000:
                v *= 1000
            values.append(v)

    if not values:
        return False

    offer_max = max(values)
    return offer_max < profile.salary_min


# ─────────────────────────────────────────────────────────────────────────────
# Core scoring function
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert technical recruiter. Your task is to score how well a job offer \
matches a candidate profile across four dimensions (0–25 each, total 100).

Scoring guide:
- technical_fit   : overlap between candidate skills and job tech stack (0=no overlap, 25=full match)
- seniority_match : years of experience vs role level (penalise both under and over-qualification)
- remote_match    : candidate remote preference vs job policy (exact=25, adjacent=12, opposite=0)
- sector_match    : preferred sectors vs company sector + culture signals

Be specific in your rationale — cite actual skills, years, or policies from both sides.\
"""

HUMAN_PROMPT = """\
## Candidate profile
{profile}

## Job offer
{offer}

Score this job offer against this candidate profile.\
"""


def _build_chain(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  HUMAN_PROMPT),
    ])
    return prompt | llm.with_structured_output(JobScore)


def score_offer(
    profile: UserProfile,
    offer: JobOffer,
    llm: ChatOpenAI | None = None,
) -> ScoredOffer | None:
    """
    Score a single JobOffer against a UserProfile.

    Returns None if the offer failed extraction or has no title
    (these should never reach the scorer, but guard here just in case).
    """
    if not offer.extraction_success or not offer.title.strip():
        logger.warning("Skipping unscoreable offer: %s", offer.url)
        return None

    if llm is None:
        llm = ChatOpenAI(
            model=AI_MODEL,
            api_key=AI_API_KEY,
            base_url=AI_ENDPOINT,
            max_tokens=MAX_TOKENS,
        )

    chain = _build_chain(llm)

    try:
        job_score: JobScore = chain.invoke({
            "profile": _profile_to_prompt(profile),
            "offer":   _offer_to_prompt(offer),
        })

        # Attach salary flag (computed locally, not by LLM)
        job_score.salary_flag = _check_salary_flag(profile, offer)

        logger.info("Scored '%s': %s", offer.title, job_score.summary())
        return ScoredOffer(offer=offer, score=job_score)

    except Exception as e:
        logger.error("Scoring failed for '%s': %s", offer.title, e)
        return None


def score_offers(
    profile: UserProfile,
    offers: list[JobOffer],
    llm: ChatOpenAI | None = None,
    skip_failed: bool = True,
) -> list[ScoredOffer]:
    """
    Score a list of JobOffer objects and return them ranked best-first.

    Args:
        profile:      Candidate profile from the CV parser.
        offers:       List of JobOffer from the scraper agent.
        llm:          Optional pre-built LLM instance (shared across calls).
        skip_failed:  If True, silently drop offers where scoring fails.

    Returns:
        List of ScoredOffer sorted by total score descending.
    """
    if llm is None:
        llm = ChatOpenAI(
            model=AI_MODEL,
            api_key=AI_API_KEY,
            base_url=AI_ENDPOINT,
            max_tokens=MAX_TOKENS,
        )

    results: list[ScoredOffer] = []

    for i, offer in enumerate(offers):
        logger.info("Scoring %d/%d: %s", i + 1, len(offers), offer.title)
        result = score_offer(profile, offer, llm=llm)
        if result is not None:
            results.append(result)
        elif not skip_failed:
            raise RuntimeError(f"Scoring failed for: {offer.title}")

    ranked = sorted(results, reverse=True)
    logger.info(
        "Scoring complete: %d/%d offers scored, top score: %s",
        len(ranked), len(offers),
        ranked[0].score.summary() if ranked else "N/A",
    )
    return ranked
