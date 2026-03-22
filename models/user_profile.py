"""
models/user_profile.py
 
Pydantic model for the candidate profile extracted from a CV.
Used as input to the scorer agent.
Compatible with LangChain .with_structured_output().
"""

from typing import Literal
from pydantic import BaseModel, Field

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
    years_of_experience: int = Field(
        ge=0,
        le=50,
        description="Total years of professional experience.",
    )
    seniority: Literal["junior", "mid", "senior"] = Field(
        description=(
            "Derived seniority level. "
            "junior = 0-2 yrs, mid = 3-5 yrs, senior = 5+ yrs."
        ),
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
    remote_preference: Literal["remote", "hybrid", "onsite"] = Field(
        description="Preferred work arrangement.",
    )
    preferred_locations: list[str] = Field(
        description="Preferred cities or regions, e.g. ['Paris', 'Lyon', 'Remote'].",
    )
    salary_min: int | None = Field(
        default=None,
        ge=0,
        description="Minimum acceptable annual salary in EUR. None if not specified.",
    )
    notes: str = Field(
        description=(
            "Free-text summary: location preferences, remote policy, "
            "salary expectations, anything not covered by the other fields."
        )
    )
