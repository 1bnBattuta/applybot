"""
models/score.py

Pydantic models for the scorer agent output.
These are passed directly to .with_structured_output() so the LLM
returns a validated, typed JobScore object with no manual parsing.
"""

from pydantic import BaseModel, Field, model_validator


class DimensionScore(BaseModel):
    """Score and rationale for one evaluation dimension (0–25)."""

    score: int = Field(
        ge=0,
        le=25,
        description=(
            "Score from 0 to 25. "
            "0 = no match at all, 25 = perfect match."
        ),
    )

    rationale: str = Field(
        description=(
            "One or two sentences explaining why this score was given. "
            "Be specific: cite skills, years, or policies from the offer and profile."
        ),
    )


class JobScore(BaseModel):
    """
    Full scoring of one job offer against one candidate profile.
    Total score = sum of the 4 dimension scores (max 100).

    This model is used directly with:
        llm.with_structured_output(JobScore)
    """

    # ── The 4 scoring dimensions ───────────────────────────────────────────

    technical_fit: DimensionScore = Field(
        description=(
            "How well the candidate's skills match the job's tech stack. "
            "Consider both required and nice-to-have technologies."
        ),
    )

    seniority_match: DimensionScore = Field(
        description=(
            "How well the candidate's years of experience match the role's level. "
            "Penalise under- and over-qualification equally."
        ),
    )

    remote_match: DimensionScore = Field(
        description=(
            "How well the job's remote policy matches the candidate's preference. "
            "Exact match = 25, adjacent (e.g. hybrid vs remote) = 12, opposite = 0."
        ),
    )

    sector_match: DimensionScore = Field(
        description=(
            "How well the company's sector aligns with the candidate's preferred sectors. "
            "Also consider company size, culture signals, and values if mentioned."
        ),
    )

    # ── Derived total ─────────────────────────────────────────────────────

    total: int = Field(
        ge=0,
        le=100,
        description="Sum of all four dimension scores. Computed automatically.",
    )

    # ── Human-readable summary ────────────────────────────────────────────

    explanation: str = Field(
        description=(
            "2–3 sentence overall summary of why this job is or is not a good match. "
            "Mention the strongest match signal and the biggest gap."
        ),
    )

    # ── Salary flag ───────────────────────────────────────────────────────

    salary_flag: bool = Field(
        default=False,
        description=(
            "True if the job's salary range appears below the candidate's minimum. "
            "False if salary matches, is not stated, or the candidate has no minimum."
        ),
    )

    # ── Validators ────────────────────────────────────────────────────────

    @model_validator(mode="before")
    @classmethod
    def compute_total(cls, values: dict) -> dict:
        """
        Compute total from the four dimension scores.
        If the LLM provides a total, it is overwritten with the real sum
        to prevent arithmetic mistakes from the model.
        """
        dims = ["technical_fit", "seniority_match", "remote_match", "sector_match"]
        try:
            values["total"] = sum(
                values[d]["score"] if isinstance(values[d], dict) else values[d].score
                for d in dims
                if d in values
            )
        except (KeyError, TypeError, AttributeError):
            pass   # let Pydantic surface the validation error naturally
        return values

    # ── Helpers ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        """One-liner for logging and display."""
        flag = " ⚠ salary below minimum" if self.salary_flag else ""
        return (
            f"Score {self.total}/100 | "
            f"tech={self.technical_fit.score} "
            f"seniority={self.seniority_match.score} "
            f"remote={self.remote_match.score} "
            f"sector={self.sector_match.score}"
            f"{flag}"
        )

    def to_dict(self) -> dict:
        """Flat dict for DataFrame / CSV export."""
        return {
            "total":               self.total,
            "technical_fit":       self.technical_fit.score,
            "seniority_match":     self.seniority_match.score,
            "remote_match":        self.remote_match.score,
            "sector_match":        self.sector_match.score,
            "salary_flag":         self.salary_flag,
            "explanation":         self.explanation,
            "rationale_technical": self.technical_fit.rationale,
            "rationale_seniority": self.seniority_match.rationale,
            "rationale_remote":    self.remote_match.rationale,
            "rationale_sector":    self.sector_match.rationale,
        }
