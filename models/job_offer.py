from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone
import uuid


@dataclass
class JobOffer:
    """Structured representation of a scraped job offer."""

    # --- Identity ---
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    platform: str = ""           # "wttj" | "indeed" | "linkedin"
    url: str = ""
    scraped_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # --- Core fields ---
    title: str = ""
    company: str = ""
    location: str = ""
    remote_policy: str = ""      # "on-site" | "hybrid" | "remote" | "unknown"

    # --- Details ---
    tech_stack: list[str] = field(default_factory=list)
    salary_range: str = ""       # e.g. "45k–60k EUR" or "" if not listed
    description_raw: str = ""

    # --- Extraction metadata ---
    extraction_success: bool = True
    extraction_notes: str = ""   # populated on partial / failed extractions

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "platform": self.platform,
            "url": self.url,
            "scraped_at": self.scraped_at,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "remote_policy": self.remote_policy,
            "tech_stack": self.tech_stack,
            "salary_range": self.salary_range,
            "description_raw": self.description_raw,
            "extraction_success": self.extraction_success,
            "extraction_notes": self.extraction_notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JobOffer":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
