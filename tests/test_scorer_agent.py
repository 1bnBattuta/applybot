"""
tests/test_scorer_agent.py

Tests for agents/scorer_agent.py.
The LLM is fully mocked — no API key, no network calls.

Coverage:
  - score_offer()  : single offer scoring, salary flag, failed offers
  - score_offers() : batch scoring, ranking, skip_failed behaviour
  - _normalise_remote()  : all platform-specific aliases
  - _check_salary_flag() : salary parsing edge cases
  - _offer_to_prompt()   : correct field rendering
  - _profile_to_prompt() : correct field rendering

Run:
    pytest tests/test_scorer_agent.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from models.job_offer import JobOffer
from models.user_profile import UserProfile
from models.score import JobScore, DimensionScore
from agents.scorer_agent import (
    ScoredOffer,
    score_offer,
    score_offers,
    _normalise_remote,
    _check_salary_flag,
    _offer_to_prompt,
    _profile_to_prompt,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def profile() -> UserProfile:
    return UserProfile(
        job_titles=["Backend Engineer", "Python Developer"],
        skills=["Python", "FastAPI", "Docker", "PostgreSQL"],
        years_of_experience=4,
        seniority="mid",
        education="MSc Computer Science",
        languages=["French", "English"],
        preferred_sectors=["FinTech", "SaaS"],
        remote_preference="remote",
        preferred_locations=["Paris", "Remote"],
        salary_min=50000,
        notes="Looking for remote-first teams.",
    )


@pytest.fixture
def offer() -> JobOffer:
    return JobOffer(
        platform="wttj",
        url="https://example.com/job/1",
        title="Senior Python Backend Engineer",
        company="Payflow",
        location="Full Remote (France)",
        remote_policy="remote",
        tech_stack=["Python", "FastAPI", "PostgreSQL", "Docker", "GCP"],
        salary_range="55k-70k EUR",
        description_raw="Payflow is a FinTech startup building payment infrastructure.",
        extraction_success=True,
    )


@pytest.fixture
def good_score() -> JobScore:
    return JobScore(
        technical_fit=DimensionScore(score=22, rationale="Python/FastAPI match"),
        seniority_match=DimensionScore(score=18, rationale="4 yrs vs 5 required"),
        remote_match=DimensionScore(score=25, rationale="Both fully remote"),
        sector_match=DimensionScore(score=20, rationale="FinTech preferred"),
        total=0,
        explanation="Strong match overall.",
    )


@pytest.fixture
def weak_score() -> JobScore:
    return JobScore(
        technical_fit=DimensionScore(score=5, rationale="No stack overlap"),
        seniority_match=DimensionScore(score=8, rationale="Overqualified"),
        remote_match=DimensionScore(score=0, rationale="Onsite vs remote preference"),
        sector_match=DimensionScore(score=5, rationale="Wrong sector"),
        total=0,
        explanation="Poor match.",
    )


@pytest.fixture
def mock_llm(good_score):
    """LLM mock whose .with_structured_output().invoke() returns good_score."""
    chain = MagicMock()
    chain.invoke.return_value = good_score

    llm = MagicMock()
    # with_structured_output returns a chain: prompt | llm.with_structured_output(...)
    # _build_chain builds: prompt | llm.with_structured_output(JobScore)
    # We patch at the chain level via _build_chain
    return llm


# ─────────────────────────────────────────────────────────────────────────────
# _normalise_remote
# ─────────────────────────────────────────────────────────────────────────────

class TestNormaliseRemote:

    def test_on_site_with_hyphen(self):
        assert _normalise_remote("on-site") == "onsite"

    def test_on_site_with_space(self):
        assert _normalise_remote("on site") == "onsite"

    def test_french_presentiel(self):
        assert _normalise_remote("présentiel") == "onsite"

    def test_full_remote(self):
        assert _normalise_remote("full remote") == "remote"

    def test_fully_remote(self):
        assert _normalise_remote("fully remote") == "remote"

    def test_teletravail_accented(self):
        assert _normalise_remote("télétravail") == "remote"

    def test_teletravail_unaccented(self):
        assert _normalise_remote("teletravail") == "remote"

    def test_french_hybride(self):
        assert _normalise_remote("hybride") == "hybrid"

    def test_remote_passthrough(self):
        assert _normalise_remote("remote") == "remote"

    def test_hybrid_passthrough(self):
        assert _normalise_remote("hybrid") == "hybrid"

    def test_onsite_passthrough(self):
        assert _normalise_remote("onsite") == "onsite"

    def test_empty_string_returns_unknown(self):
        assert _normalise_remote("") == "unknown"

    def test_whitespace_stripped(self):
        assert _normalise_remote("  remote  ") == "remote"

    def test_unknown_value_returned_as_is(self):
        assert _normalise_remote("office") == "office"


# ─────────────────────────────────────────────────────────────────────────────
# _check_salary_flag
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckSalaryFlag:

    def test_offer_above_minimum_no_flag(self, profile, offer):
        # offer max = 70k, profile min = 50k → no flag
        assert _check_salary_flag(profile, offer) is False

    def test_offer_below_minimum_flag(self, profile):
        low = JobOffer(
            title="Dev", company="X", location="Paris",
            salary_range="30k-45k EUR",
        )
        assert _check_salary_flag(profile, low) is True

    def test_no_salary_on_offer_no_flag(self, profile):
        no_salary = JobOffer(title="Dev", company="X", location="Paris", salary_range="")
        assert _check_salary_flag(profile, no_salary) is False

    def test_no_minimum_on_profile_no_flag(self, offer):
        profile_no_min = UserProfile(
            job_titles=["Dev"], skills=["Python"],
            years_of_experience=3, seniority="mid",
            education="BSc", languages=["French"],
            preferred_sectors=[], remote_preference="remote",
            preferred_locations=[], salary_min=None, notes=""
        )
        assert _check_salary_flag(profile_no_min, offer) is False

    def test_salary_with_k_suffix_parsed(self, profile):
        offer_k = JobOffer(
            title="Dev", company="X", location="Paris",
            salary_range="40k-55k EUR",
        )
        # max = 55k, min = 50k → no flag
        assert _check_salary_flag(profile, offer_k) is False

    def test_salary_exact_at_minimum_no_flag(self, profile):
        # max = 50k = profile min → no flag (equal is acceptable)
        offer_exact = JobOffer(
            title="Dev", company="X", location="Paris",
            salary_range="40k-50k EUR",
        )
        assert _check_salary_flag(profile, offer_exact) is False

    def test_salary_one_below_minimum_flag(self, profile):
        # max = 49k < profile min 50k → flag
        offer_one_below = JobOffer(
            title="Dev", company="X", location="Paris",
            salary_range="35k-49k EUR",
        )
        assert _check_salary_flag(profile, offer_one_below) is True


# ─────────────────────────────────────────────────────────────────────────────
# _offer_to_prompt and _profile_to_prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptFormatting:

    def test_offer_prompt_contains_title(self, offer):
        text = _offer_to_prompt(offer)
        assert "Senior Python Backend Engineer" in text

    def test_offer_prompt_contains_tech_stack(self, offer):
        text = _offer_to_prompt(offer)
        assert "Python" in text
        assert "FastAPI" in text

    def test_offer_prompt_normalises_remote(self):
        o = JobOffer(
            title="Dev", company="X", location="Paris",
            remote_policy="on-site",
        )
        assert "onsite" in _offer_to_prompt(o)

    def test_offer_prompt_truncates_description(self):
        long_desc = "A" * 3000
        o = JobOffer(
            title="Dev", company="X", location="Paris",
            description_raw=long_desc,
        )
        text = _offer_to_prompt(o)
        # Description capped at 1500 chars
        assert text.count("A") <= 1500

    def test_offer_prompt_skips_empty_salary(self):
        o = JobOffer(title="Dev", company="X", location="Paris", salary_range="")
        assert "Salary" not in _offer_to_prompt(o)

    def test_profile_prompt_contains_skills(self, profile):
        text = _profile_to_prompt(profile)
        assert "Python" in text
        assert "FastAPI" in text

    def test_profile_prompt_contains_remote_preference(self, profile):
        text = _profile_to_prompt(profile)
        assert "remote" in text

    def test_profile_prompt_contains_salary_min(self, profile):
        text = _profile_to_prompt(profile)
        assert "50,000" in text or "50000" in text

    def test_profile_prompt_skips_salary_when_none(self):
        p = UserProfile(
            job_titles=["Dev"], skills=["Python"],
            years_of_experience=3, seniority="mid",
            education="BSc", languages=["French"],
            preferred_sectors=[], remote_preference="remote",
            preferred_locations=[], salary_min=None,
        )
        assert "salary" not in _profile_to_prompt(p).lower()

    def test_profile_prompt_skips_notes_when_empty(self, profile):
        p = profile.model_copy(update={"notes": ""})
        assert "Notes" not in _profile_to_prompt(p)


# ─────────────────────────────────────────────────────────────────────────────
# score_offer — single offer
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreOffer:

    def test_returns_scored_offer(self, profile, offer, good_score):
        with patch("agents.scorer_agent._build_chain") as mock_build:
            chain = MagicMock()
            chain.invoke.return_value = good_score
            mock_build.return_value = chain

            result = score_offer(profile, offer, llm=MagicMock())

        assert isinstance(result, ScoredOffer)
        assert result.offer is offer
        assert result.score.total == 85

    def test_salary_flag_set_when_offer_below_minimum(self, profile, good_score):
        low_offer = JobOffer(
            title="Dev", company="X", location="Paris",
            remote_policy="remote", tech_stack=["Python"],
            salary_range="30k-45k EUR", extraction_success=True,
        )
        with patch("agents.scorer_agent._build_chain") as mock_build:
            chain = MagicMock()
            chain.invoke.return_value = good_score
            mock_build.return_value = chain

            result = score_offer(profile, low_offer, llm=MagicMock())

        assert result.score.salary_flag is True

    def test_salary_flag_false_when_offer_above_minimum(self, profile, offer, good_score):
        with patch("agents.scorer_agent._build_chain") as mock_build:
            chain = MagicMock()
            chain.invoke.return_value = good_score
            mock_build.return_value = chain

            result = score_offer(profile, offer, llm=MagicMock())

        assert result.score.salary_flag is False

    def test_returns_none_for_failed_extraction(self, profile):
        failed = JobOffer(
            title="Dev", company="X", location="Paris",
            extraction_success=False, extraction_notes="timeout",
        )
        result = score_offer(profile, failed, llm=MagicMock())
        assert result is None

    def test_returns_none_for_empty_title(self, profile):
        empty = JobOffer(
            title="", company="X", location="Paris",
            extraction_success=True,
        )
        result = score_offer(profile, empty, llm=MagicMock())
        assert result is None

    def test_returns_none_when_llm_raises(self, profile, offer):
        with patch("agents.scorer_agent._build_chain") as mock_build:
            chain = MagicMock()
            chain.invoke.side_effect = Exception("API error")
            mock_build.return_value = chain

            result = score_offer(profile, offer, llm=MagicMock())

        assert result is None

    def test_llm_called_with_profile_and_offer_strings(self, profile, offer, good_score):
        with patch("agents.scorer_agent._build_chain") as mock_build:
            chain = MagicMock()
            chain.invoke.return_value = good_score
            mock_build.return_value = chain

            score_offer(profile, offer, llm=MagicMock())

            call_kwargs = chain.invoke.call_args[0][0]
            assert "profile" in call_kwargs
            assert "offer" in call_kwargs
            assert "Python" in call_kwargs["profile"]
            assert "Payflow" in call_kwargs["offer"]


# ─────────────────────────────────────────────────────────────────────────────
# score_offers — batch
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreOffers:

    def _make_offers(self, n: int) -> list[JobOffer]:
        return [
            JobOffer(
                title=f"Job {i}",
                company=f"Company {i}",
                location="Paris",
                remote_policy="remote",
                tech_stack=["Python"],
                extraction_success=True,
            )
            for i in range(n)
        ]

    def _make_score(self, total_tech: int) -> JobScore:
        """Build a JobScore where technical_fit drives the total."""
        per = total_tech // 4
        return JobScore(
            technical_fit=DimensionScore(score=total_tech, rationale="x"),
            seniority_match=DimensionScore(score=per, rationale="x"),
            remote_match=DimensionScore(score=per, rationale="x"),
            sector_match=DimensionScore(score=per, rationale="x"),
            total=0,
            explanation="test",
        )

    def test_returns_all_valid_offers(self, profile):
        offers = self._make_offers(3)
        score = self._make_score(20)

        with patch("agents.scorer_agent._build_chain") as mock_build:
            chain = MagicMock()
            chain.invoke.return_value = score
            mock_build.return_value = chain

            results = score_offers(profile, offers, llm=MagicMock())

        assert len(results) == 3

    def test_results_sorted_best_first(self, profile):
        offers = self._make_offers(3)
        scores = [
            self._make_score(10),  # weakest
            self._make_score(22),  # strongest
            self._make_score(15),  # middle
        ]

        with patch("agents.scorer_agent._build_chain") as mock_build:
            chain = MagicMock()
            chain.invoke.side_effect = scores
            mock_build.return_value = chain

            results = score_offers(profile, offers, llm=MagicMock())

        totals = [r.score.total for r in results]
        assert totals == sorted(totals, reverse=True)

    def test_failed_extractions_skipped_by_default(self, profile, good_score):
        good = JobOffer(title="Good", company="X", location="Paris",
                        extraction_success=True)
        bad  = JobOffer(title="Bad",  company="Y", location="Paris",
                        extraction_success=False)

        with patch("agents.scorer_agent._build_chain") as mock_build:
            chain = MagicMock()
            chain.invoke.return_value = good_score
            mock_build.return_value = chain

            results = score_offers(profile, [good, bad], llm=MagicMock())

        assert len(results) == 1
        assert results[0].offer.title == "Good"

    def test_llm_error_skipped_when_skip_failed_true(self, profile):
        offers = self._make_offers(2)

        with patch("agents.scorer_agent._build_chain") as mock_build:
            chain = MagicMock()
            # First call fails, second succeeds
            chain.invoke.side_effect = [
                Exception("API timeout"),
                self._make_score(18),
            ]
            mock_build.return_value = chain

            results = score_offers(profile, offers, llm=MagicMock(), skip_failed=True)

        assert len(results) == 1

    def test_llm_error_raises_when_skip_failed_false(self, profile):
        offers = self._make_offers(1)

        with patch("agents.scorer_agent._build_chain") as mock_build:
            chain = MagicMock()
            chain.invoke.side_effect = Exception("API error")
            mock_build.return_value = chain

            with pytest.raises(RuntimeError, match="Scoring failed"):
                score_offers(profile, offers, llm=MagicMock(), skip_failed=False)

    def test_empty_offers_list_returns_empty(self, profile):
        results = score_offers(profile, [], llm=MagicMock())
        assert results == []

    def test_scored_offer_preserves_original_offer(self, profile, good_score):
        offer = JobOffer(
            title="Python Dev", company="Acme", location="Remote",
            remote_policy="remote", tech_stack=["Python"],
            extraction_success=True, url="https://acme.com/job/1",
        )
        with patch("agents.scorer_agent._build_chain") as mock_build:
            chain = MagicMock()
            chain.invoke.return_value = good_score
            mock_build.return_value = chain

            results = score_offers(profile, [offer], llm=MagicMock())

        assert results[0].offer.url == "https://acme.com/job/1"
        assert results[0].offer.company == "Acme"


# ─────────────────────────────────────────────────────────────────────────────
# ScoredOffer sorting
# ─────────────────────────────────────────────────────────────────────────────

class TestScoredOfferSorting:

    def _make_scored(self, tech_score: int) -> ScoredOffer:
        offer = JobOffer(title=f"Job {tech_score}", company="X", location="Paris")
        score = JobScore(
            technical_fit=DimensionScore(score=tech_score, rationale="x"),
            seniority_match=DimensionScore(score=5, rationale="x"),
            remote_match=DimensionScore(score=5, rationale="x"),
            sector_match=DimensionScore(score=5, rationale="x"),
            total=0,
            explanation="test",
        )
        return ScoredOffer(offer=offer, score=score)

    def test_sorted_descending(self):
        items = [self._make_scored(10), self._make_scored(22), self._make_scored(5)]
        ranked = sorted(items, reverse=True)
        assert ranked[0].score.technical_fit.score == 22
        assert ranked[-1].score.technical_fit.score == 5

    def test_equal_scores_do_not_crash(self):
        items = [self._make_scored(15), self._make_scored(15)]
        ranked = sorted(items, reverse=True)
        assert len(ranked) == 2
