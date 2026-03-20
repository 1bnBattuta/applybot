# ARCHITECTURE — The Smart Job Hunter

## 1. Project overview

**The Smart Job Hunter** is a multi-agent system that autonomously navigates job platforms (Welcome to the Jungle, Indeed, LinkedIn) to identify, extract, and rank job offers that best match a user's profile.

The user provides:
- A CV in PDF format
- Optional preferences: target location, remote work policy, desired salary range, preferred tech stack

The system returns a ranked list of job offers, each with a relevance score and a natural-language explanation of why it matches (or doesn't match) the profile.

---

## 2. High-level architecture

```
┌─────────────────────────────────────────────────────────┐
│                        User interface                    │
│              (Gradio / Streamlit — optional)             │
└────────────────────────┬────────────────────────────────┘
                         │  CV (PDF) + preferences
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Orchestrator agent                     │
│        Plans tasks, delegates to sub-agents,             │
│        manages state and stopping conditions             │
└──────┬──────────┬──────────────┬────────────────────────┘
       │          │              │
       ▼          ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────────┐
│ CV parser│ │ Browser  │ │   Scraper /  │
│  agent   │ │  agent   │ │   extractor  │
│          │ │          │ │    agent     │
└────┬─────┘ └────┬─────┘ └──────┬───────┘
     │            │              │
     ▼            ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────────┐
│pdfplumber│ │Playwright│ │  ChromaDB +  │
│+ Claude  │ │ browser  │ │  embeddings  │
│ (tools)  │ │ control  │ │  (RAG store) │
└──────────┘ └──────────┘ └──────┬───────┘
                                  │
                                  ▼
                         ┌──────────────┐
                         │ Scorer agent │
                         │ Claude API   │
                         │ (ranked list)│
                         └──────────────┘
```

---

## 3. Module breakdown

### 3.1 CV parser agent

**Role**: Extract structured information from the uploaded PDF.

**Input**: `cv.pdf`  
**Output**: a `UserProfile` JSON object

```json
{
  "job_titles": ["Software Engineer", "Backend Developer"],
  "skills": ["Python", "FastAPI", "Docker", "PostgreSQL"],
  "years_of_experience": 4,
  "education": "MSc Computer Science",
  "languages": ["French", "English"],
  "preferred_sectors": ["FinTech", "Health-tech"],
  "notes": "Looking for remote-friendly roles in France"
}
```

**Implementation**: uses `pdfplumber` to extract raw text, then calls Claude with a `parse_cv` tool (function calling, lab 04) to fill the schema. Falls back to a raw text extraction if the structured call fails.

---

### 3.2 Browser agent

**Role**: Navigate job platforms autonomously using Playwright.

Each navigation action is a tool the LLM can call:

| Tool | Description |
|---|---|
| `navigate(url)` | Open a URL in the browser |
| `click(selector)` | Click an element by CSS selector or text |
| `fill(selector, text)` | Fill a form field |
| `get_page_text()` | Return visible text of the current page |
| `get_page_html(selector)` | Return inner HTML of a section |
| `scroll_down()` | Scroll to load more results |
| `wait(ms)` | Pause (anti-bot courtesy delay) |

The agent runs in a `ReAct` loop (Reason → Act → Observe) until results are collected or `max_steps` is reached.

**Supported platforms**:

| Platform | Approach | Status |
|---|---|---|
| Welcome to the Jungle | Playwright (primary) |  Implemented |
| Indeed | Playwright |  Implemented |
| LinkedIn | Playwright (may require auth) | ⚠️Best-effort |
| Static mock site | Local HTML fixture |  Used for tests |

> **Note on LinkedIn**: LinkedIn aggressively blocks headless browsers. If scraping fails, the agent falls back to LinkedIn's public job search RSS feed. This is documented as a robustness finding in the evaluation section.

---

### 3.3 Scraper / extractor agent

**Role**: Parse raw HTML from job listing pages into structured `JobOffer` objects.

```json
{
  "id": "uuid",
  "title": "Backend Engineer",
  "company": "Acme Corp",
  "location": "Paris, France",
  "remote_policy": "hybrid",
  "tech_stack": ["Python", "Kubernetes", "GCP"],
  "salary_range": "45k–60k EUR",
  "description_raw": "...",
  "url": "https://...",
  "platform": "wttj",
  "scraped_at": "2025-04-01T10:00:00Z"
}
```

Instead of brittle CSS selectors, the extractor passes the raw page HTML to Claude and asks it to fill the schema. This makes extraction resilient to minor DOM changes — a key robustness property evaluated in the benchmark.

---

### 3.4 Scoring agent

**Role**: Score each `JobOffer` against the `UserProfile` on multiple dimensions.

**Scoring rubric** (each dimension: 0–25 pts):

| Dimension | Description |
|---|---|
| Technical fit | Overlap between required stack and candidate skills |
| Seniority match | Years of experience vs. role level |
| Remote policy | Candidate preference vs. offer policy |
| Sector / culture | Preferred sectors, company size, values keywords |

**Total**: 0–100 points. Score is computed by Claude with chain-of-thought reasoning, so the breakdown is always explainable.

---

## 4. Data flow

```
PDF upload
    │
    ▼
[CV Parser Agent] ──────────────► UserProfile (JSON)
                                        │
                                        ▼
                              [Orchestrator Agent]
                                        │
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                      [WTTJ]        [Indeed]     [LinkedIn]
                    Browser agent navigates and collects URLs
                          │             │             │
                          └──────┬──────┘             │
                                 ▼                    │
                      [Scraper Agent] ◄───────────────┘
                       Extracts JobOffer objects
                                 │
                                 ▼
                      [ChromaDB] ← embed(description)
                       Semantic pre-filter:
                       top-K offers closest to CV
                                 │
                                 ▼
                      [Scoring Agent]
                       Detailed LLM scoring
                       per dimension (0–100)
                                 │
                                 ▼
                      Ranked list of JobOffers
                       with score breakdowns
                                 │
                                 ▼
                         UI / JSON output
```

---

## 5. Agent design

### Orchestrator

The orchestrator is a stateful agent that maintains a task queue and a shared memory (a Python dict passed between tool calls). It follows the multi-agent pattern from lab 05:

```python
state = {
    "profile": UserProfile,
    "platform_queue": ["wttj", "indeed"],
    "raw_offers": [],
    "scored_offers": [],
    "step_count": 0,
    "max_steps": 50,
}
```

**Stopping conditions**:
- `max_steps` reached (hard limit — prevents infinite loops)
- `min_offers` collected and scored (soft limit — default: 20 offers)
- All platforms exhausted

### ReAct loop (browser agent)

```
THOUGHT: I need to search for Python backend roles in Paris.
ACTION: fill("#search-input", "Python backend Paris")
OBSERVATION: Page now shows 34 results.
THOUGHT: I should extract the first 10 listing URLs.
ACTION: get_page_html(".job-card")
OBSERVATION: [html of job cards]
...
```

Each loop iteration is logged with a timestamp, the action taken, and token count — feeding directly into the evaluation metrics.

---

## 6. Scoring system

### Embedding-based pre-filter

Before calling the expensive scoring LLM, all scraped job descriptions are embedded and stored in ChromaDB. The user profile is also embedded. A cosine similarity search retrieves the top-K most relevant offers, which are then passed to the scorer.

This reduces scoring calls from N (all scraped) to K (shortlist), keeping costs low.

### LLM scorer prompt structure

```
System: You are a recruitment specialist. Score the following job offer against the candidate profile.
        Respond ONLY in JSON using the provided schema. No preamble.

User:
  ## Candidate profile
  {profile_json}

  ## Job offer
  {offer_json}

  ## Scoring schema
  {score_schema}
```

The scorer is instructed to always include a `rationale` field per dimension, making scores auditable and comparable against human ratings.

---

## 7. Evaluation framework

Inspired by the **WebArena** benchmark methodology.

### Automated metrics (logged per run)

| Metric | Description |
|---|---|
| Task completion rate | % of runs that returned ≥ 1 relevant offer |
| Steps per task | Number of browser actions taken per platform |
| Offers per minute | Throughput metric |
| Scraping success rate | % of URLs where extraction succeeded |
| Pre-filter precision | % of top-K offers that pass human relevance check |

### Human vs. LLM evaluation (subset)

For 50 manually curated (profile, offer) pairs, a human annotator rates relevance 1–5. The LLM scorer's 0–100 rating is normalized and Spearman's ρ is computed between the two ratings.

This is the main academic contribution of the evaluation section.

### Robustness test

A controlled experiment where the DOM structure of the mock job site is modified (class names changed, layout restructured) to measure how many extra steps or failures the agent incurs — quantifying resilience to UI changes.

### Golden dataset format

```json
{
  "scenario_id": "s001",
  "profile": { ... },
  "platform": "wttj",
  "expected_top3_urls": ["https://...", "https://...", "https://..."],
  "human_scores": [92, 85, 70]
}
```

---

## 8. Tech stack

| Layer | Technology | Justification |
|---|---|---|
| LLM | Claude API (claude-sonnet-4) | Course requirement, function calling support |
| PDF extraction | `pdfplumber` | Accurate text + layout extraction |
| Browser automation | `playwright` (async) | Modern, stable, supports anti-bot delays |
| Vector store | `chromadb` | Lightweight, local, no infra needed |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Fast, free, good for English/French |
| Agent framework | Custom (lab 05 pattern) + optional `langchain` | Direct control, mirrors course labs |
| UI (optional) | `gradio` | Quick to build, PDF upload native |
| Evaluation | `pandas` + `scipy` (Spearman ρ) | Standard statistical tooling |
| Environment | `devcontainer` (from repo) | Reproducible for all team members |

---

## 9. Repository structure

```
smart-job-hunter/
│
├── ARCHITECTURE.md          ← this file
├── README.md
├── requirements.txt
│
├── agents/
│   ├── orchestrator.py      ← main agent loop
│   ├── cv_parser.py         ← CV → UserProfile
│   ├── browser_agent.py     ← ReAct loop + Playwright tools
│   ├── scraper_agent.py     ← HTML → JobOffer
│   └── scorer_agent.py      ← JobOffer × UserProfile → Score
│
├── platforms/
│   ├── base.py              ← PlatformDriver interface
│   ├── wttj.py              ← Welcome to the Jungle driver
│   ├── indeed.py            ← Indeed driver
│   └── linkedin.py          ← LinkedIn driver (best-effort)
│
├── tools/
│   ├── browser_tools.py     ← Playwright tool wrappers
│   ├── pdf_tools.py         ← pdfplumber helpers
│   └── embedding_tools.py   ← ChromaDB + sentence-transformers
│
├── models/
│   ├── user_profile.py      ← UserProfile dataclass
│   ├── job_offer.py         ← JobOffer dataclass
│   └── score.py             ← Score dataclass
│
├── evaluation/
│   ├── golden_dataset/      ← hand-curated test scenarios (JSON)
│   ├── run_benchmark.py     ← automated benchmark runner
│   └── human_eval.py        ← human vs LLM score comparison
│
├── mock_site/               ← static HTML job board for local tests
│   └── index.html
│
├── ui/
│   └── app.py               ← Gradio interface
│
├── notebooks/
│   ├── 01_cv_parsing.ipynb
│   ├── 02_browser_agent.ipynb
│   ├── 03_scoring.ipynb
│   └── 04_evaluation.ipynb
│
└── tests/
    ├── test_cv_parser.py
    ├── test_scraper.py
    └── test_scorer.py
```

---

## 10. Environment & configuration

### Setup

```bash
# Clone the course repo and create your project branch
git clone https://github.com/massimotisi/applied-llms-labs-2025-2026
cd applied-llms-labs-2025-2026
git checkout -b project/smart-job-hunter

# Install dependencies
pip install -r requirements.txt
playwright install chromium
```

### Environment variables

Create a `.env` file at the root (never commit it):

```env
ANTHROPIC_API_KEY=sk-ant-...
CHROMA_PERSIST_DIR=./chroma_data
MAX_STEPS=50
MIN_OFFERS=20
LOG_LEVEL=INFO
```

### Running the agent

```bash
# Full pipeline from CLI
python -m agents.orchestrator --cv path/to/cv.pdf --platforms wttj,indeed

# Run the Gradio UI
python ui/app.py

# Run the evaluation benchmark
python evaluation/run_benchmark.py --dataset evaluation/golden_dataset/
```

*Last updated: March 2026*
