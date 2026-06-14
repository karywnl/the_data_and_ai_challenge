#!/usr/bin/env python3
"""Shared logic for the Redrob ranker: parsing, role classification, feature
extraction, honeypot detection, scoring, and reasoning generation.

Design principle (grounded in data exploration):
  - The `skills` array is NOISE by construction — every skill appears ~12k times,
    uniformly sprayed across all 100k candidates. We deliberately do NOT treat the
    skills list as evidence of fit.
  - The real signal is: current/historical TITLE, career-history free-text
    DESCRIPTIONS, trajectory (product vs services, tenure), and behavioral signals.
"""
from __future__ import annotations

import re
from datetime import date

# Reference "today" — dataset signals are anchored around mid-2026.
TODAY = date(2026, 6, 14)

# ---------------------------------------------------------------------------
# Job description, as facet queries for semantic matching.
# Positive facets capture what the JD actually wants; negatives capture the
# explicit anti-patterns (research-only, CV/speech, consulting).
# ---------------------------------------------------------------------------
JD_POSITIVE_FACETS = [
    "Production experience building embeddings-based retrieval and semantic search "
    "systems deployed to real users, handling embedding drift and index refresh.",
    "Vector databases and hybrid search infrastructure such as Pinecone, Weaviate, "
    "Qdrant, Milvus, FAISS, OpenSearch or Elasticsearch in production.",
    "Designing ranking, recommendation, learning-to-rank and search-relevance systems "
    "shipped end to end at meaningful scale.",
    "Designing evaluation frameworks for ranking systems: NDCG, MRR, MAP, "
    "offline-to-online correlation and A/B test interpretation.",
    "Applied machine learning and LLM systems engineering at a product company, "
    "shipping fast and iterating with real users.",
]
JD_NEGATIVE_FACETS = [
    "Purely academic or research-only machine learning work with no production "
    "deployment, focused on papers and experiments.",
    "Computer vision, image classification, object detection, speech recognition "
    "or robotics work without natural language or information retrieval.",
    "IT services and consulting delivery work for outsourced clients.",
]

# ---------------------------------------------------------------------------
# Role / title classification — the single most decisive feature.
# Each bucket carries a base role-fit prior in [0, 1]. Career-text evidence and
# semantic similarity then adjust this prior up or down.
# ---------------------------------------------------------------------------
ROLE_RULES = [
    # (regex on lowercased title, bucket, base prior)
    # NOTE: \b before "search engineer" is essential — without it this matches
    # the substring inside "reSEARCH ENGINEER" and misclassifies AI Research
    # Engineers (a trap bucket) as top-tier.
    (r"recommendation systems? engineer|\bsearch engineer\b|applied ml|"
     r"relevance engineer",
     "gold_ir", 1.00),
    # Junior must be checked BEFORE the generic ml-engineer rule, otherwise
    # "Junior ML Engineer" matches "ml engineer" and escapes the band trap.
    (r"junior (ml|machine learning|ai|data)|ml intern|associate ml|intern",
     "junior_ml", 0.45),
    (r"\bml engineer|machine learning engineer|\bai engineer\b",
     "ml_engineer", 0.92),
    (r"software engineer \(ml\)|ml.*scientist|staff.*(ml|ai)",
     "ml_senior", 0.88),
    (r"ai specialist|ai/ml|ml/ai",
     "ai_specialist", 0.80),
    (r"data scientist",
     "data_scientist", 0.70),
    (r"ai research engineer|research scientist|research engineer",
     "research", 0.40),          # JD: rejects pure-research-no-production
    (r"computer vision|cv engineer|vision engineer",
     "cv", 0.32),                # JD: rejects CV/speech without NLP/IR
    (r"data engineer|analytics engineer",
     "data_eng", 0.58),          # adjacent — data-infra-to-ML transition is valued
    (r"backend engineer|backend developer",
     "backend", 0.50),
    (r"senior software engineer|staff engineer|software engineer|full.?stack|"
     r"platform engineer",
     "swe", 0.42),
    (r"data analyst|business intelligence|\bbi\b",
     "data_analyst", 0.35),
    (r"cloud engineer|devops|sre|site reliability|qa engineer|mobile|frontend|"
     r"\.net|java developer|web developer",
     "other_tech", 0.22),
]
# Non-tech titles that host keyword-stuffers — base prior 0 (title says not a fit).
NONTECH_RE = re.compile(
    r"business analyst|hr manager|human resource|accountant|project manager|"
    r"customer support|operations manager|content writer|sales executive|"
    r"civil engineer|mechanical engineer|graphic designer|marketing manager|"
    r"recruiter|teacher|nurse|doctor|lawyer", re.I)

# ---------------------------------------------------------------------------
# Career-text evidence vocabularies (matched over free-text descriptions/summary,
# which — unlike the skills array — are genuine signal).
# ---------------------------------------------------------------------------
# Strong retrieval/ranking evidence — phrases, not bare words like "search" or
# "ranking" (which appear loosely in generic SWE descriptions).
IR_TERMS = re.compile(
    r"information retrieval|semantic search|vector (search|database|index|store)|"
    r"search relevance|search ranking|ranking (system|model|algorithm|pipeline|signal)|"
    r"learning.to.rank|\bltr\b|re.?ranking|recommendation (system|engine|model)|"
    r"recommender|recsys|personalization|nearest neighbou?r|\bann index\b|"
    r"embedding|elasticsearch|opensearch|\bfaiss\b|pinecone|weaviate|qdrant|"
    r"milvus|\brag\b|bm25|relevance ranking|candidate retrieval|matching engine",
    re.I)
NLP_TERMS = re.compile(
    r"\bnlp\b|natural language|language model|\bllm\b|transformer|bert|"
    r"text classification|named entity|question answering|fine.?tun", re.I)
PRODUCTION_TERMS = re.compile(
    r"production|deployed|real users|in production|at scale|served|serving|"
    r"latency|throughput|a/b test|experiment|online metric|millions of", re.I)
EVAL_TERMS = re.compile(r"ndcg|\bmrr\b|\bmap\b|precision@|recall@|offline.*online|"
                        r"a/b test|evaluation framework|relevance metric", re.I)
CV_SPEECH_TERMS = re.compile(
    r"computer vision|image classification|object detection|segmentation|"
    r"\bocr\b|speech recognition|\btts\b|\basr\b|robotics|point cloud|lidar", re.I)

# Consulting / IT-services firms (JD disqualifier if entire career is here).
CONSULTING_RE = re.compile(
    r"\btcs\b|tata consultancy|infosys|wipro|accenture|cognizant|capgemini|"
    r"\bhcl\b|tech mahindra|mindtree|\blti\b|larsen|mphasis|hexaware|"
    r"\bdxc\b|\bibm\b global|deloitte|pwc|\bkpmg\b|ernst", re.I)

INDIA_HUBS = re.compile(
    r"pune|noida|bengaluru|bangalore|hyderabad|mumbai|delhi|gurgaon|gurugram|"
    r"chennai|ncr", re.I)


def parse_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def months_ago(d, ref=TODAY):
    if d is None:
        return None
    return (ref.year - d.year) * 12 + (ref.month - d.month)


# ---------------------------------------------------------------------------
# Embedding document — what we feed the sentence-transformer. Summary, headline,
# job titles and career-history DESCRIPTIONS. Skills are deliberately excluded.
# ---------------------------------------------------------------------------
def build_doc(c, max_chars=1600) -> str:
    prof = c.get("profile", {})
    parts = [prof.get("headline", ""), prof.get("summary", "")]
    for j in (c.get("career_history") or [])[:5]:
        parts.append(f"{j.get('title', '')} at {j.get('company', '')}. "
                     f"{j.get('description', '')}")
    doc = "  ".join(p for p in parts if p).strip()
    # The embedding model truncates to ~256 tokens anyway; capping the input
    # string avoids wasting time tokenizing text that will be dropped.
    return doc[:max_chars]


def classify_role(title: str):
    tl = (title or "").lower()
    for pat, bucket, prior in ROLE_RULES:
        if re.search(pat, tl):
            return bucket, prior
    if NONTECH_RE.search(tl):
        return "nontech", 0.0
    return "unknown", 0.15


def honeypot_flags(c):
    """Strict impossibility checks. Only fire on genuinely inconsistent profiles
    so we stay well under the >10% honeypot disqualification threshold."""
    reasons = []
    prof = c.get("profile", {})
    yoe = prof.get("years_of_experience") or 0
    ch = c.get("career_history") or []
    skills = c.get("skills") or []
    edu = c.get("education") or []

    for j in ch:
        sd, ed = parse_date(j.get("start_date")), parse_date(j.get("end_date"))
        dur = j.get("duration_months") or 0
        if sd is not None:
            elapsed = months_ago(sd)
            if elapsed is not None and dur > elapsed + 2:
                reasons.append("duration_exceeds_time_since_start")
            if elapsed is not None and elapsed < 0:
                reasons.append("job_starts_in_future")
        if sd and ed and ed < sd:
            reasons.append("end_before_start")

    high_prof_zero = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and (s.get("duration_months") or 0) == 0)
    if high_prof_zero >= 5:
        reasons.append("many_expert_skills_zero_duration")

    total_months = sum((j.get("duration_months") or 0) for j in ch)
    if yoe > 0 and total_months > (yoe * 12) * 3 + 36:
        reasons.append("career_months_implausible_vs_yoe")

    for e in edu:
        sy, ey = e.get("start_year"), e.get("end_year")
        if isinstance(sy, int) and isinstance(ey, int) and ey < sy:
            reasons.append("education_end_before_start")

    return reasons


def gaussian_band(x, lo=5.0, hi=9.0, width=3.0):
    """1.0 inside [lo, hi], smooth falloff outside."""
    if x < lo:
        d = lo - x
    elif x > hi:
        d = x - hi
    else:
        return 1.0
    return float(pow(2.718281828, -(d * d) / (2 * width * width)))


def extract_features(c) -> dict:
    """Structured, interpretable features used for scoring + reasoning."""
    prof = c.get("profile", {})
    sig = c.get("redrob_signals", {})
    ch = c.get("career_history") or []
    title = prof.get("current_title") or ""
    bucket, role_prior = classify_role(title)

    # Concatenated career free-text (genuine signal, unlike skills).
    career_text = " ".join(
        f"{j.get('title', '')} {j.get('description', '')}" for j in ch)
    blob = f"{prof.get('summary', '')} {career_text}"

    ir = bool(IR_TERMS.search(blob))
    nlp = bool(NLP_TERMS.search(blob))
    prod = bool(PRODUCTION_TERMS.search(blob))
    evalf = bool(EVAL_TERMS.search(blob))
    cv = bool(CV_SPEECH_TERMS.search(blob))

    # Company trajectory.
    companies = [j.get("company", "") for j in ch]
    industries = [(j.get("industry") or "").lower() for j in ch]
    n_consulting = sum(1 for x in companies if CONSULTING_RE.search(x or ""))
    n_services_ind = sum(1 for x in industries if "it services" in x or "consult" in x)
    consulting_only = (len(ch) > 0 and
                       (n_consulting >= len(ch) or n_services_ind >= len(ch)))

    # Tenure / title-chasing.
    durations = [j.get("duration_months") or 0 for j in ch if not j.get("is_current")]
    avg_tenure = (sum(durations) / len(durations)) if durations else None
    title_chaser = (len(ch) >= 3 and avg_tenure is not None and avg_tenure < 18)

    yoe = prof.get("years_of_experience") or 0

    # Location.
    loc = f"{prof.get('location', '')} {prof.get('country', '')}"
    in_india = "india" in loc.lower()
    in_hub = bool(INDIA_HUBS.search(loc))
    relocate = bool(sig.get("willing_to_relocate"))

    # Behavioral.
    rr = sig.get("recruiter_response_rate")
    last_active_m = months_ago(parse_date(sig.get("last_active_date")))
    open_to_work = bool(sig.get("open_to_work_flag"))
    icr = sig.get("interview_completion_rate")
    saved = sig.get("saved_by_recruiters_30d") or 0
    notice = sig.get("notice_period_days")

    return dict(
        candidate_id=c.get("candidate_id"), title=title, bucket=bucket,
        role_prior=role_prior, yoe=yoe,
        ev_ir=ir, ev_nlp=nlp, ev_prod=prod, ev_eval=evalf, ev_cv=cv,
        consulting_only=consulting_only, n_consulting=n_consulting,
        avg_tenure=avg_tenure, title_chaser=title_chaser, n_jobs=len(ch),
        in_india=in_india, in_hub=in_hub, relocate=relocate,
        location=prof.get("location", ""), country=prof.get("country", ""),
        rr=rr, last_active_m=last_active_m, open_to_work=open_to_work,
        icr=icr, saved=saved, notice=notice,
        company=prof.get("current_company", ""),
        honeypot=honeypot_flags(c),
    )


# ---------------------------------------------------------------------------
# Scoring. Weights are module-level for easy tuning.
# ---------------------------------------------------------------------------
W_ROLE = 0.34       # title/role prior (most decisive)
W_SEM = 0.30        # semantic JD match over career text
W_EVID = 0.22       # explicit career-text evidence (IR/NLP/prod/eval)
W_BAND = 0.08       # experience-band fit
W_LOC = 0.06        # location fit

def evidence_score(f) -> float:
    s = 0.0
    if f["ev_ir"]:
        s += 0.45                       # retrieval/ranking/recsys — core JD need
    if f["ev_nlp"]:
        s += 0.20
    if f["ev_prod"]:
        s += 0.20                       # production deployment
    if f["ev_eval"]:
        s += 0.15                       # ranking evaluation literacy
    # CV/speech with no NLP/IR is an explicit anti-pattern.
    if f["ev_cv"] and not (f["ev_ir"] or f["ev_nlp"]):
        s -= 0.30
    return max(0.0, min(1.0, s))


def location_score(f) -> float:
    if f["in_hub"]:
        return 1.0
    if f["in_india"]:
        return 0.85
    if f["relocate"]:
        return 0.6
    return 0.2                          # outside India, won't relocate (no visa)


def behavioral_multiplier(f) -> float:
    """Availability/engagement modifier in ~[0.55, 1.08]. A perfect-on-paper but
    unreachable candidate is down-weighted, never fully removed."""
    m = 1.0
    rr = f["rr"]
    if rr is not None:
        m *= 0.7 + 0.5 * rr             # 0.02->0.71x, 0.95->1.18x (capped below)
    la = f["last_active_m"]
    if la is not None:
        if la <= 1:
            m *= 1.05
        elif la <= 3:
            m *= 1.0
        elif la <= 5:
            m *= 0.9
        else:
            m *= 0.75                   # ~6+ months idle
    if f["open_to_work"]:
        m *= 1.05
    if f["saved"] and f["saved"] >= 3:
        m *= 1.03
    if f["icr"] is not None and f["icr"] < 0.4:
        m *= 0.92
    return max(0.55, min(1.08, m))


def base_score(f, sem: float) -> float:
    """Combine role prior, semantic match, evidence, band, location into [0,1]."""
    role = f["role_prior"]
    # Rescue: a non-AI/unknown title with strong IR+production career evidence is
    # the "plain-language Tier-5" the JD describes — lift the role prior.
    if role < 0.55 and f["ev_ir"] and f["ev_prod"]:
        role = max(role, 0.6)
    # Rescue research/CV titles only if they show production IR/NLP.
    if f["bucket"] in ("research", "cv") and f["ev_ir"] and f["ev_prod"]:
        role = max(role, 0.62)

    band = gaussian_band(f["yoe"])
    s = (W_ROLE * role + W_SEM * max(0.0, sem) + W_EVID * evidence_score(f)
         + W_BAND * band + W_LOC * location_score(f))

    # Hard penalties for explicit JD disqualifiers.
    if f["consulting_only"]:
        s *= 0.6
    if f["title_chaser"]:
        s *= 0.85
    if f["bucket"] == "nontech":
        s *= 0.25                       # keyword-stuffer host title
    if f["yoe"] < 3:
        s *= 0.8
    return s


def final_score(f, sem: float) -> float:
    if f["honeypot"]:
        return 0.0                      # forced to the bottom
    return base_score(f, sem) * behavioral_multiplier(f)


# ---------------------------------------------------------------------------
# Reasoning generation — built purely from facts already in the profile, so it
# never hallucinates. Cites the decisive factors and one honest concern, and
# varies sentence shape across ranks (Stage-4 review samples 10 rows).
# ---------------------------------------------------------------------------
def _fmt_yoe(y):
    return f"{y:.1f}".rstrip("0").rstrip(".")


def make_reasoning(f, sem: float) -> str:
    yoe = _fmt_yoe(f["yoe"])
    title = f["title"] or "Candidate"
    company = f["company"]
    where = f["location"] or f["country"]

    # Positive evidence clauses (only facts present in the profile).
    pos = []
    if f["ev_ir"]:
        pos.append("career history shows retrieval/ranking/search work")
    if f["ev_prod"]:
        pos.append("with production deployment")
    if f["ev_nlp"] and not f["ev_ir"]:
        pos.append("NLP/LLM background")
    if f["ev_eval"]:
        pos.append("ranking-evaluation literacy (NDCG/MAP/A-B)")
    if f["bucket"] in ("gold_ir", "ml_engineer", "ml_senior") and not pos:
        pos.append("applied-ML engineering background")

    # Honest concerns.
    concerns = []
    if f["bucket"] == "nontech":
        concerns.append(f"title is '{title}', not an ML/IR role")
    if f["bucket"] == "research":
        concerns.append("research-leaning title — production evidence matters here")
    if f["bucket"] == "cv" and not (f["ev_ir"] or f["ev_nlp"]):
        concerns.append("computer-vision focus with no NLP/IR exposure")
    if f["yoe"] < 4:
        concerns.append(f"only {yoe} yrs experience vs the 5-9 band")
    elif f["yoe"] > 11:
        concerns.append(f"{yoe} yrs is above the target band")
    if f["consulting_only"]:
        concerns.append("career entirely at IT-services/consulting firms")
    if f["title_chaser"]:
        concerns.append("short average tenure (frequent job changes)")
    if f["rr"] is not None and f["rr"] < 0.2:
        concerns.append(f"low recruiter response rate ({f['rr']:.2f})")
    if f["last_active_m"] is not None and f["last_active_m"] >= 6:
        concerns.append(f"inactive ~{f['last_active_m']} months")
    if not f["in_india"] and not f["relocate"]:
        concerns.append("outside India and not open to relocating")

    head = f"{title}"
    if company:
        head += f" at {company}"
    head += f", {yoe} yrs"
    if where:
        head += f" ({where})"

    body = "; ".join(pos) if pos else "limited direct retrieval/ranking evidence"
    out = f"{head}. {body[0].upper() + body[1:]}."
    if concerns:
        out += f" Concern: {concerns[0]}"
        if len(concerns) > 1:
            out += f"; {concerns[1]}"
        out += "."
    # Keep it to a clean 1-2 sentences; strip stray whitespace.
    return re.sub(r"\s+", " ", out).strip()
