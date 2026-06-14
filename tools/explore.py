#!/usr/bin/env python3
"""Streaming exploration of the candidate pool. Stdlib only — no deps needed.

Profiles title/industry distributions, behavioral-signal ranges, skill frequency,
and counts profiles that trip honeypot / inconsistency heuristics. Grounds the
feature engineering for the ranker.
"""
import argparse
import json
from collections import Counter
from datetime import date

# Reference "today" — dataset signals are anchored around mid-2026.
TODAY = date(2026, 6, 14)

AI_CORE = {
    "nlp", "information retrieval", "retrieval", "rag", "embeddings", "vector search",
    "vector database", "semantic search", "transformers", "fine-tuning llms", "llm",
    "llms", "pytorch", "tensorflow", "ranking", "learning to rank", "recommendation",
    "recommender systems", "elasticsearch", "faiss", "pinecone", "weaviate", "qdrant",
    "milvus", "sentence-transformers", "bert", "machine learning", "deep learning",
}


def parse_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def months_between(d, ref=TODAY):
    if d is None:
        return None
    return (ref.year - d.year) * 12 + (ref.month - d.month)


def honeypot_flags(c):
    """Return list of inconsistency reasons (a profile is suspicious if non-empty)."""
    reasons = []
    prof = c.get("profile", {})
    yoe = prof.get("years_of_experience") or 0
    ch = c.get("career_history", []) or []
    skills = c.get("skills", []) or []

    # Sum of career durations far exceeds stated YoE (overlap is normal, so use a margin)
    total_months = sum((j.get("duration_months") or 0) for j in ch)
    if yoe > 0 and total_months > (yoe * 12) * 2.0 + 24:
        reasons.append("career_months>>yoe")

    # "expert" or "advanced" proficiency but 0 months used
    zero_dur_expert = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and (s.get("duration_months") or 0) == 0
    )
    if zero_dur_expert >= 3:
        reasons.append(f"{zero_dur_expert}_high_prof_0_months")

    # A skill used longer than the person has worked (by a wide margin)
    if yoe > 0:
        for s in skills:
            if (s.get("duration_months") or 0) > yoe * 12 + 24:
                reasons.append("skill_dur>career")
                break

    # Job that starts before candidate plausibly entered workforce vs a job duration
    # longer than the gap to today, or tenure exceeding company plausibility.
    for j in ch:
        sd = parse_date(j.get("start_date"))
        dur = j.get("duration_months") or 0
        if sd is not None:
            elapsed = months_between(sd)
            if elapsed is not None and dur > elapsed + 2:
                reasons.append("duration>time_since_start")
                break

    # Current job with no end but is_current False, or end before start
    for j in ch:
        sd, ed = parse_date(j.get("start_date")), parse_date(j.get("end_date"))
        if sd and ed and ed < sd:
            reasons.append("end_before_start")
            break

    return reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default="explore_report.json")
    args = ap.parse_args()

    n = 0
    titles = Counter()
    industries = Counter()
    countries = Counter()
    skill_freq = Counter()
    proficiency = Counter()
    work_mode = Counter()

    yoe_buckets = Counter()
    resp_rates = []
    last_active_days = []
    github_scores = []
    open_to_work = 0
    saved_recruiters = []

    honeypot_count = 0
    honeypot_reasons = Counter()
    honeypot_examples = []

    # keyword-stuffer probe: many AI skills but a clearly non-AI title
    stuffer_examples = []

    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            n += 1

            prof = c.get("profile", {})
            title = (prof.get("current_title") or "").strip()
            titles[title] += 1
            industries[(prof.get("current_industry") or "").strip()] += 1
            countries[(prof.get("country") or "").strip()] += 1

            yoe = prof.get("years_of_experience") or 0
            yoe_buckets[int(yoe // 2) * 2] += 1

            skills = c.get("skills", []) or []
            ai_hits = 0
            for s in skills:
                name = (s.get("name") or "").strip()
                skill_freq[name.lower()] += 1
                proficiency[s.get("proficiency")] += 1
                if name.lower() in AI_CORE:
                    ai_hits += 1

            sig = c.get("redrob_signals", {})
            rr = sig.get("recruiter_response_rate")
            if rr is not None:
                resp_rates.append(rr)
            la = months_between(parse_date(sig.get("last_active_date")))
            if la is not None:
                last_active_days.append(la)  # in months actually
            gh = sig.get("github_activity_score")
            if gh is not None and gh >= 0:
                github_scores.append(gh)
            if sig.get("open_to_work_flag"):
                open_to_work += 1
            sr = sig.get("saved_by_recruiters_30d")
            if sr is not None:
                saved_recruiters.append(sr)
            work_mode[sig.get("preferred_work_mode")] += 1

            reasons = honeypot_flags(c)
            if reasons:
                honeypot_count += 1
                for r in reasons:
                    honeypot_reasons[r] += 1
                if len(honeypot_examples) < 15:
                    honeypot_examples.append({
                        "id": c.get("candidate_id"), "title": title,
                        "yoe": yoe, "reasons": reasons,
                    })

            tl = title.lower()
            non_ai_title = any(k in tl for k in (
                "marketing", "sales", "accountant", "hr ", "human resource",
                "graphic", "content writer", "civil engineer", "mechanical",
                "customer support", "operations manager", "project manager",
            ))
            if ai_hits >= 6 and non_ai_title and len(stuffer_examples) < 15:
                stuffer_examples.append({
                    "id": c.get("candidate_id"), "title": title,
                    "ai_skills": ai_hits, "yoe": yoe,
                })

            if args.limit and n >= args.limit:
                break

    def stats(xs):
        if not xs:
            return {}
        xs = sorted(xs)
        m = len(xs)
        return {
            "n": m, "min": round(xs[0], 2), "max": round(xs[-1], 2),
            "median": round(xs[m // 2], 2),
            "mean": round(sum(xs) / m, 2),
            "p10": round(xs[int(m * 0.1)], 2), "p90": round(xs[int(m * 0.9)], 2),
        }

    report = {
        "total": n,
        "top_titles": titles.most_common(40),
        "top_industries": industries.most_common(20),
        "top_countries": countries.most_common(15),
        "yoe_buckets": dict(sorted(yoe_buckets.items())),
        "top_skills": skill_freq.most_common(50),
        "proficiency": dict(proficiency),
        "work_mode": dict(work_mode),
        "open_to_work_pct": round(100 * open_to_work / max(n, 1), 1),
        "recruiter_response_rate": stats(resp_rates),
        "last_active_months_ago": stats(last_active_days),
        "github_activity_score_when_linked": stats(github_scores),
        "saved_by_recruiters_30d": stats(saved_recruiters),
        "honeypot_suspect_count": honeypot_count,
        "honeypot_reasons": dict(honeypot_reasons),
        "honeypot_examples": honeypot_examples,
        "keyword_stuffer_examples": stuffer_examples,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Profiled {n} candidates -> {args.out}")
    print(f"Top titles: {titles.most_common(12)}")
    print(f"Countries: {countries.most_common(8)}")
    print(f"YoE buckets: {dict(sorted(yoe_buckets.items()))}")
    print(f"open_to_work: {report['open_to_work_pct']}%")
    print(f"response_rate stats: {report['recruiter_response_rate']}")
    print(f"last_active (months ago): {report['last_active_months_ago']}")
    print(f"honeypot suspects: {honeypot_count} reasons={dict(honeypot_reasons)}")
    print(f"stuffer examples: {len(stuffer_examples)} (e.g. {stuffer_examples[:3]})")


if __name__ == "__main__":
    main()
