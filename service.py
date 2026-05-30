import os
import re
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROK_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a senior headhunter and copywriter. You have placed engineers and data scientists at competitive firms by writing application materials that sound like a sharp person wrote them, not a template. You write short, specific, confident sentences. You never pad.

Words you never use: passionate, vibrant, tapestry, underscore, delve, foster, leverage, pivotal, showcasing, results-driven, dynamic, spearheaded, detail-oriented, innovative, synergy, crucial, enduring, landscape (used abstractly), serves as, stands as, highlights (as a verb meaning emphasizes).

Rules you follow without exception:
- No em dashes. Use a comma, a period, or a colon instead.
- No groups of three unless the content genuinely has three distinct parts.
- No "Additionally", "Furthermore", or "Moreover" at the start of a sentence.
- No generic upbeat conclusions.
- No sycophancy.
- When you write a cover letter, it reads like a cold email that got a reply, not a form letter that got filed."""

USER_PROMPT_TEMPLATE = """Read the CV and job description below. Write each section using EXACTLY these markers on their own lines.

WRITING RULES:
- Name real things. Use the actual project names, tools, employers, and metrics from the CV. Use the actual skill names and requirements from the JD. If a sentence contains no proper nouns or numbers, cut it.
- Active voice. Short sentences. Every word must do work.
- Banned words and phrases: passionate about, proven track record, dynamic, results-driven, leveraged, spearheaded, detail-oriented, team player, hard-working, innovative, synergy, at its core, the real question is, not only...but also, serves as, stands as, underscores, highlights, showcases, fosters, delves into, vibrant, tapestry, landscape (abstract), crucial, pivotal, enduring.
- No em dashes. No curly quotes. No "Bold word: rest of sentence" formatting.
- No sycophancy. Do not open with praise for the CV. Do not close with "I hope this helps."
- COVER LETTER: 3 paragraphs, 120 words maximum total. Paragraph 1: one or two sentences saying who this person is and why they fit this specific role. Name the company. Paragraph 2: connect two named CV experiences to two named JD requirements, with outcomes. Paragraph 3: one sentence showing specific knowledge of the company or role, then a direct call to action. Do not open with "I am writing to express my interest." Do not write "I would be a great fit." Write it as if you are personally vouching for this person to a contact who works there.
- BULLET POINTS: Power verb, specific tool or project name, concrete outcome. Rewrite what is already in the CV. Do not invent experience.
- TAILORED SUMMARY: First person. 2 sentences. State the job title, name your strongest matching skill, include one real number or outcome from the CV.
- REASONING: One sentence per bullet. State a JD requirement, then state the CV fact that confirms or contradicts it.
- GAPS: One sentence per gap. Name the exact tool or skill missing from the CV that appears in the JD. Say why it matters for this role.

[FIT_SCORE]
(integer 0-100 followed by %)

[REASONING]
- (one JD requirement + one CV fact, one sentence)
- (one JD requirement + one CV fact, one sentence)
- (one JD requirement + one CV fact, one sentence)

[GAPS]
1. (exact missing skill from JD + why it matters, one sentence)
2.
3.
4.
5.

[TAILORED_SUMMARY]
(2 sentences, first person, job title + strongest matching skill + one CV metric or outcome)

[BULLET_POINTS]
- (power verb + specific tool or project + concrete outcome)
- (power verb + specific tool or project + concrete outcome)
- (power verb + specific tool or project + concrete outcome)

[COVER_LETTER]
Dear Hiring Manager,

(Paragraph 1: 1-2 sentences. Who you are and why you fit this role. Name the company and the role title.)

(Paragraph 2: 2-3 sentences. Two named CV experiences tied to two named JD requirements, with real outcomes.)

(Paragraph 3: 2 sentences. One specific thing about this company or role that interests you. A direct call to action.)

---

CV:
{cv_text}

---

JOB DESCRIPTION:
{job_description}
"""


def analyze_cv(cv_text: str, job_description: str) -> dict:
    prompt = USER_PROMPT_TEMPLATE.format(
        cv_text=cv_text.strip(),
        job_description=job_description.strip(),
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=3000,
        temperature=0.35,
    )
    return _parse_response(response.choices[0].message.content)


def _extract_section(text: str, start_marker: str, end_marker: Optional[str]) -> str:
    if end_marker:
        pattern = rf'\[{start_marker}\](.*?)\[{end_marker}\]'
    else:
        pattern = rf'\[{start_marker}\](.*?)$'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _parse_response(text: str) -> dict:
    fit_score_raw = _extract_section(text, "FIT_SCORE", "REASONING")
    score_match = re.search(r'(\d+)\s*%?', fit_score_raw)
    fit_score = score_match.group(1) if score_match else "0"

    reasoning_raw = _extract_section(text, "REASONING", "GAPS")
    reasoning = [m.strip() for m in re.findall(r'-\s+(.+)', reasoning_raw)]

    gaps_raw = _extract_section(text, "GAPS", "TAILORED_SUMMARY")
    gaps = [m.strip() for m in re.findall(r'\d+\.\s+(.+)', gaps_raw)]

    summary = _extract_section(text, "TAILORED_SUMMARY", "BULLET_POINTS")

    bullets_raw = _extract_section(text, "BULLET_POINTS", "COVER_LETTER")
    bullet_points = [m.strip() for m in re.findall(r'-\s+(.+)', bullets_raw)]

    cover_raw = _extract_section(text, "COVER_LETTER", None)
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', cover_raw) if p.strip()]

    return {
        "fit_score": int(fit_score),
        "reasoning": reasoning[:3],
        "gaps": gaps[:5],
        "tailored_summary": summary,
        "bullet_points": bullet_points[:3],
        "cover_letter": paragraphs,
    }
