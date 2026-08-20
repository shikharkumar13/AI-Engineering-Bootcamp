"""
demo.py — Phase 02: Prompt Engineering
Run this file to see all extraction types and prompting techniques.

  python demo.py
"""

import json
from openai import OpenAI
from dotenv import load_dotenv
from collections import Counter

from extractor import DataExtractor
from models import EmailData, JobPosting, NewsArticle, MeetingNotes, Receipt

load_dotenv()
client = OpenAI()
extractor = DataExtractor()

DIVIDER = "=" * 62


def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 1: Zero-shot, Few-shot, CoT
# ─────────────────────────────────────────────────────────────────────────────

def demo_prompting_techniques():
    section("DEMO 1 — Zero-shot vs Few-shot vs CoT")

    # -- Zero-shot ----
    print("\n▸ Zero-shot sentiment classification")
    prompt = ("Classify the sentiment as positive, negative, or neutral.\n"
              "Review: 'Fast delivery, but the packaging was damaged and one item missing.'\n"
              "Sentiment:")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=5,
    )
    print(f"  Result: {resp.choices[0].message.content.strip()}")

    # -- Few-shot ----
    print("\n▸ Few-shot sentiment (with balanced examples)")
    few_shot_prompt = """Classify sentiment as positive, negative, or neutral. Output one word only.

Review: "Absolutely love it! Works perfectly and looks great."
Sentiment: positive

Review: "Broke after two days. Complete waste of money."
Sentiment: negative

Review: "It does the job. Nothing remarkable, but no complaints."
Sentiment: neutral

Review: "Fast delivery, but the packaging was damaged and one item missing."
Sentiment:"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": few_shot_prompt}],
        temperature=0, max_tokens=5,
    )
    print(f"  Result: {resp.choices[0].message.content.strip()}")

    # -- Chain-of-Thought ----
    print("\n▸ Chain-of-Thought (step-by-step reasoning)")
    cot_prompt = """A store offers a 20% discount on a $150 jacket.
Then there's an additional 10% off the discounted price.
What is the final price?

Let's think step by step."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": cot_prompt}],
        temperature=0,
    )
    print(f"  Reasoning:\n  {resp.choices[0].message.content[:300]}...")

    # -- Self-consistency ----
    print("\n▸ Self-consistency (5 samples, majority vote)")
    problem = ("A store offers a 20% discount on a $150 jacket. "
               "Then there's an additional 10% off the discounted price. "
               "What is the final price? Think step by step. "
               "End with 'Answer: $XX.XX'")
    answers = []
    for _ in range(5):
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": problem}],
            temperature=0.7,
        )
        text = r.choices[0].message.content
        for line in text.split('\n'):
            if 'answer:' in line.lower():
                answers.append(line.split(':')[-1].strip())
                break
    vote_counts = Counter(answers)
    winner, count = vote_counts.most_common(1)[0]
    print(f"  Votes: {dict(vote_counts)}")
    print(f"  Answer: {winner}  (confidence: {count}/5)")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 2: Email extraction
# ─────────────────────────────────────────────────────────────────────────────

def demo_email_extraction():
    section("DEMO 2 — Email Extraction (instructor + Pydantic)")

    emails = [
        # Urgent complaint
        """Hi Support Team,

I placed order #78234 on October 1st and paid $299 for expedited shipping (2 days).
It's been 12 days and the item still hasn't arrived. I contacted you on Oct 6th 
and Oct 9th — no response either time.

I need this resolved immediately. Either ship the item overnight at no charge 
or issue a full refund including shipping by EOD tomorrow.

If I don't hear back, I'm disputing the charge with my bank.

— David Chen""",

        # Simple meeting request
        """Hi Sarah,

Just following up on our chat last week. Can we schedule 30 minutes this week
to go over the Q4 roadmap? I'm free Thursday afternoon or Friday morning.

Also, could you share the slides from Monday's presentation?

Thanks!
Mike""",
    ]

    for i, email in enumerate(emails, 1):
        print(f"\n▸ Email {i}:")
        print(f"  Preview: '{email[:80].strip()}...'")

        result = extractor.extract_email(email)

        print(f"  Intent:   {result.intent}")
        print(f"  Priority: {result.priority.value}")
        print(f"  Sentiment:{result.sentiment.value}")
        print(f"  Summary:  {result.subject_summary}")
        if result.action_items:
            print(f"  Actions:")
            for item in result.action_items:
                deadline = f" (by {item.deadline})" if item.deadline else ""
                print(f"    • {item.task}{deadline}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 3: Job posting extraction
# ─────────────────────────────────────────────────────────────────────────────

def demo_job_posting():
    section("DEMO 3 — Job Posting Extraction")

    job_text = """
Senior Machine Learning Engineer — AI Products Team
Anthropic | San Francisco, CA (Hybrid)

We're looking for a Senior ML Engineer to join our AI Products team.
You'll work on deploying and optimizing large language models in production.

About the role:
- Design and implement ML pipelines for model training and inference
- Optimize model performance (latency, throughput, cost) at scale
- Collaborate with research teams to productionize new model capabilities
- Build tooling for model evaluation and monitoring

Requirements:
- 5+ years of ML engineering experience
- Strong Python skills (PyTorch, TensorFlow)
- Experience with distributed training and inference at scale
- Background with transformer architectures and LLMs
- Proficiency with cloud platforms (AWS, GCP, or Azure)
- Experience with MLOps tools (MLflow, Weights & Biases)

Nice to have:
- PhD in CS, ML, or related field
- Experience with RLHF or fine-tuning large models
- Publications at NeurIPS, ICML, or ICLR
- Familiarity with CUDA and GPU optimization

Compensation: $200,000 - $350,000 base salary + equity + benefits
Benefits: Health/dental/vision, 401k matching, unlimited PTO, remote work stipend
"""

    result = extractor.extract_job_posting(job_text)

    print(f"\n  Title:       {result.job_title}")
    print(f"  Company:     {result.company}")
    print(f"  Location:    {result.location}")
    print(f"  Work type:   {result.work_type.value}")
    print(f"  Salary:      ${result.salary_min:,} - ${result.salary_max:,}")
    print(f"  Experience:  {result.experience_years}")
    print(f"  Required skills ({len(result.required_skills)}):")
    for skill in result.required_skills[:5]:
        print(f"    • {skill}")
    print(f"  Nice to have ({len(result.nice_to_have_skills)}):")
    for skill in result.nice_to_have_skills[:3]:
        print(f"    • {skill}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 4: News article extraction (with CoT)
# ─────────────────────────────────────────────────────────────────────────────

def demo_news_article():
    section("DEMO 4 — News Article Extraction (with Chain-of-Thought)")

    article = """
OpenAI Launches GPT-5 with Unprecedented Reasoning Capabilities

SAN FRANCISCO, Nov 15 — OpenAI on Thursday unveiled GPT-5, its most powerful language
model to date, claiming significant improvements in mathematical reasoning and code
generation over its predecessor.

The model, available immediately to ChatGPT Plus subscribers at $20 per month, scored
92% on the MATH benchmark, compared to 76% for GPT-4o. In programming tasks, GPT-5
outperformed human experts on LeetCode hard problems in 68% of cases.

CEO Sam Altman described the release as "a meaningful step toward systems that can
reason like scientists." The announcement came just weeks after rival Google released
Gemini Ultra 2, which had briefly claimed the top position on several AI benchmarks.

Microsoft, which has invested over $13 billion in OpenAI, saw its stock rise 4.2%
following the announcement. Shares of Anthropic's competitor Claude also saw increased
traffic as users compared the two systems.

Enterprise customers can access GPT-5 via API at $30 per million input tokens, triple
the cost of GPT-4o, which OpenAI said reflects the increased computational requirements.
"""

    result = extractor.extract_news_article(article)

    print(f"\n  Headline:    {result.headline}")
    print(f"  Category:    {result.category}")
    print(f"  Sentiment:   {result.sentiment.value}")
    print(f"  Entities ({len(result.entities)}):")
    for entity in result.entities[:5]:
        role = f" — {entity.role}" if entity.role else ""
        print(f"    • [{entity.type}] {entity.name}{role}")
    print(f"  Key facts:")
    for fact in result.key_facts[:4]:
        print(f"    • {fact}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 5: Meeting notes extraction
# ─────────────────────────────────────────────────────────────────────────────

def demo_meeting_notes():
    section("DEMO 5 — Meeting Notes Extraction")

    notes = """
Product Sync — October 21, 2024
Attendees: Priya (CPO), James (Engineering Lead), Mei (Design), Tom (QA)

We opened by reviewing the October roadmap status. James confirmed the backend
API is complete and merged. Mei walked through the updated UI designs — the team
approved the new navigation but asked for one more iteration on the dark mode palette.

Key decisions:
1. We're moving the v2.0 launch from October 31 to November 8 to allow QA time.
2. The dark mode feature will NOT be in v2.0 — it'll ship in v2.1 instead.
3. Pricing for the Pro tier will stay at $29/month (no change from initial plan).

Action items:
- Mei to complete dark mode designs by Oct 28 (for v2.1 planning)
- Tom to begin QA testing on Nov 1, targeting 5 business days for full pass
- Priya to update the stakeholder deck with new launch date by this Friday
- James to write migration guide for existing API users by Nov 5

Open questions: Do we need a beta program for the new pricing tier? Decision deferred.

Next meeting: October 28 at 10am (launch readiness review)
"""

    result = extractor.extract_meeting_notes(notes)

    print(f"\n  Meeting:    {result.meeting_title}")
    print(f"  Date:       {result.date}")
    print(f"  Attendees:  {', '.join(result.attendees)}")
    print(f"  Decisions ({len(result.key_decisions)}):")
    for d in result.key_decisions:
        print(f"    • {d.decision}")
    print(f"  Action items ({len(result.action_items)}):")
    for item in result.action_items:
        print(f"    • [{item.assignee}] {item.task} → {item.deadline}")
    if result.open_questions:
        print(f"  Open questions:")
        for q in result.open_questions:
            print(f"    ? {q}")
    print(f"  Next meeting: {result.next_meeting}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 6: Receipt extraction
# ─────────────────────────────────────────────────────────────────────────────

def demo_receipt_extraction():
    section("DEMO 6 — Receipt Extraction")

    receipt = """
Coffee House Roasters
123 Main St, Portland, OR

Date: 03/14/2025

2x Latte (Large)          $5.50 each     $11.00
1x Blueberry Muffin        $3.75          $3.75
1x Cold Brew (Medium)      $4.25          $4.25

Subtotal:                                $19.00
Tax (8.5%):                               $1.62
Total:                                   $20.62

Payment: Visa ending in 4471
Thank you for visiting!
"""

    result = extractor.extract_receipt(receipt)

    print(f"\n  Merchant:  {result.merchant}")
    print(f"  Date:      {result.date}")
    print(f"  Items ({len(result.items)}):")
    for item in result.items:
        print(f"    • {item.name} — qty {item.quantity} @ ${item.unit_price} = ${item.total_price}")
    print(f"  Subtotal:  ${result.subtotal}")
    print(f"  Tax:       ${result.tax}")
    print(f"  Total:     ${result.total}")
    print(f"  Payment:   {result.payment_method}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 7: Auto-detect and extract
# ─────────────────────────────────────────────────────────────────────────────

def demo_auto_extract():
    section("DEMO 7 — Auto Document Type Detection")

    texts = [
        "Hi, I need to update my billing address for account #88234. Can you help?",
        "Software Engineer II — Stripe | Remote US | $160k-$220k | 3+ yrs Python required",
        "Apple reported Q3 earnings of $85.8B, beating analyst expectations by 6%.",
    ]

    for text in texts:
        print(f"\n▸ Input: '{text[:70]}...'")
        result = extractor.auto_extract(text)
        print(f"  Detected type: {result['document_type']}")
        data = result['data']
        # Print first 2 fields of whatever model came back
        data_dict = data.model_dump()
        for key, val in list(data_dict.items())[:3]:
            if val is not None:
                print(f"  {key}: {val}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 8: Raw function calling (under the hood)
# ─────────────────────────────────────────────────────────────────────────────

def demo_raw_function_calling():
    section("DEMO 8 — Raw Function Calling (what instructor does under the hood)")

    email = """
The Q3 report is ready for review. I've uploaded it to the shared drive.
Please review and send feedback by Wednesday.
— Lisa
"""

    print("\n▸ Calling raw function calling API...")
    result = extractor.extract_raw_function_calling(email)
    print("  Raw JSON arguments from model:")
    print("  " + json.dumps(result, indent=2).replace("\n", "\n  "))

    print("\n▸ Same extraction with instructor (cleaner):")
    structured = extractor.extract_email(email)
    print(f"  intent:   {structured.intent}")
    print(f"  priority: {structured.priority.value}")
    print(f"  actions:  {[a.task for a in structured.action_items]}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 9: Batch extraction
# ─────────────────────────────────────────────────────────────────────────────

def demo_batch():
    section("DEMO 9 — Batch Extraction")

    emails = [
        "Hi, can you update my shipping address? Order #1234. — Tom",
        "This product is TERRIBLE! I want a full refund NOW. Three weeks no delivery!",
        "Just wanted to say thanks, the team was super helpful. Great experience!",
    ]

    print(f"\n▸ Extracting {len(emails)} emails:")
    results = extractor.batch_extract(emails, EmailData)

    print(f"\n  Results:")
    for i, (email, result) in enumerate(zip(emails, results), 1):
        if isinstance(result, Exception):
            print(f"  {i}. ERROR: {result}")
        else:
            print(f"  {i}. [{result.priority.value.upper()}] {result.intent} "
                  f"| {result.sentiment.value} | '{result.subject_summary[:60]}'")


# ─────────────────────────────────────────────────────────────────────────────
# Run all demos
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_prompting_techniques()
    demo_email_extraction()
    demo_job_posting()
    demo_news_article()
    demo_meeting_notes()
    demo_receipt_extraction()
    demo_auto_extract()
    demo_raw_function_calling()
    demo_batch()

    print(f"\n{DIVIDER}")
    print("  ✅  Phase 02 complete.")
    print("  You now have a structured data extraction pipeline.")
    print(DIVIDER)
