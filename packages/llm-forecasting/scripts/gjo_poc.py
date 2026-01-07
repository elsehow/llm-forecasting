#!/usr/bin/env python
"""Proof-of-concept: Fetch questions from Good Judgment Open.

GJO doesn't have a public API, so we scrape the HTML pages.
This script demonstrates that we can extract:
- Question IDs and titles
- Crowd forecast probabilities
- Close dates
- Resolution status
"""

import asyncio
import re

import httpx
from bs4 import BeautifulSoup

BASE_URL = "https://www.gjopen.com"


async def fetch_question_ids(client: httpx.AsyncClient, limit: int = 5) -> list[str]:
    """Fetch question IDs from the questions list page."""
    response = await client.get(f"{BASE_URL}/questions", params={"status": "active"})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Find question links - they can be full URLs or relative
    # Full: https://www.gjopen.com/questions/4943-slug-here
    # Relative: /questions/4943-slug-here
    question_ids = []
    pattern = re.compile(r"(?:https://www\.gjopen\.com)?/questions/(\d+)")

    for link in soup.find_all("a", href=pattern):
        href = link.get("href", "")
        match = pattern.search(href)
        if match:
            qid = match.group(1)
            if qid not in question_ids:
                question_ids.append(qid)
                if len(question_ids) >= limit:
                    break

    return question_ids


async def fetch_question(client: httpx.AsyncClient, question_id: str) -> dict | None:
    """Fetch a single question's details including probability."""
    response = await client.get(f"{BASE_URL}/questions/{question_id}")
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title - GJO uses h3 for question titles
    title = None
    # Try h3 first (GJO's format), then h1, h2
    for tag in ["h3", "h1", "h2"]:
        heading = soup.find(tag)
        if heading:
            text = heading.get_text(strip=True)
            # Skip non-question headings
            if text and len(text) > 20 and "sign up" not in text.lower():
                title = text
                break

    # Extract probabilities - look for percentage patterns
    # GJO shows probabilities in various formats
    probabilities = []

    # Look for elements containing percentages (e.g., "64%")
    for text in soup.stripped_strings:
        # Match patterns like "64%" or "64.5%"
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if pct_match:
            pct = float(pct_match.group(1))
            # Filter out likely non-forecast percentages (like "100% complete")
            if 0 < pct < 100:
                probabilities.append(pct)

    # Deduplicate while preserving order
    seen = set()
    unique_probs = []
    for p in probabilities:
        if p not in seen:
            seen.add(p)
            unique_probs.append(p)

    # Extract close date - look for date patterns
    close_date = None
    for text in soup.stripped_strings:
        # Match patterns like "Jun 30, 2026" or "January 1, 2027"
        date_match = re.search(
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}",
            text,
            re.IGNORECASE,
        )
        if date_match:
            close_date = date_match.group(0)
            break

    # Check resolution status
    resolved = False
    resolution = None
    page_text = soup.get_text().lower()
    if "resolved" in page_text or "closed" in page_text:
        resolved = True
        # Try to find resolution value
        if "resolved: yes" in page_text:
            resolution = "Yes"
        elif "resolved: no" in page_text:
            resolution = "No"

    return {
        "id": question_id,
        "title": title,
        "probabilities": unique_probs[:5],  # Limit to first 5 (for multi-choice questions)
        "close_date": close_date,
        "resolved": resolved,
        "resolution": resolution,
        "url": f"{BASE_URL}/questions/{question_id}",
    }


async def main():
    """Fetch and display questions from Good Judgment Open."""
    print("=" * 60)
    print("Good Judgment Open - Proof of Concept")
    print("=" * 60)
    print()

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Get question IDs
        print("Fetching question IDs from list page...")
        question_ids = await fetch_question_ids(client, limit=5)
        print(f"Found {len(question_ids)} question IDs: {question_ids}")
        print()

        # Step 2: Fetch each question's details
        print("Fetching individual question details...")
        print()

        for i, qid in enumerate(question_ids, 1):
            question = await fetch_question(client, qid)
            if question:
                print(f"{i}. Question ID: {question['id']}")
                print(f"   Title: {question['title'][:70]}..." if question["title"] and len(question["title"]) > 70 else f"   Title: {question['title']}")
                if question["probabilities"]:
                    probs_str = " / ".join(f"{p}%" for p in question["probabilities"])
                    print(f"   Probabilities: {probs_str}")
                else:
                    print("   Probabilities: (not found)")
                print(f"   Close Date: {question['close_date']}")
                print(f"   Resolved: {question['resolved']}")
                print(f"   URL: {question['url']}")
                print()

            # Small delay to be polite to the server
            await asyncio.sleep(0.5)

    print("=" * 60)
    print("Done! Successfully scraped GJO questions.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
