#!/usr/bin/env python3
"""
Import Goodreads library into MoodReads Notion DB
and auto-enrich each book with Mood/Tone/etc using OpenAI.

Usage:
    python goodreads_to_moodreads.py goodreads_library_export.csv
"""

import csv
import json
import os
import sys
from datetime import datetime
from getpass import getpass
import time

import requests

# ==============================
#  Configuration
# ==============================

NOTION_API_VERSION = "2022-06-28"

# Your MoodReads Notion database ID (from URL / file name)
DATABASE_ID = "5813a47dec1d4ae3ab2711433538ce4f"

# Throttling so we don't hammer APIs
OPENAI_SLEEP_SECONDS = 0.7
NOTION_SLEEP_SECONDS = 0.3

# ==============================
#  Helpers
# ==============================

def get_env_or_prompt(name: str, prompt_text: str) -> str:
    val = os.getenv(name)
    if val:
        return val.strip()
    return getpass(prompt_text).strip()


def load_valid_tag_lists(path: str = "mood_reads_fields.csv") -> dict:
    """
    Load allowed tags for Mood/Tone/etc from the CSV we generated.
    Assumes a single row, semicolon-separated values.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    def split(val: str):
        if not val:
            return []
        return [x.strip() for x in val.split(";") if x.strip()]

    return {
        "valid_moods": split(row.get("Mood", "")),
        "valid_tones": split(row.get("Tone", "")),
        "valid_setting_types": split(row.get("Setting Type", "")),
        "valid_aesthetics": split(row.get("Aesthetic", "")),
        "valid_themes": split(row.get("Themes", "")),
        "valid_tropes_general": split(row.get("Tropes General", "")),
        "valid_tropes_romance": split(row.get("Tropes Romance", "")),
        "valid_romantasy_creatures": split(row.get("Romantasy Creature", "")),
        "valid_spice_adult_themes": split(row.get("Spice / Adult Themes", "")),
    }


def clean_date_for_notion(date_str: str):
    """
    Goodreads dates are often like '2023/10/05' or '2023-10-05'.
    Convert to 'YYYY-MM-DD' or return None if empty/invalid.
    """
    if not date_str:
        return None
    date_str = date_str.strip()
    if not date_str or date_str.lower() == "null":
        return None

    # Try a few formats
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            dt = datetime.strptime(date_str.split(" ")[0], fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    # If we can't parse, just return None to be safe
    return None


def parse_float(val):
    try:
        if val is None or val == "":
            return None
        return float(val)
    except ValueError:
        return None


# ==============================
#  OpenAI: Metadata generation
# ==============================

def generate_book_metadata(openai_api_key: str,
                           title: str,
                           author: str,
                           description: str,
                           tag_lists: dict) -> dict | None:
    """
    Call OpenAI to generate mood/tone/etc based on the book,
    constrained to our allowed tag lists.
    """

    valid_moods = tag_lists["valid_moods"]
    valid_tones = tag_lists["valid_tones"]
    valid_setting_types = tag_lists["valid_setting_types"]
    valid_aesthetics = tag_lists["valid_aesthetics"]
    valid_themes = tag_lists["valid_themes"]
    valid_tropes_general = tag_lists["valid_tropes_general"]
    valid_tropes_romance = tag_lists["valid_tropes_romance"]
    valid_romantasy_creatures = tag_lists["valid_romantasy_creatures"]
    valid_spice_adult_themes = tag_lists["valid_spice_adult_themes"]

    if description is None:
        description = ""

    prompt = f"""
You are an expert librarian and story analyst. You label books with consistent,
reader-friendly metadata for a vibes-based TBR database.

ONLY use the allowed tags listed below. Do NOT invent new tags. If nothing fits,
return an empty list [] for that field.

Allowed tags:

- mood_tags: {", ".join(valid_moods)}
- tone_tags: {", ".join(valid_tones)}
- setting_type_tags: {", ".join(valid_setting_types)}
- aesthetic_tags: {", ".join(valid_aesthetics)}
- theme_tags: {", ".join(valid_themes)}
- tropes_general_tags: {", ".join(valid_tropes_general)}
- tropes_romance_tags: {", ".join(valid_tropes_romance)}
- romantasy_creature_tags: {", ".join(valid_romantasy_creatures)}
- spice_adult_theme_tags: {", ".join(valid_spice_adult_themes)}

For each book, return a SINGLE JSON object with exactly these keys:

- "mood": list of 1-3 mood tags from mood_tags
- "tone": list of 1-3 tone tags from tone_tags
- "setting_type": list of 0-3 tags from setting_type_tags
- "aesthetic": list of 0-3 tags from aesthetic_tags
- "themes": list of 0-5 tags from theme_tags
- "tropes_general": list of 0-6 tags from tropes_general_tags
- "tropes_romance": list of 0-6 tags from tropes_romance_tags
- "romantasy_creature": list of 0-2 tags from romantasy_creature_tags
- "spice_adult_themes": list of 0-6 tags from spice_adult_theme_tags

Plus these more freeform fields:

- "pacing": one of ["slow", "medium", "fast"]
- "energy_required": one of ["low", "medium", "high"]
- "spice_level": integer 0-5 (0 = none, 5 = explicit & frequent)
- "vibes": short ~1 sentence vibe summary (max 140 chars)
- "when_where_to_read": short phrase like "rainy weekend at home"
- "time_of_year": list of 1-3 from ["spring", "summer", "autumn", "winter", "anytime"]

If you are unsure for any tag list, you may return an empty list [] for that field.

Be concise. Reply with JSON ONLY, no extra text.

Book:
Title: {title}
Author: {author}
Description / user review text:
{description}
"""

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "gpt-4o-mini",  # fast and cost-effective model for structured output
        "messages": [
            {"role": "system", "content": "You are a precise JSON-only book metadata generator."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
    }

    res = requests.post(url, headers=headers, json=body)
    if res.status_code != 200:
        print(f"   ❌ OpenAI error {res.status_code}")
        try:
            print("   ", res.text)
        except Exception:
            pass
        return None

    try:
        content = res.json()["choices"][0]["message"]["content"].strip()
        # Remove markdown code block wrapper if present
        if content.startswith("```"):
            # Find the end of the opening ``` line
            first_newline = content.find("\n")
            # Find the closing ```
            last_backticks = content.rfind("```")
            if first_newline != -1 and last_backticks > first_newline:
                content = content[first_newline + 1:last_backticks].strip()
        data = json.loads(content)
        return data
    except Exception as e:
        print("   ❌ Failed to parse OpenAI JSON:", e)
        try:
            print("   Raw:", res.text[:500])
        except Exception:
            pass
        return None


# ==============================
#  Notion: create pages
# ==============================

def ms(names):
    """Helper to build multi_select entries."""
    return [{"name": n} for n in names if n]


def build_notion_properties(row: dict, metadata: dict) -> dict:
    """
    Map Goodreads row + AI metadata into Notion properties payload.
    """

    title = (row.get("Title") or "").strip()
    author = (row.get("Author") or "").strip()
    my_rating = parse_float(row.get("My Rating"))
    date_added_raw = (row.get("Date Added") or "").strip()
    exclusive_shelf = (row.get("Exclusive Shelf") or "").strip().lower()
    my_review = (row.get("My Review") or "").strip()

    props = {}

    # Title (Notion title property)
    if title:
        props["Title"] = {
            "title": [
                {"type": "text", "text": {"content": title}}
            ]
        }

    # Author
    if author:
        props["Author"] = {
            "rich_text": [
                {"type": "text", "text": {"content": author}}
            ]
        }

    # Description: start with Goodreads "My Review" if present
    if my_review:
        props["Description"] = {
            "rich_text": [
                {"type": "text", "text": {"content": my_review[:1900]}}
            ]
        }

    # Date Added
    date_clean = clean_date_for_notion(date_added_raw)
    if date_clean:
        props["Date Added"] = {"date": {"start": date_clean}}

    # Goodreads Rating (we're putting your rating here)
    if my_rating is not None and my_rating > 0:
        props["Goodreads Rating"] = {"number": my_rating}

    # Read Status from Goodreads exclusive shelf
    status_map = {
        "to-read": "want to read",
        "currently-reading": "currently reading",
        "read": "read",
    }
    status_val = status_map.get(exclusive_shelf)
    if status_val:
        # Assumes "Read Status" is a select property
        props["Read Status"] = {"select": {"name": status_val}}

    # === AI metadata mapping ===
    mood_tags = metadata.get("mood") or []
    tone_tags = metadata.get("tone") or []
    setting_tags = metadata.get("setting_type") or []
    aesthetic_tags = metadata.get("aesthetic") or []
    theme_tags = metadata.get("themes") or []
    tropes_general_tags = metadata.get("tropes_general") or []
    tropes_romance_tags = metadata.get("tropes_romance") or []
    romantasy_creature_tags = metadata.get("romantasy_creature") or []
    spice_adult_tags = metadata.get("spice_adult_themes") or []

    pacing = metadata.get("pacing")
    energy = metadata.get("energy_required")
    spice_level = metadata.get("spice_level")
    vibes_text = metadata.get("vibes") or ""
    when_where = metadata.get("when_where_to_read") or ""
    time_of_year_tags = metadata.get("time_of_year") or []

    if mood_tags:
        props["Mood"] = {"multi_select": ms(mood_tags)}

    if tone_tags:
        props["Tone"] = {"multi_select": ms(tone_tags)}

    if setting_tags:
        props["Setting Type"] = {"multi_select": ms(setting_tags)}

    if aesthetic_tags:
        props["Aesthetic"] = {"multi_select": ms(aesthetic_tags)}

    if theme_tags:
        props["Themes"] = {"multi_select": ms(theme_tags)}

    if tropes_general_tags:
        props["Tropes General"] = {"multi_select": ms(tropes_general_tags)}

    if tropes_romance_tags:
        props["Tropes Romance"] = {"multi_select": ms(tropes_romance_tags)}

    if romantasy_creature_tags:
        props["Romantasy Creature"] = {"multi_select": ms(romantasy_creature_tags)}

    if spice_adult_tags:
        props["Spice / Adult Themes"] = {"multi_select": ms(spice_adult_tags)}

    if pacing:
        props["Pacing"] = {"multi_select": ms([pacing])}

    if energy:
        props["Energy Required"] = {"multi_select": ms([energy])}

    if isinstance(spice_level, (int, float)):
        try:
            props["Spice Level"] = {"number": float(spice_level)}
        except Exception:
            pass

    if vibes_text:
        props["Vibes"] = {
            "rich_text": [
                {"type": "text", "text": {"content": vibes_text[:200]}}
            ]
        }

    if when_where:
        props["When/Where to Read"] = {
            "rich_text": [
                {"type": "text", "text": {"content": when_where[:200]}}
            ]
        }

    if time_of_year_tags:
        props["Time of Year"] = {"multi_select": ms(time_of_year_tags)}

    return props


def create_notion_page(notion_token: str, properties: dict) -> bool:
    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_API_VERSION,
    }
    payload = {
        "parent": {"database_id": DATABASE_ID},
        "properties": properties,
    }
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code != 200:
        print(f"   ❌ Notion error {res.status_code}")
        try:
            print("   ", res.text[:500])
        except Exception:
            pass
        return False
    return True


# ==============================
#  Main
# ==============================

def main():
    if len(sys.argv) < 2:
        print("Usage: python goodreads_to_moodreads.py goodreads_library_export.csv")
        sys.exit(1)

    goodreads_csv_path = sys.argv[1]

    if not os.path.exists(goodreads_csv_path):
        print(f"❌ File not found: {goodreads_csv_path}")
        sys.exit(1)

    notion_token = get_env_or_prompt("NOTION_TOKEN", "Notion integration token: ")
    openai_api_key = get_env_or_prompt("OPENAI_API_KEY", "OpenAI API key: ")

    print("📚 Loading allowed tag lists from mood_reads_fields.csv...")
    tag_lists = load_valid_tag_lists()

    print(f"📥 Reading Goodreads export: {goodreads_csv_path}")
    with open(goodreads_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    print(f"   Found {total} rows in Goodreads export.")

    processed = 0
    success = 0
    failed = 0

    for row in rows:
        title = (row.get("Title") or "").strip()
        author = (row.get("Author") or "").strip()

        if not title:
            continue

        print(f"\n📖 {title} — {author}")

        # Use My Review as some description context; if empty, it's fine
        description = (row.get("My Review") or "").strip()

        # Call OpenAI to get metadata
        print("   ✨ Generating metadata via OpenAI...")
        metadata = generate_book_metadata(openai_api_key, title, author, description, tag_lists)
        if metadata is None:
            print("   ⚠️ Skipping AI metadata (parse error).")
            metadata = {}

        # Build Notion properties
        props = build_notion_properties(row, metadata)

        # Create page in Notion
        print("   🧱 Creating Notion page...")
        ok = create_notion_page(notion_token, props)
        if ok:
            success += 1
            print("   ✅ Created.")
        else:
            failed += 1
            print("   ❌ Failed to create page.")

        processed += 1

        # Respect rate limits
        time.sleep(NOTION_SLEEP_SECONDS + OPENAI_SLEEP_SECONDS)

    print("\n====== Done ======")
    print(f"Processed: {processed}")
    print(f"Created successfully: {success}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
