#!/usr/bin/env python3
"""
MoodReads Enrichment Importer
=============================

This script takes a Goodreads export CSV and enriches each book with
metadata for the MoodReads Notion database. It performs the following steps:

1. Reads a Goodreads export CSV file containing fields such as Title,
   Author, My Rating, Date Added and Exclusive Shelf.
2. Calls an LLM (via the OpenAI API) to generate mood, tone, trope and
   additional metadata based on a description or review of the book. The
   model is constrained to use only allowed tags read from a tags CSV file.
3. Fetches a cover image URL using the Brave Search API. The first
   image result ending in `.jpg`, `.jpeg` or `.png` is used.
4. Maps the LLM output plus Goodreads information into Notion properties
   according to the schema defined by the user. Property types are
   explicitly controlled (text, select, multi-select, checkbox or number).
5. Creates a new page in the target Notion database for each book.

Required environment variables:
    NOTION_TOKEN        - Your Notion integration token
    OPENAI_API_KEY      - Your OpenAI API key
    NOTION_DATABASE_ID  - The ID of your MoodReads database

Optional environment variables:
    BRAVE_API_KEY           - Brave search API key for cover image fetching
    OPENAI_MODEL            - Model to use (default: gpt-4o-mini)
    MOODREADS_TAG_LIST      - Path to allowed tags CSV (default: moodreads_tags.csv)
    OPENAI_SLEEP_SECONDS    - Delay between OpenAI calls (default: 0.5)
    NOTION_SLEEP_SECONDS    - Delay between Notion calls (default: 0.3)
    BRAVE_SLEEP_SECONDS     - Delay between Brave calls (default: 0.3)

Usage:
    python3 moodreads_importer.py /path/to/goodreads_export.csv
    python3 moodreads_importer.py --help
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------

NOTION_API_VERSION = "2022-06-28"

# Default model for metadata generation
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Sleep durations between requests to avoid hitting rate limits
OPENAI_SLEEP_SECONDS = float(os.getenv("OPENAI_SLEEP_SECONDS", "0.5"))
NOTION_SLEEP_SECONDS = float(os.getenv("NOTION_SLEEP_SECONDS", "0.3"))
BRAVE_SLEEP_SECONDS = float(os.getenv("BRAVE_SLEEP_SECONDS", "0.3"))

# Path to the allowed tag list CSV
TAG_LIST_CSV = os.getenv("MOODREADS_TAG_LIST", "moodreads_tags.csv")

# Retry configuration for HTTP requests
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 1.0  # 1s, 2s, 4s
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

# -----------------------------------------------------------------------------
# Notion Property Type Definitions
# -----------------------------------------------------------------------------

TEXT_PROPS = [
    "Author",
    "Audiobook Best-Use",
    "AI Metadata Version",
    "Best For Reader Type",
    "Vibe Color Palette",
    "Insufficient Data Flags",
    "Signal Strength Scores",
    "Who Should Avoid",
    "Reader Brain State",
    "Magic System Flavor",
    "Description",
    "AI Evidence Snippets",
    "Personal Notes",
    "Vibes",
    "Historical Setting",
    "AI Confidence Scores",
]

SELECT_PROPS = [
    "Priority to Read",
    "Relationship Dynamic",
    "Twist Factor",
    "Historical Event Focus",
    "Hard vs Soft Sci-Fi",
    "Horror Intensity",
    "Emotional Arc Curve",
    "Tech Level",
    "Gore/Violence Level",
    "Spice Level",
    "Emotional Impact",
    "Investigation Type",
    "World-Building Depth",
    "Read Status",
    "Character Archetypes",
    "Atmospheric vs Jump Scares",
    "Emotional Safety Level",
    "Ending Type",
    "Emotional Color",
    "Mystery Type",
    "HEA vs HFN",
    "Character Count",
    "Standalone vs Series",
    "POV",
    "Chapter Length Feel",
    "Romance Subplot",
    "Pacing",
    "Energy Required",
    "Historical Accuracy",
    "Age Appropriateness",
    "Emotional Weight",
    "Technical Complexity",
    "Series Status",
    "Source/Where I Found It",
    "Skimmability / Density",
    "Steam vs Fade-to-Black",
    "Time Period",
    "Magic System Complexity",
    "Writing Style",
    "Suspense Level",
    "Sci-Fi Tone",
    "Monster/Threat",
    # Nonfiction-specific select fields
    "Academic Rigor",
    "Research Intensity",
    "Cognitive Load",
    "Political Weight",
    "Teaching Style",
]

MULTI_SELECT_PROPS = [
    "Fantasy Sub-genre",
    "Audiobook Characterization Style",
    "Representation",
    "Time of Year",
    "Tropes General",
    "Sci-Fi Sub-genre",
    "Tone",
    "Genre",
    "Tropes Romance",
    "Audiobook Vibe Tags",
    "Audiobook Format",
    "Content/Trigger Warnings",
    "Spice / Adult Themes",
    "Romance Sub-genre",
    "When/Where to Read",
    "Setting Type",
    "Mood",
    "Aesthetic",
    "Themes",
    "Romantasy Creature",
    "Scare Type",
    # Nonfiction-specific multi-select fields
    "Nonfiction Category / Type",
    "Author Lens",
    "Reader Outcome / Learning Goal",
    "Controversial Topic Categories",
]

CHECKBOX_PROPS = [
    "Audiobook Available?",
    "Coming-of-Age",
    "Found Family",
    "Social Commentary",
    "Character Study",
]

NUMBER_PROPS = [
    "Series - Book Number",
    "Goodreads Rating",
    "Fandom Overlap Index",
    "Page Count",
    "Publication Year",
]

# Mapping of Notion property names to metadata keys
META_KEY_MAP: dict[str, str] = {
    "Spice Level": "spice_level",
    "Series - Book Number": "series_book_number",
    "Goodreads Rating": "goodreads_rating",
    "Fandom Overlap Index": "fandom_overlap_index",
    "Page Count": "page_count",
    "Publication Year": "publication_year",
    "Suspense Level": "suspense_level",
    "Atmospheric vs Jump Scares": "atmospheric_vs_jump_scares",
    # Nonfiction property mappings
    "Nonfiction Category / Type": "nonfiction_category",
    "Academic Rigor": "academic_rigor",
    "Research Intensity": "research_intensity",
    "Author Lens": "author_lens",
    "Reader Outcome / Learning Goal": "reader_goal",
    "Cognitive Load": "cognitive_load",
    "Political Weight": "political_weight",
    "Controversial Topic Categories": "controversial_topic",
    "Teaching Style": "teaching_style",
}

# LLM prompt template for metadata generation
METADATA_PROMPT_TEMPLATE = """
You are an expert librarian and story analyst. You assign metadata to
books for a vibes-based TBR database. Only use the allowed tags below.
Do not invent new labels or paraphrase existing ones. If no tag applies
for a field, output an empty list [].

Allowed tags by key:
{allowed_text}

For the following book, output a single JSON object with keys:
  mood, tone, setting_type, aesthetic, themes, tropes_general,
  tropes_romance, romantasy_creature, spice_adult_themes,
  pacing, energy_required, spice_level, vibes, when_where_to_read,
  time_of_year, hard_vs_soft_sci_fi, investigation_type,
  world_building_depth, character_archetypes, monster_threat,
  character_count, pov, chapter_length_feel, romance_subplot,
  skimmability_density, emotional_impact, emotional_arc_curve,
  tech_level, gore_violence_level, twist_factor, emotional_color,
  mystery_type, hea_vs_hfn, standalone_vs_series, suspense_level,
  sci_fi_tone, magic_system_complexity, writing_style,
  historical_event_focus, historical_accuracy,
  sci_fi_sub_genre, fantasy_sub_genre, romance_sub_genre,
  audiobook_vibe_tags, audiobook_format, audiobook_characterization_style,
  audiobook_best_use, audiobook_available, coming_of_age,
  found_family, social_commentary, character_study, emotional_weight,
  magic_system_flavor, technical_complexity, age_appropriateness,
  historical_era, time_period, themes_secondary, scare_type,
  horror_intensity, atmospheric_vs_jump_scares, seasonality,
  nonfiction_category, academic_rigor, research_intensity,
  author_lens, reader_goal, cognitive_load, political_weight,
  controversial_topic, teaching_style

Return JSON only. Do not wrap in markdown fences. Use concise lists.

Book Title: {title}
Author: {author}
Description: {description}
"""


# -----------------------------------------------------------------------------
# HTTP Session with Retry Logic
# -----------------------------------------------------------------------------


def create_retry_session(
    retries: int = MAX_RETRIES,
    backoff_factor: float = RETRY_BACKOFF_FACTOR,
    status_forcelist: list[int] | None = None,
) -> requests.Session:
    """Create a requests session with automatic retry logic.

    Args:
        retries: Maximum number of retries for failed requests.
        backoff_factor: Factor for exponential backoff between retries.
        status_forcelist: HTTP status codes that trigger a retry.

    Returns:
        A configured requests.Session with retry capabilities.
    """
    if status_forcelist is None:
        status_forcelist = RETRY_STATUS_CODES

    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# Global session for reuse
_http_session: requests.Session | None = None


def get_http_session() -> requests.Session:
    """Get or create the global HTTP session with retry logic."""
    global _http_session
    if _http_session is None:
        _http_session = create_retry_session()
    return _http_session


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def clean_date_for_notion(date_str: str) -> str | None:
    """Convert various date formats from Goodreads into ISO date strings.

    Goodreads often exports dates in formats like 'YYYY/MM/DD', 'YYYY-MM-DD',
    'MM/DD/YY', etc. This function tries several formats and returns
    'YYYY-MM-DD' if successful, otherwise returns None.

    Args:
        date_str: A date string from Goodreads export.

    Returns:
        ISO formatted date string (YYYY-MM-DD) or None if parsing fails.
    """
    if not date_str:
        return None

    date_str = date_str.strip()
    if not date_str or date_str.lower() in ("null", "none", ""):
        return None

    # Handle datetime strings by extracting date part
    date_part = date_str.split(" ")[0]

    formats = [
        "%Y/%m/%d",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%d/%m/%Y",
        "%d/%m/%y",
        "%Y.%m.%d",
        "%d.%m.%Y",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_part, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    logger.warning("Could not parse date: %s", date_str)
    return None


def coerce_select(value: Any) -> str | None:
    """Ensure that a value is a single string suitable for a Notion select.

    If the value is a list, return the first element. If it is a string
    already, strip whitespace. If the value is falsy, return None.

    Args:
        value: The value to coerce (string, list, or other).

    Returns:
        A single string or None.
    """
    if value is None:
        return None

    if isinstance(value, list):
        if not value:
            return None
        first = value[0]
        return str(first).strip() if first else None

    if isinstance(value, str):
        val = value.strip()
        return val if val else None

    # Fall back to string representation
    val = str(value).strip()
    return val if val else None


def coerce_multi(value: Any) -> list[str]:
    """Ensure that a value is a list of strings suitable for Notion multi-select.

    Accepts lists or strings separated by semicolons or commas. Filters out
    empty values and strips whitespace.

    Args:
        value: The value to coerce (string, list, or other).

    Returns:
        A list of non-empty strings.
    """
    if not value:
        return []

    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    if isinstance(value, str):
        # Try splitting on semicolon first, then comma
        for sep in [";", ","]:
            if sep in value:
                items = [p.strip() for p in value.split(sep)]
                return [i for i in items if i]
        # Single value
        stripped = value.strip()
        return [stripped] if stripped else []

    return [str(value).strip()] if str(value).strip() else []


def build_meta_key(property_name: str | None) -> str:
    """Derive a metadata key from a Notion property name.

    Converts spaces and punctuation to underscores and lowercases the result.

    Args:
        property_name: The Notion property name.

    Returns:
        A snake_case key suitable for metadata dictionary lookup.
    """
    if not property_name:
        return ""

    if property_name in META_KEY_MAP:
        return META_KEY_MAP[property_name]

    # Replace special characters with underscore
    key = property_name
    for ch in " -/?:.()":
        key = key.replace(ch, "_")
    key = key.lower()

    # Collapse multiple underscores
    while "__" in key:
        key = key.replace("__", "_")

    # Strip leading/trailing underscores
    key = key.strip("_")

    return key


def load_allowed_tags(path: str) -> dict[str, list[str]]:
    """Load allowed tags from the CSV file specified by TAG_LIST_CSV.

    The CSV should have columns like 'Mood', 'Tone', etc. Each cell contains
    semicolon-separated lists of allowed options. The returned dictionary
    maps a meta key (e.g. 'mood') to the list of allowed strings.

    Args:
        path: Path to the tags CSV file.

    Returns:
        Dictionary mapping meta keys to lists of allowed tag values.
    """
    allowed: dict[str, list[str]] = {}

    if not os.path.exists(path):
        logger.warning("Tag list file not found: %s", path)
        return allowed

    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            try:
                row = next(reader)
            except StopIteration:
                logger.warning("Tag list file is empty: %s", path)
                return allowed

            for col, val in row.items():
                # Skip empty column names (can happen with trailing commas in CSV)
                if not col:
                    continue
                key = build_meta_key(col)
                if not key:
                    continue
                if val:
                    allowed[key] = [v.strip() for v in val.split(";") if v.strip()]
                else:
                    allowed[key] = []

    except csv.Error as e:
        logger.error("Failed to parse tag list CSV %s: %s", path, e)
    except OSError as e:
        logger.error("Failed to read tag list file %s: %s", path, e)

    return allowed


# -----------------------------------------------------------------------------
# API Functions
# -----------------------------------------------------------------------------


def fetch_cover_image(title: str, author: str, api_key: str | None) -> str | None:
    """Fetch a cover image URL using the Brave search API.

    Queries the Brave image search endpoint for '{title} {author} book cover'
    and returns the first image URL that ends with .jpg, .jpeg or .png.

    Args:
        title: Book title.
        author: Book author.
        api_key: Brave API key. If None or empty, returns None.

    Returns:
        URL of a cover image, or None if not found.
    """
    if not api_key:
        return None

    query = f"{title} {author} book cover"
    headers = {
        "Accept": "application/json",
        "X-API-Key": api_key,
    }
    params = {"q": query, "count": 5}

    session = get_http_session()

    try:
        response = session.get(
            "https://api.search.brave.com/res/v1/images/search",
            headers=headers,
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching cover image for '%s'", title)
        return None
    except requests.exceptions.RequestException as e:
        logger.warning("Failed to fetch cover image for '%s': %s", title, e)
        return None
    except json.JSONDecodeError as e:
        logger.warning("Invalid JSON response from Brave API: %s", e)
        return None

    valid_extensions = (".jpg", ".jpeg", ".png")
    for img in data.get("results", []) or data.get("images", []):
        url = img.get("url") or img.get("thumbnail", {}).get("src")
        if not url:
            continue
        if url.lower().endswith(valid_extensions):
            return url

    return None


def generate_book_metadata(
    openai_api_key: str,
    title: str,
    author: str,
    description: str,
    tag_lists: dict[str, list[str]],
    model: str = DEFAULT_OPENAI_MODEL,
) -> dict[str, Any] | None:
    """Generate metadata for a book using the OpenAI Chat API.

    The prompt instructs the model to assign moods, tones, tropes and other
    metadata using only the allowed tags provided. The model returns a JSON
    object containing all fields required for Notion.

    Args:
        openai_api_key: OpenAI API key.
        title: Book title.
        author: Book author.
        description: Book description or review text.
        tag_lists: Dictionary of allowed tags by category.
        model: OpenAI model to use.

    Returns:
        Dictionary of generated metadata, or None on failure.
    """
    if not openai_api_key:
        logger.error("OpenAI API key not provided")
        return None

    # Build allowed tag strings for the prompt
    allowed_strings = []
    for key, tags in sorted(tag_lists.items()):
        if tags:
            allowed_strings.append(f"- {key}: {', '.join(tags)}")
    allowed_text = "\n".join(allowed_strings) if allowed_strings else "(No tag constraints)"

    prompt = METADATA_PROMPT_TEMPLATE.format(
        allowed_text=allowed_text,
        title=title,
        author=author,
        description=description or "(No description available)",
    )

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }
    messages = [
        {"role": "system", "content": "You are a JSON-only generator for book metadata."},
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
    }

    session = get_http_session()

    try:
        response = session.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        logger.error("Timeout calling OpenAI API for '%s'", title)
        return None
    except requests.exceptions.RequestException as e:
        logger.error("OpenAI API request failed for '%s': %s", title, e)
        return None
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON response from OpenAI API: %s", e)
        return None

    # Validate response structure
    if "choices" not in data or not data["choices"]:
        logger.error("Unexpected OpenAI response structure: %s", data)
        return None

    try:
        content = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        logger.error("Failed to extract content from OpenAI response: %s", e)
        return None

    # Remove markdown code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first and last lines if they're fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        metadata = json.loads(content)
        if not isinstance(metadata, dict):
            logger.error("OpenAI returned non-dict JSON: %s", type(metadata))
            return None
        return metadata
    except json.JSONDecodeError as e:
        logger.error("Failed to parse metadata JSON for '%s': %s", title, e)
        logger.debug("Raw content: %s", content[:500])
        return None


def build_notion_properties(
    row: dict[str, str], meta: dict[str, Any]
) -> dict[str, Any]:
    """Construct a Notion property dictionary from Goodreads row and metadata.

    The resulting dictionary can be passed directly to the Notion pages API.
    It respects the type of each property (text, select, multi-select,
    checkbox or number) according to the lists defined at the top of this
    module.

    Args:
        row: Dictionary from Goodreads CSV row.
        meta: Dictionary of LLM-generated metadata.

    Returns:
        Notion properties dictionary ready for page creation.
    """
    props: dict[str, Any] = {}

    # Title property (required)
    title = (row.get("Title") or "").strip()
    if title:
        props["Title"] = {
            "title": [
                {
                    "type": "text",
                    "text": {"content": title},
                }
            ]
        }

    # Text properties from Goodreads row or metadata
    for prop in TEXT_PROPS:
        if prop == "Author":
            val = (row.get("Author") or "").strip()
        else:
            key = build_meta_key(prop)
            val = meta.get(key)
            if isinstance(val, list):
                val = ", ".join([str(v).strip() for v in val if str(v).strip()])
            elif val is None:
                val = ""
            else:
                val = str(val)
            val = val.strip()

        if val:
            # Notion rich_text has a 2000 character limit per block
            props[prop] = {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {"content": val[:2000]},
                    }
                ]
            }

    # Select properties
    for prop in SELECT_PROPS:
        key = build_meta_key(prop)
        value = meta.get(key)
        selected = coerce_select(value)
        if selected:
            props[prop] = {"select": {"name": selected}}

    # Multi-select properties
    for prop in MULTI_SELECT_PROPS:
        key = build_meta_key(prop)
        value = meta.get(key)
        tags = coerce_multi(value)
        if tags:
            props[prop] = {"multi_select": [{"name": t} for t in tags]}

    # Checkbox properties
    for prop in CHECKBOX_PROPS:
        key = build_meta_key(prop)
        value = meta.get(key)
        if value is not None:
            # Accept truthy values as True, falsy as False
            checked = bool(value)
            props[prop] = {"checkbox": checked}

    # Number properties
    for prop in NUMBER_PROPS:
        key = build_meta_key(prop)
        value = meta.get(key)

        # Check Goodreads row for certain fields if not in metadata
        if value is None:
            goodreads_field_map = {
                "Page Count": ["Number of Pages", "Page Count"],
                "Publication Year": ["Original Publication Year", "Year Published", "Publication Year"],
                "Goodreads Rating": ["Average Rating", "My Rating"],
            }
            for gr_field in goodreads_field_map.get(prop, []):
                if gr_field in row and row[gr_field]:
                    try:
                        value = float(row[gr_field])
                        break
                    except ValueError:
                        continue

        if value is not None:
            try:
                num = float(value)
                props[prop] = {"number": num}
            except (ValueError, TypeError):
                logger.debug("Could not convert %s to number for %s", value, prop)

    # Date Added from Goodreads
    date_added_raw = row.get("Date Added", "")
    date_iso = clean_date_for_notion(date_added_raw)
    if date_iso:
        props["Date Added"] = {"date": {"start": date_iso}}

    # Date Read from Goodreads
    date_read_raw = row.get("Date Read", "")
    date_read_iso = clean_date_for_notion(date_read_raw)
    if date_read_iso:
        props["Date Read"] = {"date": {"start": date_read_iso}}

    # Cover Image from metadata (if generated by fetch_cover_image)
    cover_url = meta.get("cover_image_url")
    if cover_url:
        props["Cover Image"] = {
            "files": [
                {
                    "type": "external",
                    "name": title or "Cover",
                    "external": {"url": cover_url},
                }
            ]
        }

    return props


def create_notion_page(
    notion_token: str, database_id: str, properties: dict[str, Any]
) -> bool:
    """Create a new page in the specified Notion database.

    Args:
        notion_token: Notion integration token.
        database_id: Target database ID.
        properties: Notion properties dictionary.

    Returns:
        True on success, False otherwise.
    """
    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_API_VERSION,
    }
    payload = {
        "parent": {"database_id": database_id},
        "properties": properties,
    }

    session = get_http_session()

    try:
        response = session.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code in (200, 201):
            return True

        # Log error details
        try:
            err = response.json()
            logger.error(
                "Notion API error (status %d): %s",
                response.status_code,
                err.get("message", err),
            )
        except json.JSONDecodeError:
            logger.error(
                "Notion API error (status %d): %s",
                response.status_code,
                response.text[:200],
            )
        return False

    except requests.exceptions.Timeout:
        logger.error("Timeout creating Notion page")
        return False
    except requests.exceptions.RequestException as e:
        logger.error("Failed to create Notion page: %s", e)
        return False


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Import Goodreads export CSV to MoodReads Notion database with AI enrichment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  NOTION_TOKEN          Notion integration token (required)
  OPENAI_API_KEY        OpenAI API key (required)
  NOTION_DATABASE_ID    Target Notion database ID (required)
  BRAVE_API_KEY         Brave Search API key for cover images (optional)
  OPENAI_MODEL          OpenAI model to use (default: gpt-4o-mini)
  MOODREADS_TAG_LIST    Path to allowed tags CSV (default: moodreads_tags.csv)

Examples:
  python3 moodreads_importer.py goodreads_export.csv
  python3 moodreads_importer.py --dry-run goodreads_export.csv
  python3 moodreads_importer.py --skip 10 --limit 5 goodreads_export.csv
""",
    )
    parser.add_argument(
        "csv_path",
        help="Path to Goodreads export CSV file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process books without creating Notion pages",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of books to skip from the beginning",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of books to process (0 = no limit)",
    )
    parser.add_argument(
        "--no-cover",
        action="store_true",
        help="Skip cover image fetching",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        logger.error("File not found: %s", csv_path)
        return 1

    # Load environment variables
    notion_token = os.getenv("NOTION_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")
    database_id = os.getenv("NOTION_DATABASE_ID")
    brave_key = os.getenv("BRAVE_API_KEY") if not args.no_cover else None

    missing_vars = []
    if not notion_token:
        missing_vars.append("NOTION_TOKEN")
    if not openai_key:
        missing_vars.append("OPENAI_API_KEY")
    if not database_id:
        missing_vars.append("NOTION_DATABASE_ID")

    if missing_vars:
        logger.error(
            "Missing required environment variables: %s",
            ", ".join(missing_vars),
        )
        return 1

    # Load allowed tag lists
    tag_lists = load_allowed_tags(TAG_LIST_CSV)
    if tag_lists:
        logger.info("Loaded %d tag categories from %s", len(tag_lists), TAG_LIST_CSV)
    else:
        logger.warning("No tag constraints loaded; LLM will generate freely")

    # Read Goodreads CSV
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except csv.Error as e:
        logger.error("Failed to parse CSV %s: %s", csv_path, e)
        return 1
    except OSError as e:
        logger.error("Failed to read CSV %s: %s", csv_path, e)
        return 1

    logger.info("Loaded %d entries from %s", len(rows), csv_path)

    # Apply skip and limit
    if args.skip > 0:
        rows = rows[args.skip:]
        logger.info("Skipped first %d entries", args.skip)

    if args.limit > 0:
        rows = rows[: args.limit]
        logger.info("Limited to %d entries", args.limit)

    if args.dry_run:
        logger.info("DRY RUN: No pages will be created in Notion")

    created = 0
    skipped = 0
    failed = 0
    total = len(rows)

    for idx, row in enumerate(rows, start=1):
        title = (row.get("Title") or "").strip()
        author = (row.get("Author") or "").strip()

        if not title:
            logger.warning("[%d/%d] Skipping row with no title", idx, total)
            skipped += 1
            continue

        logger.info("[%d/%d] Processing: %s by %s", idx, total, title, author or "Unknown")

        # Prepare description: use My Review if present, otherwise empty
        description = (row.get("My Review") or "").strip()

        # Fetch cover image
        cover_url = None
        if brave_key:
            cover_url = fetch_cover_image(title, author, brave_key)
            if cover_url:
                logger.debug("  Cover image found")
            else:
                logger.debug("  No cover image found")
            time.sleep(BRAVE_SLEEP_SECONDS)

        # Generate metadata via LLM
        meta = generate_book_metadata(
            openai_key,
            title,
            author,
            description,
            tag_lists,
            model=DEFAULT_OPENAI_MODEL,
        )

        if meta is None:
            logger.warning("  Metadata generation failed; skipping book")
            failed += 1
            continue

        time.sleep(OPENAI_SLEEP_SECONDS)

        # Attach cover URL into metadata
        if cover_url:
            meta["cover_image_url"] = cover_url

        # Build Notion properties
        properties = build_notion_properties(row, meta)

        if args.dry_run:
            logger.info("  [DRY RUN] Would create page with %d properties", len(properties))
            created += 1
            continue

        # Create page
        ok = create_notion_page(notion_token, database_id, properties)
        if ok:
            created += 1
            logger.info("  Successfully created page")
        else:
            failed += 1
            logger.warning("  Failed to create page")

        time.sleep(NOTION_SLEEP_SECONDS)

    # Summary
    logger.info("=" * 50)
    logger.info("Import complete:")
    logger.info("  Created: %d", created)
    logger.info("  Skipped: %d", skipped)
    logger.info("  Failed:  %d", failed)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
