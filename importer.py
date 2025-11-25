import os
import csv
import json
import time
import logging
import argparse
import datetime
import requests
import re
from typing import Dict, List, Optional, Any, Set
from urllib.parse import quote, urlparse
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self, args):
        self.notion_token = os.getenv("NOTION_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.notion_database_id = os.getenv("NOTION_DATABASE_ID")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.tag_list_file = os.getenv("MOODREADS_TAG_LIST", "moodreads_tags.csv")
        self.openai_sleep_seconds = float(os.getenv("OPENAI_SLEEP_SECONDS", "1.0"))
        self.notion_sleep_seconds = float(os.getenv("NOTION_SLEEP_SECONDS", "0.5"))
        self.max_llm_retries = int(os.getenv("MAX_LLM_RETRIES", "3"))

        # CLI args
        self.csv_path = args.csv_file
        self.check_dupes = args.check_dupes
        self.dry_run = args.dry_run
        self.limit = args.limit
        self.skip = args.skip
        self.no_cover = args.no_cover
        self.verbose = args.verbose

        if self.verbose:
            logger.setLevel(logging.DEBUG)

        # Validate required
        if not self.notion_token and not self.dry_run:
            logger.warning("NOTION_TOKEN not set. Operations requiring Notion will fail.")
        if not self.notion_database_id and not self.dry_run:
            logger.warning("NOTION_DATABASE_ID not set. Operations requiring Notion will fail.")
        if not self.openai_api_key and not self.dry_run:
            logger.warning("OPENAI_API_KEY not set. Metadata generation will fail.")

class RateLimiter:
    @staticmethod
    def sleep(seconds: float):
        time.sleep(seconds)

def validate_url(url: str) -> bool:
    """Validate that a URL is safe and well-formed."""
    if not url or not isinstance(url, str):
        return False

    try:
        parsed = urlparse(url)
        # Only allow http/https schemes
        if parsed.scheme not in ('http', 'https'):
            return False
        # Must have a valid netloc (domain)
        if not parsed.netloc:
            return False
        # Block localhost and internal IPs for security
        if any(blocked in parsed.netloc.lower() for blocked in ['localhost', '127.0.0.1', '0.0.0.0', '::1']):
            logger.warning(f"Blocked suspicious URL: {url}")
            return False
        return True
    except Exception as e:
        logger.warning(f"URL validation failed for '{url}': {e}")
        return False

class TagValidator:
    def __init__(self, config: Config):
        self.config = config
        self.allowed_tags, self.property_types = self._load_tags()

    def _load_tags(self) -> tuple:
        """Load allowed tags and property types from CSV."""
        allowed_tags = {}
        property_types = {}

        try:
            with open(self.config.tag_list_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    prop_name = row['property_name'].strip()
                    prop_type = row['property_type'].strip()
                    valid_opts_str = row.get('valid_options', '').strip()

                    property_types[prop_name] = prop_type

                    if valid_opts_str:
                        tags = [opt.strip() for opt in valid_opts_str.split('|')]
                        allowed_tags[prop_name] = set(tags)
                    else:
                        allowed_tags[prop_name] = set()

            logger.info(f"Loaded allowed tags for {len(allowed_tags)} categories")
            return allowed_tags, property_types
        except Exception as e:
            logger.error(f"Failed to load tags from {self.config.tag_list_file}: {e}")
            return {}, {}

    def validate(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        validated = {}
        # Fields where we allow the AI to generate new tags (Open Schema)
        open_fields = ["Themes", "Tropes General", "Tropes Romance", "Audiobook Characterization Style", "Audiobook Format", "Audiobook Vibe Tags"]

        for key, value in metadata.items():
            # Skip validation for free-text fields or if key not in allowed tags
            if key not in self.allowed_tags or key.lower() == 'vibes':
                validated[key] = value
                continue

            allowed = self.allowed_tags[key]
            allowed_lower = {t.lower(): t for t in allowed}

            if isinstance(value, list):
                valid_items = []
                for item in value:
                    # Handle comma-separated strings inside list items
                    sub_items = [s.strip() for s in item.split(',')] if isinstance(item, str) else [item]

                    for sub_item in sub_items:
                        if sub_item == "N/A":
                            valid_items.append(sub_item)
                        elif sub_item.lower() in allowed_lower:
                            valid_items.append(allowed_lower[sub_item.lower()])
                        elif key in open_fields:
                            valid_items.append(sub_item)

                if len(valid_items) < len(value):
                    logger.debug(f"Filtered invalid tags for {key}: {set(value) - set(valid_items)}")

                # Force N/A if empty and N/A is a valid option
                if not valid_items and "n/a" in allowed_lower:
                    valid_items = ["N/A"]

                validated[key] = valid_items
            elif isinstance(value, str):
                if value == "N/A":
                    validated[key] = value
                elif value.lower() in allowed_lower:
                    validated[key] = allowed_lower[value.lower()]
                else:
                    if "n/a" in allowed_lower:
                        validated[key] = "N/A"
                        logger.debug(f"Invalid value for {key}: {value}")
                    else:
                        validated[key] = value
            else:
                validated[key] = value

        return validated


class GoogleBooksClient:
    def __init__(self):
        self.base_url = "https://www.googleapis.com/books/v1/volumes"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def search(self, title: str, author: str) -> Optional[Dict[str, Any]]:
        """Search Google Books for metadata"""
        try:
            query = f"{title} {author}"
            params = {'q': query, 'maxResults': 1}
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if not data.get('items'):
                logger.warning(f"No results found on Google Books for: {title}")
                return None

            volume = data['items'][0]['volumeInfo']
            return {
                'description': volume.get('description', ''),
                'categories': ', '.join(volume.get('categories', [])),
                'average_rating': volume.get('averageRating'),
                'thumbnail': volume.get('imageLinks', {}).get('thumbnail'),
                'isbn': next((id['identifier'] for id in volume.get('industryIdentifiers', [])
                            if id['type'] in ['ISBN_13', 'ISBN_10']), None),
                'page_count': volume.get('pageCount'),
                'published_date': volume.get('publishedDate'),
                'publisher': volume.get('publisher')
            }
        except Exception as e:
            logger.error(f"Google Books API error for '{title}': {e}")
            return None

class AudiobookClient:
    def __init__(self):
        self.base_url = "https://www.googleapis.com/books/v1/volumes"
        self.audible_base_url = "https://www.audible.com/search"

    def scrape_audible(self, title: str, author: str) -> Optional[Dict[str, Any]]:
        """Scrape Audible for audiobook availability"""
        try:
            # Clean title for search (remove subtitle)
            search_title = title.split(':')[0].split('(')[0].strip()
            query = f"{search_title} {author}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
            }
            params = {"keywords": query}

            response = requests.get(self.audible_base_url, params=params, headers=headers, timeout=10)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find results container
            results = soup.find_all('li', class_='productListItem')

            for item in results:
                # Check title match
                item_title_tag = item.find('h3', class_='bc-heading')
                if not item_title_tag:
                    continue

                item_title = item_title_tag.get_text(strip=True)

                # Fuzzy match: check if search title is in result title
                if search_title.lower() in item_title.lower():
                    # Extract narrator
                    narrator = None
                    narrator_tag = item.find('li', class_='narratorLabel')
                    if narrator_tag:
                        narrator = narrator_tag.get_text(strip=True).replace("Narrated by:", "").strip()

                    # Detect format
                    fmt = "Standard Audible Narration"
                    text_content = item.get_text().lower()
                    if "graphicaudio" in text_content:
                        fmt = "GraphicAudio dramatization"
                    elif "dramatized" in text_content:
                        fmt = "Dramatized"
                    elif "full cast" in text_content or "multi-cast" in text_content:
                        fmt = "Multi-cast narration"

                    return {
                        "available": True,
                        "narrator": narrator,
                        "format": fmt
                    }

            return None

        except Exception as e:
            logger.debug(f"Audible scraping error: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def search(self, title: str, author: str) -> Optional[Dict[str, Any]]:
        """Search for audiobook metadata using Audible scrape then Google Books"""
        # 1. Try Audible Scraping (Most Accurate)
        audible_data = self.scrape_audible(title, author)
        if audible_data:
            return audible_data

        # 2. Fallback to Google Books API
        try:
            query = f"{title} {author} audiobook"
            params = {'q': query, 'maxResults': 5}
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if not data.get('items'):
                return {"available": False, "narrator": None, "format": None}

            # Search through results for audiobook format
            for item in data['items']:
                volume = item.get('volumeInfo', {})

                # Check if this is an audiobook
                categories = volume.get('categories', [])
                description = volume.get('description', '').lower()
                title_check = volume.get('title', '').lower()

                is_audiobook = (
                    'audiobook' in title_check or
                    'audio book' in description or
                    any('audiobook' in cat.lower() for cat in categories)
                )

                if is_audiobook:
                    # Try to extract narrator from description
                    narrator = None
                    desc = volume.get('description', '')
                    if 'narrated by' in desc.lower():
                        match = re.search(r'narrated by ([^.,\n]+)', desc, re.IGNORECASE)
                        if match:
                            narrator = match.group(1).strip()

                    return {
                        "available": True,
                        "narrator": narrator,
                        "format": "Standard Audible Narration"
                    }

            return {"available": False, "narrator": None, "format": None}

        except Exception as e:
            logger.debug(f"Audiobook API error for '{title}': {e}")
            return {"available": None, "narrator": None, "format": None}

class OpenAIClient:
    def __init__(self, config: Config, validator: TagValidator):
        self.config = config
        self.validator = validator
        self.client = OpenAI(api_key=self.config.openai_api_key)

    def _construct_prompt(self, book_info: Dict[str, Any]) -> str:
        tags_context = "\n".join([f"- {k}: {', '.join(list(v))}" for k, v in self.validator.allowed_tags.items()])

        return f"""
You are a metadata generator for a book database. Your goal is to generate comprehensive metadata for the given book.

=== INSTRUCTIONS ===
1. **MANDATORY FIELDS**: You MUST provide a value for **EVERY** key listed in the 'ALLOWED TAGS' section below. Do NOT skip any fields.
2. **Counts**:
    - **Tone**: Select **EXACTLY 3** tags.
    - **Mood**: Select **AT LEAST 2** tags.
    - **Multi-Select**: Select **UP TO 5** most relevant tags for others.

3. **CRITICAL - SUBJECTIVE FIELDS (NEVER N/A)**:
    - **Tone**: MUST select EXACTLY 3 tags. NEVER use N/A. These define the authorial voice.
        * **WARNING**: Do NOT default to "imaginative", "exploratory", "defiant", "political", or "rebellious" unless they are the ABSOLUTE BEST fit.
        * **VARIETY**: Use the full range of tags (e.g. "witty", "somber", "urgent", "warm", "cynical", "romantic").
    - **Aesthetic**: MUST select at least 1 tag. NEVER use N/A. Choose based on setting, mood, or genre.
    - **When/Where to Read**: MUST select at least 1 tag. NEVER use N/A. Think about ideal reading contexts.
    - For these fields, use creativity and inference. Even if you don't know the book, infer based on:
        * Genre (Romance → "Cozy Evening", "Bedtime")
        * Setting (Historical → aesthetic tags like "gothic" or "vintage")
        * Tone keywords in description

4. **Vibes**: Write a short sentence description that captures the overall vibe/feeling. Think of it like a quick atmospheric snapshot.
    - Example: "Dark academia meets forbidden romance in a crumbling manor"
    - Example: "Cozy small-town mystery with autumnal vibes"
    - Example: "Epic space battles with found family feels"

5. **Audiobook**: ONLY use the provided audiobook data. Do NOT guess.
    - If audiobook_available is True: Populate 'Audiobook Format', 'Audiobook Characterization Style', and 'Audiobook Vibe Tags'.
    - If audiobook_available is False or None: Set 'Audiobook Available?' to False and audiobook fields to N/A.

6. **Genre Specifics**:
    - **Sci-Fi**: Only populate 'Sci-Fi Sub-genre', 'Tech Level', 'Hard vs Soft Sci-Fi' if the Genre is actually Science Fiction. Otherwise use "N/A".
    - **Themes**: Avoid 'Identity and self-discovery' unless it is the CENTRAL theme. You may add new themes if they are relevant.
    - **Writing Style**: Select the most accurate tags from the list. Do not default to 'Lyrical' unless the prose is truly poetic.
    - **Emotional Impact**: Distinguish between 'Emotional' (moving drama) and 'Devastating' (tragic/heartbreaking).

7. **Learning Outcome**: 'Just for Fun' is a valid and common option.
8. **Strive for Population**: Try your best to populate EVERY field with a real value. Use "N/A" ONLY if the field is completely irrelevant (e.g. 'Magic System' for Non-fiction).
    - **Unknown Books**: If you cannot find specific information about the book, you MUST **infer** plausible metadata based on the Title, Author, Genre, and Cover. Do NOT default to N/A just because you are unsure. Make an educated guess.

=== ALLOWED TAGS (REQUIRED KEYS) ===
{tags_context}

=== BOOK INFO ===
Title: {book_info.get('title')}
Author: {book_info.get('author')}
Description: {book_info.get('description')}
Categories: {book_info.get('categories')}
My Review: {book_info.get('my_review')}
Audiobook Available: {book_info.get('audiobook_available', 'Unknown')}
Audiobook Narrator: {book_info.get('audiobook_narrator', 'Unknown')}

Return valid JSON only. Keys must match the 'ALLOWED TAGS' keys exactly.
"""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _generate_single_pass(self, book_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single metadata pass."""
        prompt = self._construct_prompt(book_info)

        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2500,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()

            # Parse JSON (response_format ensures it's valid JSON, but still handle edge cases)
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            metadata = json.loads(content)
            return metadata

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.debug(f"Raw content: {content[:500]}")
            raise
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _refine_metadata(self, book_info: Dict[str, Any], candidate1: Dict[str, Any], candidate2: Dict[str, Any], candidate3: Dict[str, Any]) -> Dict[str, Any]:
        tags_context = "\n".join([f"- {k}: {', '.join(list(v))}" for k, v in self.validator.allowed_tags.items()])

        prompt = f"""
You are a Quality Control Expert for a book database. Your job is to review THREE sets of AI-generated metadata for a book and create a FINAL, PERFECT set.

=== BOOK INFO ===
Title: {book_info.get('title')}
Author: {book_info.get('author')}
Description: {book_info.get('description')}
Categories: {book_info.get('categories')}

=== CANDIDATE METADATA 1 ===
{json.dumps(candidate1, indent=2)}

=== CANDIDATE METADATA 2 ===
{json.dumps(candidate2, indent=2)}

=== CANDIDATE METADATA 3 ===
{json.dumps(candidate3, indent=2)}

=== INSTRUCTIONS ===
1. **Aggressive Population**: Your goal is to populate EVERY field.
    - If **ANY** of the 3 candidates has a valid value for a field, **USE IT**.
    - Do NOT use "N/A" unless ALL 3 candidates agree that it is N/A or if it is strictly impossible (e.g. Magic System in Non-fiction).
    - For subjective fields (Aesthetic, Vibes, When/Where to Read), pick the most creative and descriptive option from the candidates.

2. **CRITICAL - HARD REJECTIONS (NEVER ACCEPT N/A)**:
    - **Tone**: If ANY candidate has valid Tone tags, USE THEM. If all are N/A (should NEVER happen), YOU MUST invent 3 plausible tags based on genre/description.
        * **DIVERSIFY**: If the candidates only offer "imaginative/exploratory/defiant", YOU MUST swap at least one for a more specific tag (e.g. "dark", "funny", "tense").
    - **Aesthetic**: If ANY candidate has valid Aesthetic tags, USE THEM. If all are N/A, YOU MUST invent at least 1 tag.
    - **When/Where to Read**: If ANY candidate has valid tags, USE THEM. If all are N/A, YOU MUST invent at least 1 tag based on genre/mood.
    - **Mood**: Must have at least 2 tags. NEVER N/A.
    - **Vibes**: Must be a descriptive sentence. NEVER N/A or empty.

3. **Fix Hallucinations**:
    - Ensure "Lyrical" is NOT used for gritty/action-heavy books.
    - Ensure "Funny" isn't used for dark tragedies.
    - Ensure "Sci-Fi" fields are N/A if the book is not Sci-Fi.

4. **Completeness**: Ensure ALL mandatory fields (Themes, Tropes, Nonfiction Type, etc.) are populated.
5. **Vibes**: Ensure 'Vibes' is a descriptive sentence (e.g., "Atmospheric Gothic horror..."), NOT just a list of keywords.
6. **Counts**: Ensure Tone has EXACTLY 3 tags and Mood has AT LEAST 2.
7. **Audiobook**: Use the provided audiobook data. Do NOT override with guesses.
8. **Mandatory Fields**: You MUST provide a value for EVERY key in the 'ALLOWED TAGS' list below.

=== ALLOWED TAGS ===
{tags_context}

Return valid JSON only.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=2500,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()

            # Parse JSON
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            result = json.loads(content)
            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in QC: {e}")
            logger.debug(f"Raw content: {content[:500]}")
            logger.warning("Falling back to candidate 1 due to parsing error")
            return candidate1
        except Exception as e:
            logger.error(f"OpenAI QC error: {e}")
            return candidate1

    def generate_metadata_with_qc(self, book_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generates metadata using a 3-pass Generate & Refine workflow."""
        logger.info("  Generating metadata (Attempt 1)...")
        c1 = self._generate_single_pass(book_info)

        logger.info("  Generating metadata (Attempt 2)...")
        c2 = self._generate_single_pass(book_info)

        logger.info("  Generating metadata (Attempt 3)...")
        c3 = self._generate_single_pass(book_info)

        logger.info("  Refining metadata (QC)...")
        final = self._refine_metadata(book_info, c1, c2, c3)

        if not final.get('Genre'):
            logger.warning("Generated metadata missing 'Genre' field or it was filtered out.")

        return final


class NotionClient:
    def __init__(self, config: Config, validator: TagValidator):
        self.config = config
        self.validator = validator
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.config.notion_token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
        self.available_properties = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _get_database_schema(self) -> Set[str]:
        if self.config.dry_run:
            return set()

        try:
            response = requests.get(
                f"{self.base_url}/databases/{self.config.notion_database_id}",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return set(data.get("properties", {}).keys())
        except Exception as e:
            logger.error(f"Failed to fetch database schema: {e}")
            return set()

    def get_existing_titles(self) -> Dict[str, str]:
        if self.config.dry_run:
            return {}

        self.available_properties = self._get_database_schema()
        if self.available_properties:
            logger.info(f"Fetched database schema. Available properties: {', '.join(sorted(list(self.available_properties)))}")

        titles = {}
        has_more = True
        start_cursor = None

        logger.info("Querying Notion for existing books...")

        while has_more:
            payload = {"page_size": 100}
            if start_cursor:
                payload["start_cursor"] = start_cursor

            try:
                response = requests.post(
                    f"{self.base_url}/databases/{self.config.notion_database_id}/query",
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                for page in data["results"]:
                    try:
                        props = page["properties"]
                        title_list = props.get("Title", {}).get("title", [])
                        if title_list:
                            title = title_list[0]["plain_text"]
                            titles[title] = page["id"]
                    except Exception as e:
                        logger.warning(f"Error parsing page {page.get('id', 'unknown')}: {e}")

                has_more = data.get("has_more", False)
                start_cursor = data.get("next_cursor")

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to query Notion database: {e}")
                break

        logger.info(f"Found {len(titles)} existing books in Notion.")
        return titles

    def _format_properties(self, book_info: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        props = {}

        # Helper functions
        def set_text(key, value):
            if value and isinstance(value, list):
                value = ', '.join(str(v) for v in value)
            if value:
                props[key] = {"rich_text": [{"text": {"content": str(value)[:2000]}}]}

        def set_select(key, value):
            if value:
                # Handle list values
                if isinstance(value, list):
                    value = value[0] if value else None
                    if not value:
                        return

                val_str = str(value)
                val_str = val_str.replace("[", "").replace("]", "").replace("'", "").replace('"', "")

                # Notion Select cannot contain commas
                if "," in val_str:
                    val_str = val_str.split(",")[0].strip()

                # Notion Select limit is 100 characters
                val_str = val_str[:100]

                if val_str:
                    props[key] = {"select": {"name": val_str}}

        def set_multi_select(key, value):
            if value and isinstance(value, list):
                # Ensure no commas and respect length limit
                items = []
                for v in value:
                    s = str(v).replace(",", "")[:100]
                    if s:
                        items.append({"name": s})
                props[key] = {"multi_select": items}
            elif value and isinstance(value, str):
                # Split by comma if string passed to multi-select
                items = [{"name": s.strip()[:100]} for s in value.split(",") if s.strip()]
                props[key] = {"multi_select": items}

        def set_number(key, value):
            if value is not None:
                try:
                    props[key] = {"number": float(value)}
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not convert '{value}' to number for {key}: {e}")

        def set_checkbox(key, value):
            if value is not None:
                if isinstance(value, str):
                    v = value.lower().strip()
                    val_bool = v in ["true", "yes", "y", "1", "checked"]
                else:
                    val_bool = bool(value)
                props[key] = {"checkbox": val_bool}

        def set_date(key, value):
            if value:
                try:
                    for fmt in ["%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%d", "%B %d, %Y"]:
                        try:
                            dt = datetime.datetime.strptime(value, fmt)
                            iso_date = dt.strftime("%Y-%m-%d")
                            props[key] = {"date": {"start": iso_date}}
                            return
                        except ValueError:
                            continue
                    logger.warning(f"Could not parse date '{value}' for {key}. Skipping.")
                except Exception as e:
                    logger.warning(f"Error processing date '{value}': {e}")

        def set_url(key, value):
            if value and validate_url(value):
                props[key] = {"url": value}
            elif value:
                logger.warning(f"Invalid URL for {key}: {value}")

        def set_file(key, value):
            if value and validate_url(value):
                props[key] = {
                    "files": [
                        {
                            "type": "external",
                            "name": "Cover",
                            "external": {"url": value}
                        }
                    ]
                }
            elif value:
                logger.warning(f"Invalid file URL for {key}: {value}")

        # Combine source data and LLM metadata
        combined_data = metadata.copy()

        # Extract series info from title
        series_name = None
        series_number = None
        title = book_info.get("title", "")
        match = re.search(r'\((.+?),\s*#(\d+)\)', title)
        if match:
            series_name = match.group(1)
            series_number = match.group(2)

        # Map Read Status
        read_status = book_info.get("exclusive_shelf")
        if read_status == "read":
            read_status = "Already Read"
        elif read_status == "currently-reading":
            read_status = "Currently Reading"
        elif read_status == "to-read":
            read_status = "Want to Read"

        source_map = {
            "Title": title,
            "Author": book_info.get("author"),
            "Page Count": book_info.get("page_count"),
            "Publication Year": book_info.get("year_published"),
            "Goodreads Rating": book_info.get("average_rating"),
            "Date Added": book_info.get("date_added"),
            "Personal Notes": book_info.get("my_review"),
            "Series Status":  series_name,
            "Series - Book Number": series_number,
            "Read Status": read_status,
            "Source/Where I Found It": "Goodreads",
            "Cover Image": book_info.get("cover_image")
        }

        for key, value in source_map.items():
            if value is not None:
                combined_data[key] = value

        # Schema-driven iteration to ensure ALL fields are populated
        if not self.available_properties:
            self.available_properties = self._get_database_schema()

        for key, prop_type_from_validator in self.validator.property_types.items():
            if key not in self.available_properties:
                continue

            prop_type = prop_type_from_validator.lower()
            value = combined_data.get(key)

            # Handle missing values with defaults
            if value is None or value == "":
                if prop_type == "checkbox":
                    value = False
                elif prop_type in ["select", "multi_select", "text"]:
                    value = "N/A"
                else:
                    continue

            # Handle special fields first
            if key == "Title":
                props["Title"] = {"title": [{"text": {"content": str(value)[:2000]}}]}
            elif key == "Cover Image":
                set_file(key, value)
            elif prop_type == "text" or prop_type == "rich_text":
                set_text(key, value)
            elif prop_type == "select":
                set_select(key, value)
            elif prop_type == "multi_select":
                set_multi_select(key, value)
            elif prop_type == "number":
                set_number(key, value)
            elif prop_type == "checkbox":
                set_checkbox(key, value)
            elif prop_type == "date":
                set_date(key, value)
            elif prop_type == "url":
                set_url(key, value)
            elif prop_type == "files":
                set_file(key, value)

        return props

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def create_page(self, page_data: Dict[str, Any], book_info: Dict[str, Any]):
        if self.config.dry_run:
            logger.info(f"Dry run: Would create page for {book_info.get('title')}")
            return

        properties = self._format_properties(book_info, page_data)

        payload = {
            "parent": {"database_id": self.config.notion_database_id},
            "properties": properties
        }

        logger.debug(f"Creating page with properties: {json.dumps(properties, indent=2)}")

        try:
            response = requests.post(f"{self.base_url}/pages", headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            logger.info(f"Successfully created page for {book_info.get('title')}")
        except requests.exceptions.HTTPError as e:
            logger.error(f"Failed to create page: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response Status: {e.response.status_code}")
                logger.error(f"Response Body: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Failed to create page: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def update_page(self, page_id: str, page_data: Dict[str, Any], book_info: Dict[str, Any]):
        if self.config.dry_run:
            logger.info(f"Dry run: Would update page {page_id} for {book_info.get('title')}")
            return

        properties = self._format_properties(book_info, page_data)
        if "Title" in properties:
            del properties["Title"]

        payload = {"properties": properties}

        try:
            response = requests.patch(f"{self.base_url}/pages/{page_id}", headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            logger.info(f"Successfully updated page for {book_info.get('title')}")
        except requests.exceptions.HTTPError as e:
            logger.error(f"Failed to update page: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response Status: {e.response.status_code}")
                logger.error(f"Response Body: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Failed to update page: {e}")
            raise


class Processor:
    def __init__(self, config: Config):
        self.config = config
        self.google_books = GoogleBooksClient()
        self.audiobook_client = AudiobookClient()
        self.tag_validator = TagValidator(config)
        self.openai_client = OpenAIClient(config, self.tag_validator)
        self.notion_client = NotionClient(config, self.tag_validator)

    def _get_cover_image(self, isbn: str, google_thumbnail: str) -> Optional[str]:
        if self.config.no_cover:
            return None

        if isbn:
            ol_url = f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
            try:
                r = requests.head(ol_url, timeout=5)
                if r.status_code == 200 and int(r.headers.get("Content-Length", 0)) > 1000:
                    # Validate the URL before returning
                    if validate_url(ol_url):
                        return ol_url
            except Exception as e:
                logger.debug(f"OpenLibrary cover check failed: {e}")

        if google_thumbnail and validate_url(google_thumbnail):
            return google_thumbnail.replace("&edge=curl", "&edge=none")

        return None

    def run(self):
        logger.info(f"Starting import from {self.config.csv_path}")

        try:
            with open(self.config.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            return

        total = len(rows)
        skipped = 0
        processed = 0
        failed = 0

        existing_titles = {}
        if self.config.check_dupes:
            existing_titles = self.notion_client.get_existing_titles()

        start_idx = self.config.skip if self.config.skip else 0
        end_idx = start_idx + self.config.limit if self.config.limit else total
        rows_to_process = rows[start_idx:end_idx]

        logger.info(f"Processing {len(rows_to_process)} books (skipped {start_idx})")

        for i, row in enumerate(rows_to_process):
            idx = start_idx + i + 1
            title = row.get('Title', 'Unknown')
            author = row.get('Author', 'Unknown')
            isbn = row.get('ISBN', '').replace('=', '').replace('"', '')

            logger.info(f"[{idx}/{total}] Processing: {title} by {author}")

            try:
                if self.config.check_dupes and title in existing_titles:
                    logger.info(f"  Book already exists: {title}")
                    page_id = existing_titles[title]
                else:
                    page_id = None

                gb_data = self.google_books.search(title, author) or {}
                if gb_data:
                    logger.info("  Successfully scraped metadata from Google Books")

                # Search for audiobook data
                audiobook_data = self.audiobook_client.search(title, author)
                if audiobook_data and audiobook_data.get('available'):
                    logger.info(f"  Audiobook found (Narrator: {audiobook_data.get('narrator', 'Unknown')})")
                elif audiobook_data and audiobook_data.get('available') == False:
                    logger.info("  No audiobook found")

                book_info = {
                    "title": title,
                    "author": author,
                    "isbn": isbn,
                    "my_rating": row.get('My Rating'),
                    "average_rating": gb_data.get('average_rating') or row.get('Average Rating'),
                    "page_count": gb_data.get('page_count') or row.get('Number of Pages'),
                    "year_published": gb_data.get('published_date', '')[:4] if gb_data.get('published_date') else row.get('Year Published'),
                    "date_added": row.get('Date Added'),
                    "exclusive_shelf": row.get('Exclusive Shelf'),
                    "description": gb_data.get('description', ''),
                    "categories": gb_data.get('categories', ''),
                    "my_review": row.get('My Review', ''),
                    "publisher": gb_data.get('publisher'),
                    "audiobook_available": audiobook_data.get('available') if audiobook_data else None,
                    "audiobook_narrator": audiobook_data.get('narrator') if audiobook_data else None,
                    "audiobook_format": audiobook_data.get('format') if audiobook_data else None,
                }

                book_info["cover_image"] = self._get_cover_image(isbn, gb_data.get('thumbnail'))
                if book_info["cover_image"]:
                    logger.info("  Cover image found")

                try:
                    metadata = self.openai_client.generate_metadata_with_qc(book_info)
                    logger.info("  Metadata generated successfully")
                except Exception as e:
                    logger.error(f"  Failed to generate metadata: {e}")
                    failed += 1
                    continue

                validated_metadata = self.tag_validator.validate(metadata)

                if page_id:
                    self.notion_client.update_page(page_id, validated_metadata, book_info)
                else:
                    self.notion_client.create_page(validated_metadata, book_info)

                processed += 1

            except Exception as e:
                logger.error(f"  Failed to process book: {e}")
                # Unwrap RetryError to show actual cause if possible
                if "RetryError" in str(type(e)):
                    try:
                        logger.error(f"  Caused by: {e.last_attempt.exception()}")
                    except Exception:
                        pass
                failed += 1

            RateLimiter.sleep(self.config.openai_sleep_seconds)

        logger.info("Import complete")
        logger.info(f"Processed: {processed}, Skipped: {skipped}, Failed: {failed}")

def main():
    parser = argparse.ArgumentParser(description="MoodReads Book Metadata Importer")
    parser.add_argument("csv_file", help="Path to Goodreads export CSV")
    parser.add_argument("--check-dupes", action="store_true", help="Skip books already in database")
    parser.add_argument("--dry-run", action="store_true", help="Process but don't create Notion pages")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N books")
    parser.add_argument("--limit", type=int, help="Process max N books")
    parser.add_argument("--no-cover", action="store_true", help="Skip cover image fetching")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")

    args = parser.parse_args()
    config = Config(args)
    config.csv_path = args.csv_file
    processor = Processor(config)
    processor.run()

if __name__ == "__main__":
    main()
