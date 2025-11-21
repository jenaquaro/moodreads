# MoodReads

A tool to import your Goodreads library into a Notion database with AI-generated mood and vibe metadata.

## Features

- Imports books from Goodreads CSV exports
- Generates rich metadata using OpenAI (mood, tone, tropes, pacing, etc.)
- Fetches cover images via Brave Search API
- Creates pages in your Notion database with proper property types

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set the following environment variables:

```bash
# Required
export NOTION_TOKEN="your-notion-integration-token"
export OPENAI_API_KEY="your-openai-api-key"
export NOTION_DATABASE_ID="your-notion-database-id"

# Optional
export BRAVE_API_KEY="your-brave-api-key"  # For cover images
export OPENAI_MODEL="gpt-4o-mini"          # Default model
export MOODREADS_TAG_LIST="moodreads_tags.csv"  # Allowed tags file
```

## Usage

```bash
# Basic usage
python3 moodreads_importer.py goodreads_export.csv

# Dry run (no pages created)
python3 moodreads_importer.py --dry-run goodreads_export.csv

# Process specific range
python3 moodreads_importer.py --skip 10 --limit 5 goodreads_export.csv

# Skip cover image fetching
python3 moodreads_importer.py --no-cover goodreads_export.csv

# Verbose logging
python3 moodreads_importer.py -v goodreads_export.csv
```

## Exporting from Goodreads

1. Go to [Goodreads My Books](https://www.goodreads.com/review/list)
2. Click "Import and export" in the left sidebar
3. Click "Export Library" to download your CSV

## License

MIT