"""
scripts/refresh_rebrickable_data.py
=====================================
Download the latest Rebrickable CSV files and build a local SQLite database.

Run this script once before first use, then periodically to keep data fresh.
Rebrickable updates their database daily. No API key required for CSV downloads.

Usage:
    python scripts/refresh_rebrickable_data.py
    python scripts/refresh_rebrickable_data.py --force  # re-download even if recent
"""

import os, sys, csv, gzip, sqlite3, requests, argparse
from datetime import datetime, timedelta
from pathlib import Path


# CSV files to download from Rebrickable
REBRICKABLE_CSV_URLS = {
    "colors":           "https://rebrickable.com/media/downloads/colors.csv.gz",
    "parts":            "https://rebrickable.com/media/downloads/parts.csv.gz",
    "part_categories":  "https://rebrickable.com/media/downloads/part_categories.csv.gz",
    "elements":         "https://rebrickable.com/media/downloads/elements.csv.gz",
    "sets":             "https://rebrickable.com/media/downloads/sets.csv.gz",
    "inventory_parts":  "https://rebrickable.com/media/downloads/inventory_parts.csv.gz",
}

DATA_DIR = Path("data/rebrickable")
DB_PATH = Path("data/lego.db")


def download_csv(name, url, force=False):
    """Download and decompress a Rebrickable CSV file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = DATA_DIR / f"{name}.csv.gz"
    csv_path = DATA_DIR / f"{name}.csv"

    if csv_path.exists() and not force:
        age = datetime.now() - datetime.fromtimestamp(csv_path.stat().st_mtime)
        if age < timedelta(days=1):
            print(f"  {name}.csv is fresh ({age.seconds//3600}h old), skipping")
            return csv_path

    print(f"  Downloading {name}.csv...", end=" ", flush=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    gz_path.write_bytes(resp.content)

    with gzip.open(gz_path, "rb") as f_in:
        csv_path.write_bytes(f_in.read())
    gz_path.unlink()

    rows = sum(1 for _ in open(csv_path)) - 1
    print(f"{rows:,} rows")
    return csv_path


def build_database(force=False):
    """Build SQLite database from downloaded CSVs."""
    if DB_PATH.exists() and not force:
        age = datetime.now() - datetime.fromtimestamp(DB_PATH.stat().st_mtime)
        if age < timedelta(hours=23):
            print(f"  lego.db is fresh ({age.seconds//3600}h old), skipping rebuild")
            return

    print("  Building lego.db...")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Colors table
    c.execute("DROP TABLE IF EXISTS colors")
    c.execute("""CREATE TABLE colors (
        id INTEGER PRIMARY KEY, name TEXT, rgb TEXT, is_trans TEXT
    )""")
    with open(DATA_DIR/"colors.csv") as f:
        reader = csv.DictReader(f)
        c.executemany("INSERT OR REPLACE INTO colors VALUES (?,?,?,?)",
                      [(r["id"],r["name"],r["rgb"],r["is_trans"]) for r in reader])

    # Parts table
    c.execute("DROP TABLE IF EXISTS parts")
    c.execute("""CREATE TABLE parts (
        part_num TEXT PRIMARY KEY, name TEXT, part_cat_id INTEGER
    )""")
    with open(DATA_DIR/"parts.csv") as f:
        reader = csv.DictReader(f)
        c.executemany("INSERT OR REPLACE INTO parts VALUES (?,?,?)",
                      [(r["part_num"],r["name"],r["part_cat_id"]) for r in reader])

    # Elements table (part+color combos that actually exist)
    c.execute("DROP TABLE IF EXISTS elements")
    c.execute("""CREATE TABLE elements (
        element_id TEXT PRIMARY KEY, part_num TEXT, color_id INTEGER
    )""")
    with open(DATA_DIR/"elements.csv") as f:
        reader = csv.DictReader(f)
        c.executemany("INSERT OR REPLACE INTO elements VALUES (?,?,?)",
                      [(r["element_id"],r["part_num"],r["color_id"]) for r in reader])

    # Index for fast lookups
    c.execute("CREATE INDEX IF NOT EXISTS idx_elements_part ON elements(part_num)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_elements_color ON elements(color_id)")

    conn.commit()
    conn.close()

    # Stats
    conn = sqlite3.connect(DB_PATH)
    parts_count = conn.execute("SELECT COUNT(*) FROM parts").fetchone()[0]
    colors_count = conn.execute("SELECT COUNT(*) FROM colors").fetchone()[0]
    elements_count = conn.execute("SELECT COUNT(*) FROM elements").fetchone()[0]
    conn.close()
    print(f"  Database built: {parts_count:,} parts, {colors_count} colors, {elements_count:,} elements")


def check_part_color_exists(part_num, color_id):
    """Check if a specific part+color combination exists in the LEGO catalog."""
    conn = sqlite3.connect(DB_PATH)
    result = conn.execute(
        "SELECT COUNT(*) FROM elements WHERE part_num=? AND color_id=?",
        (str(part_num), int(color_id))
    ).fetchone()[0]
    conn.close()
    return result > 0


def get_part_info(part_num):
    """Get part name and category from the database."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT name, part_cat_id FROM parts WHERE part_num=?", (str(part_num),)
    ).fetchone()
    conn.close()
    return {"name": row[0], "cat_id": row[1]} if row else None


def main():
    parser = argparse.ArgumentParser(description="Refresh Rebrickable data")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if data is fresh")
    args = parser.parse_args()

    print("Refreshing Rebrickable LEGO database...")
    print(f"Data directory: {DATA_DIR.absolute()}")
    print()

    print("Downloading CSVs:")
    for name, url in REBRICKABLE_CSV_URLS.items():
        download_csv(name, url, force=args.force)

    print()
    print("Building SQLite database:")
    build_database(force=args.force)

    print()
    print("Done! Your local LEGO parts database is ready.")
    print(f"DB path: {DB_PATH.absolute()}")


if __name__ == "__main__":
    main()
