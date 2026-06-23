#!/usr/bin/env python3

import argparse
import csv
import gzip
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BRICKSET_API = "https://brickset.com/api/v3.asmx"
REBRICKABLE_SETS_CSV_GZ = "https://cdn.rebrickable.com/media/downloads/sets.csv.gz"
USER_AGENT = "lego-instructions-downloader/4.0"


def load_dotenv(path=".env", override=False):
    env_path = Path(path)

    if not env_path.exists() or not env_path.is_file():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key:
            if override or key not in os.environ:
                os.environ[key] = value


def safe_name(text):
    text = str(text or "").strip()
    text = re.sub(r"[^\w.\- ()]+", "_", text)
    text = re.sub(r"\s+", " ", text)
    return text[:180] or "unknown"


def normalize_set_num(set_num):
    set_num = str(set_num or "").strip()
    if not set_num:
        return ""
    if "-" not in set_num:
        return f"{set_num}-1"
    return set_num


def clean_url(url):
    url = str(url).strip()
    parts = urllib.parse.urlsplit(url)
    path = urllib.parse.quote(parts.path, safe="/")
    query = urllib.parse.quote(parts.query, safe="=&%:+/?")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, query, parts.fragment))


def fetch_bytes(url, timeout=300):
    url = clean_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_json(url):
    data = fetch_bytes(url, timeout=120)
    return json.loads(data.decode("utf-8"))


def brickset_get_instructions(api_key, set_num):
    set_num = normalize_set_num(set_num)

    query = urllib.parse.urlencode({
        "apiKey": api_key,
        "setNumber": set_num,
    })

    url = f"{BRICKSET_API}/getInstructions2?{query}"
    data = fetch_json(url)

    if data.get("status") != "success":
        raise RuntimeError(data)

    return data.get("instructions", [])


def instruction_url(inst):
    return (
        inst.get("URL")
        or inst.get("url")
        or inst.get("Url")
        or inst.get("downloadURL")
        or inst.get("DownloadURL")
    )


def instruction_filename(set_num, inst, index):
    url = instruction_url(inst) or ""
    parsed_name = Path(urllib.parse.urlparse(url).path).name

    if parsed_name.lower().endswith(".pdf"):
        return safe_name(parsed_name)

    desc = (
        inst.get("description")
        or inst.get("Description")
        or inst.get("name")
        or inst.get("Name")
        or f"book_{index}"
    )

    return safe_name(f"{set_num}_{index:02d}_{desc}.pdf")


def should_keep_instruction(url, filename):
    text = f"{url} {filename}".lower()

    # Skip non-English language-specific digital booklets.
    if "digital booklet" in text:
        keep_markers = [
            " - en",
            "_en",
            "-en.",
            " english",
            " - us",
            "_us",
            " - uk",
            "_uk",
        ]
        return any(marker in text for marker in keep_markers)

    # Keep normal numbered LEGO instruction PDFs.
    return True


def download_file(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Skipped existing: {dest}")
        return "skipped"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    print(f"Downloading: {url}")
    data = fetch_bytes(url, timeout=300)
    tmp.write_bytes(data)
    tmp.rename(dest)

    print(f"Saved: {dest}")
    return "downloaded"


def download_set(api_key, set_num, output, download_delay):
    set_num = normalize_set_num(set_num)
    print(f"Checking {set_num}")

    instructions = brickset_get_instructions(api_key, set_num)
    print(f"Found {len(instructions)} instruction record(s).")

    set_dir = Path(output) / safe_name(set_num)

    downloaded = 0
    skipped = 0
    failed = 0

    for i, inst in enumerate(instructions, start=1):
        url = instruction_url(inst)

        if not url:
            print(f"No URL in record: {inst}")
            failed += 1
            continue

        filename = instruction_filename(set_num, inst, i)

        if not should_keep_instruction(url, filename):
            print(f"Skipped non-English/language extra: {filename}")
            continue

        dest = set_dir / filename

        try:
            result = download_file(url, dest)
            if result == "downloaded":
                downloaded += 1
            elif result == "skipped":
                skipped += 1
        except Exception as e:
            failed += 1
            print(f"FAILED: {url} -> {e}", file=sys.stderr)

        time.sleep(download_delay)

    return {
        "found": len(instructions),
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }


def download_rebrickable_set_catalog(cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "sets.csv.gz"

    if path.exists() and path.stat().st_size > 0:
        print(f"Using cached set catalog: {path}")
        return path

    print("Downloading set catalog...")
    data = fetch_bytes(REBRICKABLE_SETS_CSV_GZ, timeout=180)
    path.write_bytes(data)
    print(f"Saved set catalog: {path}")
    return path


def load_all_set_nums(cache_dir, year_min=None, year_max=None):
    path = download_rebrickable_set_catalog(cache_dir)
    set_nums = []

    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            set_num = normalize_set_num(row.get("set_num", ""))
            year_text = row.get("year", "")

            if not set_num:
                continue

            try:
                year = int(year_text)
            except Exception:
                year = None

            if year_min is not None and year is not None and year < year_min:
                continue

            if year_max is not None and year is not None and year > year_max:
                continue

            set_nums.append(set_num)

    return set_nums


def load_sets_file(path):
    nums = []

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        nums.append(normalize_set_num(line))

    return nums


def load_progress(path):
    path = Path(path)

    if not path.exists():
        return {"done": []}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"done": []}


def save_progress(path, progress):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def main():
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Download LEGO instruction PDFs using Brickset API.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--set", action="append", help="Set number. Example: --set 75192")
    group.add_argument("--sets-file", help="Text file with one set number per line.")
    group.add_argument("--all", action="store_true", help="Check all sets from the set catalog.")

    parser.add_argument("--output", default="./instructions")
    parser.add_argument("--cache", default="./.lego-cache")
    parser.add_argument("--api-key", default=os.getenv("BRICKSET_API_KEY"))
    parser.add_argument("--daily-limit", type=int, default=10, help="Max set numbers to check per run.")
    parser.add_argument("--set-delay", type=float, default=20.0, help="Seconds to wait between sets.")
    parser.add_argument("--download-delay", type=float, default=1.0, help="Seconds to wait between PDF downloads.")
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=None)
    parser.add_argument("--reset-progress", action="store_true")

    args = parser.parse_args()

    if not args.api_key:
        print("Missing BRICKSET_API_KEY", file=sys.stderr)
        sys.exit(1)

    cache_dir = Path(args.cache)
    progress_path = cache_dir / "progress.json"

    if args.reset_progress and progress_path.exists():
        progress_path.unlink()
        print("Progress reset.")

    if args.set:
        set_nums = [normalize_set_num(s) for s in args.set]
        use_progress = False
    elif args.sets_file:
        set_nums = load_sets_file(args.sets_file)
        use_progress = False
    else:
        all_sets = load_all_set_nums(args.cache, args.year_min, args.year_max)
        progress = load_progress(progress_path)
        done = set(progress.get("done", []))

        remaining = [s for s in all_sets if s not in done]
        set_nums = remaining[: args.daily_limit]
        use_progress = True

        print(f"Total sets in catalog: {len(all_sets)}")
        print(f"Already checked: {len(done)}")
        print(f"Remaining: {len(remaining)}")
        print(f"Checking this run: {len(set_nums)}")

    print(f"Output folder: {Path(args.output).resolve()}")

    if not set_nums:
        print("No sets to check.")
        return

    total_found = 0
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for idx, set_num in enumerate(set_nums, start=1):
        print()
        print(f"[{idx}/{len(set_nums)}]")

        try:
            result = download_set(args.api_key, set_num, args.output, args.download_delay)
            total_found += result["found"]
            total_downloaded += result["downloaded"]
            total_skipped += result["skipped"]
            total_failed += result["failed"]

            if use_progress:
                progress = load_progress(progress_path)
                done = set(progress.get("done", []))
                done.add(set_num)
                progress["done"] = sorted(done)
                save_progress(progress_path, progress)

        except Exception as e:
            total_failed += 1
            message = str(e)
            print(f"FAILED SET {set_num}: {e}", file=sys.stderr)

            if "429" in message or "Too Many Requests" in message:
                print("Hit Brickset rate limit. Stopping this run. Try again later with a longer --set-delay.", file=sys.stderr)
                break

        # Always sleep between set checks to avoid Brickset rate limits.
        if idx < len(set_nums):
            print(f"Sleeping {int(args.set_delay)} seconds before next set...")
            time.sleep(args.set_delay)

    print()
    print("Done for this run.")
    print(f"Instruction records found: {total_found}")
    print(f"Downloaded: {total_downloaded}")
    print(f"Skipped existing: {total_skipped}")
    print(f"Failed: {total_failed}")
    print(f"Output: {Path(args.output).resolve()}")

    if use_progress:
        print(f"Progress file: {progress_path}")


if __name__ == "__main__":
    main()
