#!/usr/bin/env python3
"""
scrape_fed.py - build the FedSpeak text corpus from federalreserve.gov.

Three sources, all from the Board's official archive pages:
  * FOMC statements   (1994 -> present)
  * FOMC minutes      (1994 -> present)
  * Beige Books       (Oct 1996 -> present; earlier ones exist only as PDF)

Output layout (relative to this file):
  data/raw/    cached HTML exactly as downloaded, one file per URL
  data/clean/  one cleaned .txt per document, grouped by source
  data/fedspeak.txt   every cleaned document, sorted by date, ready for training

Usage:
  python scrape_fed.py --list                 # discover documents, download nothing new
  python scrape_fed.py --years 2010,2026      # scrape only these years (for testing)
  python scrape_fed.py                        # full run
"""

import argparse
import datetime as dt
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urldefrag, urljoin

import requests
from bs4 import BeautifulSoup, Comment

BASE = "https://www.federalreserve.gov"
# Identify the scraper and give the site operator a way to reach the project,
# without putting anyone's personal email address in every request.
USER_AGENT = "FedSpeak-research/0.1 (personal learning project; +https://github.com/csem11/fedSpeak)"
REQUEST_DELAY = 0.5          # seconds between live requests; cached pages cost nothing
START_YEAR = 1994
THIS_YEAR = dt.date.today().year

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
CORPUS = DATA / "fedspeak.txt"

log = logging.getLogger("fedspeak")


# --------------------------------------------------------------------------
# 1. Fetching: one function, always cached, always polite.
# --------------------------------------------------------------------------

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_last_request = 0.0


def cache_path(url: str) -> Path:
    """Map a URL to a stable file under data/raw/, mirroring the URL path."""
    path = url.replace(BASE, "").strip("/")
    if path.endswith("/") or not path:
        path += "index.htm"
    return RAW / path.replace("/", "__")


def fetch(url: str) -> bytes:
    """Return the page as raw bytes. Bytes, not str: older Fed pages are
    ISO-8859-1 and BeautifulSoup does a better job sniffing the charset than
    requests does guessing it."""
    global _last_request
    url = urljoin(BASE, url)
    p = cache_path(url)
    if p.exists():
        return p.read_bytes()

    wait = REQUEST_DELAY - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    log.debug("GET %s", url)
    resp = _session.get(url, timeout=30)
    _last_request = time.time()
    resp.raise_for_status()

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(resp.content)
    return resp.content


def soup_of(url: str) -> BeautifulSoup:
    return BeautifulSoup(fetch(url), "html.parser")


# --------------------------------------------------------------------------
# 2. Extraction: find the body text inside whichever layout the page uses.
# --------------------------------------------------------------------------

# Tags that never contain prose we want. <sup> is footnote markers.
DROP_TAGS = ["script", "style", "nav", "header", "footer", "noscript",
             "form", "iframe", "sup", "img", "button", "select"]

BLOCK_TAGS = ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr",
              "table", "blockquote", "pre", "section", "article"]


def main_container(soup: BeautifulSoup):
    """The Fed changed its page templates several times since 1994.
    Try the modern container first and fall back to older ones."""
    for selector in ("div#article", "div#content"):
        el = soup.select_one(selector)
        if el and len(el.get_text(strip=True)) > 200:
            return el
    # 1994-2005 pages have no content wrapper: prose is scattered across many
    # table cells and sometimes outside any table. Take the whole body and let
    # the line filters in clean_text() remove the (small) navigation.
    return soup.body or soup


def extract_text(html: bytes) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(DROP_TAGS):
        tag.decompose()
    container = main_container(soup)

    # 1990s pages hard-wrap paragraphs with raw newlines in the source. Flatten
    # those to spaces first, so the only newlines left are ones *we* add for
    # structure in the next step.
    for comment in container.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()          # HTML comments are strings too; drop them or they become text
    for node in container.find_all(string=True):
        node.replace_with(re.sub(r"\s*[\r\n]+\s*", " ", node))

    # Turn structure into newlines *before* flattening, so that inline tags
    # like <a> and <em> do not split sentences, but paragraphs still separate.
    for br in container.find_all("br"):
        br.replace_with("\n")
    for tag in container.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.append("\n")
    return container.get_text()


# --------------------------------------------------------------------------
# 3. Cleaning: normalise characters, drop site chrome, tidy whitespace.
# --------------------------------------------------------------------------

# Typographic characters -> plain ASCII equivalents. A character-level model
# has one vocabulary slot per distinct character, so fewer variants is better.
CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "--", "―": "--",
    "…": "...",
    " ": " ", " ": " ", " ": " ", "­": "",
})

# Lines that are navigation, press-release framing, or footers, not prose.
# Matched against each stripped line; a match drops the whole line.
BOILERPLATE = [re.compile(p, re.I) for p in (
    r"^home\b.*(>|›)",                 # breadcrumbs: Home > Monetary Policy > ...
    r"^(home|print|share|pdf|html|accessibility|contact us|return to top)$",
    r"^skip to ",
    r"^last update",
    r"^please enable javascript",
    r"^board of governors of the federal reserve system",
    r"^(monetary policy|federal open market committee|beige book|press release|news & events|press releases)$",
    r"^(for (immediate )?release|release date)\b",
    r"^\d{1,3}$",                      # bare page numbers
    r"^(summary|districts|full report)$",
    r"^summary\s+districts\s+boston\b",           # old Beige Book section nav
    r"^(boston|new york|philadelphia|cleveland|richmond|atlanta|chicago|st\. louis|minneapolis|kansas city|dallas|san francisco)$",
    r"^\d{4} calendar$",
    r"^related (information|content)$",
    r"^implementation note issued ",
    r"^federal reserve issues fomc statement$",
    r"^fomc minutes$",                 # tab label on 2006-2010 minutes pages
    r"^>\s",                          # breadcrumb fragments: "> Monetary Policy"
    r"^\|",                           # "| Monetary Policy" footer fragment
    r"^.{0,60}\|.{0,60}$",            # short line with a pipe = 1990s footer nav ("Home | Press releases")
    r"^to comment on this site",
    r"^(disclaimer|website policies|foia|pdf reader)$",
    r"^accessibility\b.*\bcontact us\b",   # joined footer link row
    r"^_{3,}$",                        # footnote separator rule
    r"^for media inquiries",
    r"^for more information about district economic conditions",
)]

# Suffixes glued onto otherwise-real lines.
INLINE_JUNK = re.compile(r"\s*\b(Return to text|Return to top)\b\.?$", re.I)


def clean_text(text: str) -> tuple[str, int]:
    """Return (clean_text, number_of_non_ascii_chars_dropped)."""
    text = text.translate(CHAR_MAP)
    # Decompose accented letters (é -> e + accent) then drop what is not ASCII.
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    dropped = sum(1 for ch in decomposed if ord(ch) > 127)

    lines = []
    for line in ascii_text.splitlines():
        line = re.sub(r"[ \t\r\f\v]+", " ", line).strip()
        line = INLINE_JUNK.sub("", line).strip()
        if not line:
            lines.append("")
            continue
        if any(p.search(line) for p in BOILERPLATE):
            continue
        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)   # at most one blank line between paragraphs
    return text.strip() + "\n", dropped


# --------------------------------------------------------------------------
# 4. Collectors: one per source. Each yields Doc(source, date, url, [extra urls]).
# --------------------------------------------------------------------------

@dataclass
class Doc:
    source: str          # "statements" | "minutes" | "beigebook"
    date: str            # YYYYMMDD
    url: str             # main page
    parts: tuple = ()    # extra pages that belong to the same document (2024+ Beige Book districts)

    @property
    def year(self) -> int:
        return int(self.date[:4])


def canonical(href: str) -> str:
    """Absolute https URL without fragment, so one page is never counted twice
    (the Fed links some pages both relatively and as http://www... absolutes)."""
    url, _fragment = urldefrag(urljoin(BASE, href))
    return url.replace("http://", "https://")


def date_in(url: str) -> str | None:
    m = re.search(r"(\d{8})", url)
    return m.group(1) if m else None


def fomc_index_pages():
    """The Fed lists recent meetings on one calendar page and older meetings on
    one page per year. Yield (label, soup) for every page we need."""
    yield "current", soup_of("/monetarypolicy/fomccalendars.htm")
    for year in range(START_YEAR, THIS_YEAR + 1):
        try:
            yield str(year), soup_of(f"/monetarypolicy/fomchistorical{year}.htm")
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                log.debug("no historical page for %s (covered by current page)", year)
                continue
            raise


def collect_statements_and_minutes() -> tuple[list[Doc], list[Doc]]:
    statements, minutes = {}, {}
    for label, soup in fomc_index_pages():
        for a in soup.find_all("a", href=True):
            href = canonical(a["href"])
            text = a.get_text(" ", strip=True)
            date = date_in(href)
            if not date or not (href.endswith(".htm") or href.endswith("/")):
                continue

            # Link *labels* have stayed stable across thirty years of redesigns
            # even though the URLs have not. The policy statement is labelled
            # exactly "Statement" on the per-year pages and "HTML" (inside a
            # "Statement:" cell) on the current calendar page. Anything with a
            # descriptive label ("Statement on Longer-Run Goals ...") is a
            # different document and is skipped on purpose.
            is_statement = text == "Statement" or (
                text == "HTML" and re.search(r"/pressreleases/monetary\d{8}a\.htm$", href))
            # Minutes URLs contain "minutes" except for June 2008, which the Fed
            # published at /monetarypolicy/fomc20080625.htm.
            is_minutes = text in ("Minutes", "HTML") and re.search(
                r"minutes|/monetarypolicy/fomc\d{8}\.htm$", href, re.I)

            if is_statement:
                statements.setdefault(href, Doc("statements", date, href))
            elif is_minutes:
                minutes.setdefault(href, Doc("minutes", date, href))
        log.debug("index %-7s -> running totals: %d statements, %d minutes", label, len(statements), len(minutes))
    return list(statements.values()), list(minutes.values())


DISTRICT_SLUGS = ["boston", "new-york", "philadelphia", "cleveland", "richmond", "atlanta",
                  "chicago", "st-louis", "minneapolis", "kansas-city", "dallas", "san-francisco"]


# Reports that exist on the site but are missing from the Fed's own year index.
EXTRA_BEIGE_BOOKS = [
    ("20030903", "/fomc/beigebook/2003/20030903/FullReport.htm"),   # absent from beigebook2003.htm
]


def collect_beige_books() -> list[Doc]:
    """Four URL generations, mapped during reconnaissance:
        1996-2010  /fomc/beigebook/YYYY/YYYYMMDD/default.htm   -> FullReport.htm has everything
        2011-2016  /monetarypolicy/beigebook/beigebookYYYYMM.htm -> page is the full report
        2017-2023  /monetarypolicy/beigebookYYYYMM.htm          -> page is the full report
        2024-      /monetarypolicy/beigebookYYYYMM-summary.htm  -> summary + 12 district pages
    """
    index_urls = [f"/monetarypolicy/beigebook{y}.htm" for y in range(1996, THIS_YEAR)]
    index_urls.append("/monetarypolicy/publications/beige-book-default.htm")

    docs = {}
    for index_url in index_urls:
        soup = soup_of(index_url)
        anchors = soup.find_all("a", href=True)
        for i, a in enumerate(anchors):
            if a.get_text(strip=True) != "HTML":
                continue
            href = urljoin(BASE, a["href"])

            # The release date is only reliably in the sibling PDF link
            # (BeigeBook_YYYYMMDD.pdf); older HTML URLs also contain it.
            date = date_in(href)
            if not date:
                for sib in anchors[i + 1:i + 3]:
                    if sib["href"].lower().endswith(".pdf"):
                        date = date_in(sib["href"])
                        break
            if not date:
                m = re.search(r"beigebook(\d{6})", href)
                date = m.group(1) + "01" if m else None
            if not date:
                log.warning("could not date Beige Book link %s", href)
                continue

            if "/fomc/beigebook/" in href:                         # 1996-2010
                main = href.replace("default.htm", "FullReport.htm")
                parts = ()
            elif href.endswith("-summary.htm"):                    # 2024+
                main = href
                parts = tuple(href.replace("-summary.htm", f"-{d}.htm") for d in DISTRICT_SLUGS)
            else:                                                  # 2011-2023
                main = href
                parts = ()
            docs.setdefault(main, Doc("beigebook", date, main, parts))
    for date, url in EXTRA_BEIGE_BOOKS:
        docs.setdefault(urljoin(BASE, url), Doc("beigebook", date, urljoin(BASE, url)))
    return list(docs.values())


# --------------------------------------------------------------------------
# 5. Driver
# --------------------------------------------------------------------------

def process(doc: Doc) -> tuple[Path, int, int]:
    """Download, extract, clean and save one document. Returns (path, chars, dropped)."""
    pieces = []
    total_dropped = 0
    for url in (doc.url, *doc.parts):
        text, dropped = clean_text(extract_text(fetch(url)))
        pieces.append(text)
        total_dropped += dropped
    text = "\n".join(pieces)
    out = CLEAN / doc.source / f"{doc.date}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="ascii")
    return out, len(text), total_dropped


def parse_years(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    years = set()
    for chunk in spec.split(","):
        if "-" in chunk:
            a, b = chunk.split("-")
            years.update(range(int(a), int(b) + 1))
        else:
            years.add(int(chunk))
    return years


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="discover documents and print counts; download nothing else")
    ap.add_argument("--years", help="only these years, e.g. 2010,2018-2020,2026")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    statements, minutes = collect_statements_and_minutes()
    beige = collect_beige_books()
    docs = sorted(statements + minutes + beige, key=lambda d: (d.date, d.source))

    years = parse_years(args.years)
    if years:
        docs = [d for d in docs if d.year in years]

    log.info("discovered %d documents: %d statements, %d minutes, %d beige books",
             len(docs), *(sum(d.source == s for d in docs) for s in ("statements", "minutes", "beigebook")))
    if args.list:
        by_year = {}
        for d in docs:
            by_year.setdefault(d.year, {"statements": 0, "minutes": 0, "beigebook": 0})[d.source] += 1
        print(f"{'year':>6} {'stmts':>6} {'mins':>6} {'beige':>6}")
        for y in sorted(by_year):
            c = by_year[y]
            print(f"{y:>6} {c['statements']:>6} {c['minutes']:>6} {c['beigebook']:>6}")
        return

    stats = {"statements": [0, 0, 0], "minutes": [0, 0, 0], "beigebook": [0, 0, 0]}
    for n, doc in enumerate(docs, 1):
        try:
            path, chars, dropped = process(doc)
        except requests.HTTPError as e:
            log.warning("skip %s %s: HTTP %s", doc.source, doc.date, e.response.status_code)
            continue
        s = stats[doc.source]
        s[0] += 1; s[1] += chars; s[2] += dropped
        log.info("[%4d/%d] %-10s %s  %7d chars  -> %s", n, len(docs), doc.source, doc.date, chars, path.relative_to(HERE))

    # Concatenate in date order so a positional train/val split is chronological
    # across all three sources rather than "train on minutes, validate on Beige Books".
    with CORPUS.open("w", encoding="ascii") as f:
        for doc in docs:
            p = CLEAN / doc.source / f"{doc.date}.txt"
            if p.exists():
                f.write(p.read_text(encoding="ascii"))
                f.write("\n")

    log.info("---- corpus summary ----")
    for source, (n, chars, dropped) in stats.items():
        log.info(f"{source:<10} {n:4d} docs  {chars:>12,} chars  ({dropped:,} non-ASCII chars dropped)")
    log.info(f"{'TOTAL':<10} {sum(v[0] for v in stats.values()):4d} docs  "
             f"{CORPUS.stat().st_size:>12,} chars  -> {CORPUS.relative_to(HERE)}")


if __name__ == "__main__":
    main()
