# python scripts/research/paper_watch.py --out_dir doc/notes/paper-watch
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

ARXIV_API = "https://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org/works"
OPENALEX_FIELDS = "id,title,doi,publication_date,abstract_inverted_index"
OPENALEX_KEY_VARIABLE = "OPENALEX_API_KEY"
ATOM = "{http://www.w3.org/2005/Atom}"
SEARCH_TERMS = (
    "text-to-motion",
    "streaming motion generation",
    "Mamba motion generation",
    "state space model human motion",
    "motion tokenizer FSQ",
    "HumanML3D",
)
CITED_PAPERS = MappingProxyType(
    {
        "T2M-GPT": "W4386076288",
        "MoMask": "W4402754111",
    }
)
USER_AGENT = "thesis-paper-watch/1.0"
RETRY_STATUS = 429
RETRY_COUNT = 5
RETRY_WAIT_SECONDS = 10.0
ARXIV_WAIT_SECONDS = 3.0
ARXIV_VERSION = re.compile(r"v\d+$")
ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(.+)$", re.IGNORECASE)
DOI_PREFIX = "https://doi.org/"
TITLE_WORD = re.compile(r"[a-z0-9]+")
ABSTRACT_LENGTH = 600
CITATION_LIMIT = 100
SEEN_FILE = "seen.json"
MOTION_WORDS = ("motions?", "animation", "humanml3d", "smpl", "smpl-x")
TOPIC_WORDS = (
    "text",
    "textual",
    "language",
    "captions?",
    "prompts?",
    "mamba",
    "state[- ]space",
    "ssm",
    "streaming",
    "autoregressive",
    "tokens?",
    "tokenizers?",
    r"quantiz\w*",
    "fsq",
    "vq-vae",
)
MOTION_PATTERN = re.compile(rf"\b({'|'.join(MOTION_WORDS)})\b")
TOPIC_PATTERN = re.compile(rf"\b({'|'.join(TOPIC_WORDS)})\b")


@dataclass(frozen=True)
class Paper:
    key: str
    title: str
    abstract: str
    url: str
    date: str
    source: str


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(RETRY_COUNT):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            if error.code != RETRY_STATUS:
                raise
            if attempt == RETRY_COUNT - 1:
                raise
            time.sleep(retry_wait_seconds(error, attempt))
    raise RuntimeError(f"no response from {url}")


def retry_wait_seconds(error: urllib.error.HTTPError, attempt: int) -> float:
    backoff = RETRY_WAIT_SECONDS * (attempt + 1)
    header = error.headers.get("Retry-After") if error.headers else None
    if header is None or not header.strip().isdigit():
        return backoff
    return max(backoff, float(header) + 1.0)


def arxiv_query_url(term: str, max_results: int) -> str:
    query = urllib.parse.urlencode(
        {
            "search_query": f'all:"{term}"',
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results,
        }
    )
    return f"{ARXIV_API}?{query}"


def arxiv_key(entry_id: str) -> str:
    identifier = entry_id.rstrip("/").rsplit("/", 1)[-1]
    return ARXIV_VERSION.sub("", identifier).lower()


def entry_text(entry: ElementTree.Element, tag: str) -> str:
    node = entry.find(f"{ATOM}{tag}")
    if node is None:
        return ""
    if node.text is None:
        return ""
    return " ".join(node.text.split())


def parse_arxiv_feed(feed: str) -> list[Paper]:
    papers = []
    for entry in ElementTree.fromstring(feed).iter(f"{ATOM}entry"):
        entry_id = entry_text(entry, "id")
        paper = Paper(
            key=arxiv_key(entry_id),
            title=entry_text(entry, "title"),
            abstract=entry_text(entry, "summary"),
            url=entry_id,
            date=entry_text(entry, "published")[:10],
            source="arXiv",
        )
        papers.append(paper)
    return papers


def openalex_url(filters: str, limit: int, api_key: str) -> str:
    parameters = {
        "filter": filters,
        "per-page": limit,
        "select": OPENALEX_FIELDS,
        "sort": "publication_date:desc",
    }
    if api_key:
        parameters["api_key"] = api_key
    return f"{OPENALEX_API}?{urllib.parse.urlencode(parameters)}"


def openalex_search_filter(term: str, since: datetime.date) -> str:
    return f'title_and_abstract.search:"{term}",from_publication_date:{since.isoformat()}'


def openalex_citation_filter(work_id: str, since: datetime.date) -> str:
    return f"cites:{work_id},from_publication_date:{since.isoformat()}"


def abstract_text(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    placed = [
        (position, word) for word, positions in inverted_index.items() for position in positions
    ]
    return " ".join(word for _, word in sorted(placed))


def openalex_key(record: dict) -> str:
    doi = str(record.get("doi") or "").removeprefix(DOI_PREFIX)
    arxiv = ARXIV_DOI.search(doi)
    if arxiv:
        return ARXIV_VERSION.sub("", arxiv.group(1)).lower()
    if doi:
        return doi.lower()
    return str(record["id"]).rsplit("/", 1)[-1].lower()


def parse_openalex(payload: dict, source: str) -> list[Paper]:
    papers = []
    for record in payload.get("results") or []:
        if not record.get("id"):
            continue
        paper = Paper(
            key=openalex_key(record),
            title=" ".join(str(record.get("title") or "").split()),
            abstract=abstract_text(record.get("abstract_inverted_index")),
            url=str(record.get("doi") or record["id"]),
            date=str(record.get("publication_date") or ""),
            source=source,
        )
        papers.append(paper)
    return papers


def matches_thesis(paper: Paper) -> bool:
    text = f"{paper.title} {paper.abstract}".lower()
    has_motion = MOTION_PATTERN.search(text) is not None
    has_topic = TOPIC_PATTERN.search(text) is not None
    return has_motion and has_topic


def title_key(title: str) -> str:
    return " ".join(TITLE_WORD.findall(title.lower()))


def paper_names(paper: Paper) -> frozenset[str]:
    return frozenset({paper.key, title_key(paper.title)})


def unseen_papers(papers: list[Paper], seen: frozenset[str]) -> list[Paper]:
    known = set(seen)
    fresh = []
    for paper in papers:
        names = paper_names(paper)
        if names & known:
            continue
        known.update(names)
        fresh.append(paper)
    return fresh


def read_seen(out_dir: Path) -> frozenset[str]:
    path = out_dir / SEEN_FILE
    if not path.exists():
        return frozenset()
    return frozenset(json.loads(path.read_text(encoding="utf-8")))


def write_seen(out_dir: Path, seen: frozenset[str]) -> None:
    path = out_dir / SEEN_FILE
    path.write_text(json.dumps(sorted(seen), indent=1), encoding="utf-8")


def report_markdown(papers: list[Paper], day: datetime.date) -> str:
    lines = [f"# Paper watch {day.isoformat()}", "", f"New papers: {len(papers)}", ""]
    for paper in sorted(papers, key=lambda item: item.date, reverse=True):
        lines.extend(
            [
                f"## {paper.title}",
                f"- id: {paper.key} | date: {paper.date} | source: {paper.source}",
                f"- url: {paper.url}",
                f"- abstract: {paper.abstract[:ABSTRACT_LENGTH]}",
                "- mark: ",
                "",
            ]
        )
    return "\n".join(lines)


def collect_papers(max_results: int, since: datetime.date, api_key: str) -> list[Paper]:
    papers = []
    for term in SEARCH_TERMS:
        papers.extend(parse_arxiv_feed(fetch_text(arxiv_query_url(term, max_results))))
        time.sleep(ARXIV_WAIT_SECONDS)
        search_url = openalex_url(openalex_search_filter(term, since), max_results, api_key)
        papers.extend(parse_openalex(json.loads(fetch_text(search_url)), "OpenAlex search"))
    for name, work_id in CITED_PAPERS.items():
        citation_url = openalex_url(
            openalex_citation_filter(work_id, since), CITATION_LIMIT, api_key
        )
        papers.extend(parse_openalex(json.loads(fetch_text(citation_url)), f"cites {name}"))
    return papers


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("doc/notes/paper-watch"))
    parser.add_argument("--max_results", type=int, default=20)
    parser.add_argument("--days", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    api_key = os.environ.get(OPENALEX_KEY_VARIABLE, "")
    print(f"OpenAlex key: {'set' if api_key else 'not set, shared free budget'}")
    today = datetime.date.today()
    since = today - datetime.timedelta(days=arguments.days)
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    report = arguments.out_dir / f"{today.isoformat()}.md"
    if report.exists():
        raise FileExistsError(f"{report} exists; one run per day keeps its papers")
    seen = read_seen(arguments.out_dir)
    collected = collect_papers(arguments.max_results, since, api_key)
    relevant = [paper for paper in collected if matches_thesis(paper)]
    fresh = unseen_papers(relevant, seen)
    report.write_text(report_markdown(fresh, today), encoding="utf-8")
    names = frozenset().union(*(paper_names(paper) for paper in fresh))
    write_seen(arguments.out_dir, seen | names)
    print(f"{len(collected)} collected, {len(relevant)} relevant, {len(fresh)} new -> {report}")


if __name__ == "__main__":
    main()
