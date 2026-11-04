import importlib.util
import sys
from pathlib import Path

WATCH_SCRIPT = Path("scripts/research/paper_watch.py")
ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.06052v2</id>
    <published>2023-01-15T10:00:00Z</published>
    <title>T2M-GPT:  Generating
      Human Motion</title>
    <summary>Discrete tokens.</summary>
  </entry>
</feed>
"""


def __watch_module(monkeypatch):
    spec = importlib.util.spec_from_file_location("paper_watch", WATCH_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_an_arxiv_entry_keeps_its_id_without_the_version(monkeypatch):
    watch = __watch_module(monkeypatch)
    papers = watch.parse_arxiv_feed(ARXIV_FEED)
    assert len(papers) == 1
    assert papers[0].key == "2301.06052"
    assert papers[0].title == "T2M-GPT: Generating Human Motion"
    assert papers[0].date == "2023-01-15"


def test_an_openalex_record_gets_the_best_key_and_its_abstract_back(monkeypatch):
    watch = __watch_module(monkeypatch)
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W1",
                "title": "A",
                "doi": "https://doi.org/10.48550/arXiv.2501.00001",
                "abstract_inverted_index": {"tokens": [1], "Discrete": [0]},
            },
            {"id": "https://openalex.org/W2", "title": "B", "doi": "https://doi.org/10.1145/X.Y"},
            {"id": "https://openalex.org/W3", "title": "C", "doi": None},
            {"id": None, "title": None},
        ]
    }
    papers = watch.parse_openalex(payload, "cites T2M-GPT")
    assert [paper.key for paper in papers] == ["2501.00001", "10.1145/x.y", "w3"]
    assert papers[0].abstract == "Discrete tokens"


def test_the_filter_needs_a_motion_word_and_a_topic_word(monkeypatch):
    watch = __watch_module(monkeypatch)
    mamba = watch.Paper("1", "Motion Mamba", "", "", "", "arXiv")
    video = watch.Paper("2", "Video super-resolution", "text prompts", "", "", "arXiv")
    dance = watch.Paper("3", "Dance motion from music", "", "", "", "arXiv")
    assert watch.matches_thesis(mamba)
    assert not watch.matches_thesis(video)
    assert not watch.matches_thesis(dance)


def test_a_paper_seen_before_or_found_twice_appears_once(monkeypatch):
    watch = __watch_module(monkeypatch)
    first = watch.Paper("2501.00001", "A", "", "", "2025-01-01", "arXiv")
    again = watch.Paper("10.1145/x.y", "A:", "", "", "2025-01-01", "OpenAlex search")
    old = watch.Paper("2401.00001", "B", "", "", "2024-01-01", "arXiv")
    fresh = watch.unseen_papers([first, again, old], frozenset({"2401.00001"}))
    assert fresh == [first]
