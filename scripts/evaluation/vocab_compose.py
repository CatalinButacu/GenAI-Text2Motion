"""Interactive compound-text -> atomic-label decomposition.

Reads a labels.csv produced by codebook_label.py and human-filled, then
decomposes a compound text prompt (e.g. "walk forward then sit down") into
a sequence of atomic action labels.

The same CodebookVocab class is what the SSM stage will call at inference:
    vocab.decomposeText("...")  -> [("walk_forward", 0.81), ("sit_down", 0.74)]

Usage:
    python scripts/evaluation/vocab_compose.py --labels <path>/labels.csv
    python scripts/evaluation/vocab_compose.py --labels <...>/labels.csv \\
        --text "the person walks forward then kicks a ball"
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.modules.motion.codebook_vocab import CodebookVocab

log = logging.getLogger(__name__)

DEMO_PROMPTS = [
    "the person walks forward",
    "walk forward then sit down",
    "stand up, walk to the door, then open it",
    "kick a ball and then run away",
    "wave the right hand",
]

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Path to labels.csv")
    parser.add_argument("--text", default=None, help="Compound text to decompose")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    vocab = CodebookVocab(args.labels)
    log.info("[compose] vocab labels: %s", vocab.labels())

    prompts = [args.text] if args.text else DEMO_PROMPTS

    for prompt in prompts:
        decomposition = vocab.decomposeText(prompt)
        log.info("[compose] %r", prompt)

        for label, score in decomposition:
            log.info("[compose]   -> %-24s  (sim=%.3f)", label, score)

    return 0

if __name__ == "__main__":
    sys.exit(main())
