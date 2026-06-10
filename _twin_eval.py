"""Back-compat shim — the driver was promoted to ``text2motion.eval.evaluate`` (same CLI, plus
``--mm_clips/--mm_repeats`` MultiModality and ``--length_mode end``). Commands in STATUS/ADRs that
reference ``_twin_eval.py`` keep working.

    PYTHONPATH=src python _twin_eval.py --backbone both --split test --cfg_scale 5.0
"""

from text2motion.eval.evaluate import main

if __name__ == "__main__":
    main()
