import dis
import io

#  Patch sys.path so we can import from project root
import os
import pathlib
import sys
import types
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

#  Lazy: avoid triggering GPU imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # no GPU needed for analysis

from src.modules.planner.planner import ScenePlanner
from src.modules.understanding import SpacyParser
from src.pipeline import Pipeline

#  Helpers


@dataclass
class FuncStats:
    name: str
    nInstructions: int
    nLoadAttr: int  # attribute lookups (can cache)
    nCall: int  # function calls (overhead)
    nForIter: int  # loop iterations
    nCompare: int  # comparisons
    nLoadGlobal: int  # global lookups (can hoist)

    @property
    def hottestOpcode(self) -> str:
        counts = {
            "LOAD_ATTR": self.nLoadAttr,
            "CALL": self.nCall,
            "FOR_ITER": self.nForIter,
            "COMPARE_OP": self.nCompare,
            "LOAD_GLOBAL": self.nLoadGlobal,
        }
        return max(counts, key=counts.__getitem__)


def analyseFunction(func: types.FunctionType) -> FuncStats:
    """Disassemble *func* and count key instruction types."""
    buf = io.StringIO()
    dis.dis(func, file=buf)
    lines = buf.getvalue().splitlines()

    counters = dict.fromkeys(
        ("LOAD_ATTR", "CALL", "CALL_FUNCTION", "FOR_ITER", "COMPARE_OP", "LOAD_GLOBAL"), 0
    )
    total = 0
    for line in lines:
        for op in counters:
            if op in line:
                counters[op] += 1
        if line.strip() and line.strip()[0:2] not in (">>", ">>"):
            if any(c.isdigit() for c in line[:10]):
                total += 1

    return FuncStats(
        name=func.__qualname__,
        nInstructions=total,
        nLoadAttr=counters["LOAD_ATTR"],
        nCall=counters["CALL"] + counters["CALL_FUNCTION"],
        nForIter=counters["FOR_ITER"],
        nCompare=counters["COMPARE_OP"],
        nLoadGlobal=counters["LOAD_GLOBAL"],
    )


def printBytecode(func: types.FunctionType, highlightHot: bool = True) -> None:
    """Print annotated bytecode for a function."""
    print(f"\n{'='*70}")
    print(f"  BYTECODE: {func.__qualname__}")
    print(f"{'='*70}")
    buf = io.StringIO()
    dis.dis(func, file=buf)
    text = buf.getvalue()
    if highlightHot:
        HOT_OPS = ("FOR_ITER", "LOAD_ATTR", "LOAD_GLOBAL")
        for line in text.splitlines():
            marker = "  HOT" if any(op in line for op in HOT_OPS) else ""
            print(line + marker)
    else:
        print(text)


def printStats(funcs: list) -> None:
    """Print a summary table of function stats."""
    print(f"\n{'='*70}")
    print("  PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    hdr = f"{'Function':<45} {'Instr':>6} {'Attr':>5} {'Call':>5} {'Loop':>5} {'Global':>7}"
    print(hdr)
    print("-" * 70)
    for stat in sorted(funcs, key=lambda s: s.nInstructions, reverse=True):
        print(
            f"{stat.name:<45} {stat.nInstructions:>6} "
            f"{stat.nLoadAttr:>5} {stat.nCall:>5} "
            f"{stat.nForIter:>5} {stat.nLoadGlobal:>7}"
        )


def printRecommendations(stats: list) -> None:
    """Print concrete optimization advice based on bytecode analysis."""
    print(f"\n{'='*70}")
    print("  OPTIMIZATION RECOMMENDATIONS")
    print(f"{'='*70}")

    for stat in stats:
        issues = []

        # High attribute lookups inside loops -> cache as local var
        if stat.nLoadAttr > 8 and stat.nForIter > 0:
            issues.append(
                f"    {stat.nLoadAttr} LOAD_ATTR inside loops "
                f"-> cache self.xxx as local variables before loop entry\n"
                f"     e.g.: clips = self._motion_gen  (saves repeated dict/obj lookup)"
            )

        # Many global lookups -> hoist to local
        if stat.nLoadGlobal > 6:
            issues.append(
                f"  !!  {stat.nLoadGlobal} LOAD_GLOBAL -> hoist frequently used globals "
                f"(numpy, GRAVITY) to local vars at function start"
            )

        # High call count -> spot inline-able helpers
        if stat.nCall > 15:
            issues.append(
                f"  !!  {stat.nCall} CALL instructions -> "
                f"consider inlining hot helper calls or using __slots__"
            )

        if issues:
            print(f"\n  [{stat.name}]")
            for issue in issues:
                print(issue)
        else:
            print(f"\n  [{stat.name}]  OK No major hotspots detected.")


#  Import target modules (without GPU)


def main():
    print("Importing modules for analysis...")

    # Pipeline methods removed with the physics/diffusion stages; the runtime
    # is now a thin orchestrator with a single `run` method, so we profile
    # that plus the per-stage entry points.
    targets = [
        Pipeline.run,
        SpacyParser.parse,
        ScenePlanner.plan,
    ]

    stats = [analyseFunction(fn) for fn in targets]
    printStats(stats)
    printRecommendations(stats)

    # Detailed bytecode for two hottest functions
    hottest = sorted(stats, key=lambda s: s.nInstructions, reverse=True)[:2]
    hotFuncs = [fn for fn in targets if fn.__qualname__ in {s.name for s in hottest}]
    for fn in hotFuncs:
        printBytecode(fn)

    print(f"\n{'='*70}")
    print("  INSTRUCTION COUNT LEGEND")
    print(f"{'='*70}")
    print("  Instr   = total bytecode instructions (lower = simpler)")
    print("  Attr    = LOAD_ATTR (attribute lookups; expensive in loops)")
    print("  Call    = CALL instructions (function call overhead)")
    print("  Loop    = FOR_ITER (each iteration boundary)")
    print("  Global  = LOAD_GLOBAL (module-level name lookups)")
    print()
    print("  General guidance (CPython 3.12):")
    print("  - FOR_ITER in tight loops: consider numpy vectorisation")
    print("  - LOAD_ATTR in loops: hoist obj.attr to local before loop")
    print("  - LOAD_GLOBAL repeated: alias module at function top (np = numpy)")
    print("  - Repeated CALL in loop: use list comprehension or np.vectorize")


if __name__ == "__main__":
    main()
