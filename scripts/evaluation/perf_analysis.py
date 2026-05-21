import dis
import io

#  Patch sys.path so we can import from project root
import os
import pathlib
import types
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).parents[1]
#  Lazy: avoid triggering GPU imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # no GPU needed for analysis

from src.modules.planner.planner import ScenePlanner
from src.modules.understanding import SpacyParser
from src.pipeline import Pipeline

#  Helpers

@dataclass
class FuncStats:
    name: str
    n_instructions: int
    n_load_attr: int  # attribute lookups (can cache)
    n_call: int  # function calls (overhead)
    n_for_iter: int  # loop iterations
    n_compare: int  # comparisons
    n_load_global: int  # global lookups (can hoist)

    @property
    def hottest_opcode(self) -> str:
        counts = {
            "LOAD_ATTR": self.n_load_attr,
            "CALL": self.n_call,
            "FOR_ITER": self.n_for_iter,
            "COMPARE_OP": self.n_compare,
            "LOAD_GLOBAL": self.n_load_global,
        }
        return max(counts, key=counts.__getitem__)

def analyse_function(func: types.FunctionType) -> FuncStats:
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
        n_instructions=total,
        n_load_attr=counters["LOAD_ATTR"],
        n_call=counters["CALL"] + counters["CALL_FUNCTION"],
        n_for_iter=counters["FOR_ITER"],
        n_compare=counters["COMPARE_OP"],
        n_load_global=counters["LOAD_GLOBAL"],
    )

def print_bytecode(func: types.FunctionType, highlight_hot: bool = True) -> None:
    """Print annotated bytecode for a function."""
    print(f"\n{'='*70}")
    print(f"  BYTECODE: {func.__qualname__}")
    print(f"{'='*70}")
    buf = io.StringIO()
    dis.dis(func, file=buf)
    text = buf.getvalue()
    if highlight_hot:
        HOT_OPS = ("FOR_ITER", "LOAD_ATTR", "LOAD_GLOBAL")
        for line in text.splitlines():
            marker = "  HOT" if any(op in line for op in HOT_OPS) else ""
            print(line + marker)
    else:
        print(text)

def print_stats(funcs: list) -> None:
    """Print a summary table of function stats."""
    print(f"\n{'='*70}")
    print("  PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    hdr = f"{'Function':<45} {'Instr':>6} {'Attr':>5} {'Call':>5} {'Loop':>5} {'Global':>7}"
    print(hdr)
    print("-" * 70)
    for stat in sorted(funcs, key=lambda s: s.n_instructions, reverse=True):
        print(
            f"{stat.name:<45} {stat.n_instructions:>6} "
            f"{stat.n_load_attr:>5} {stat.n_call:>5} "
            f"{stat.n_for_iter:>5} {stat.n_load_global:>7}"
        )

def print_recommendations(stats: list) -> None:
    """Print concrete optimization advice based on bytecode analysis."""
    print(f"\n{'='*70}")
    print("  OPTIMIZATION RECOMMENDATIONS")
    print(f"{'='*70}")

    for stat in stats:
        issues = []

        # High attribute lookups inside loops -> cache as local var
        if stat.n_load_attr > 8 and stat.n_for_iter > 0:
            issues.append(
                f"    {stat.n_load_attr} LOAD_ATTR inside loops "
                f"-> cache self.xxx as local variables before loop entry\n"
                f"     e.g.: clips = self._motion_gen  (saves repeated dict/obj lookup)"
            )

        # Many global lookups -> hoist to local
        if stat.n_load_global > 6:
            issues.append(
                f"  !!  {stat.n_load_global} LOAD_GLOBAL -> hoist frequently used globals "
                f"(numpy, GRAVITY) to local vars at function start"
            )

        # High call count -> spot inline-able helpers
        if stat.n_call > 15:
            issues.append(
                f"  !!  {stat.n_call} CALL instructions -> "
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

    stats = [analyse_function(fn) for fn in targets]
    print_stats(stats)
    print_recommendations(stats)

    # Detailed bytecode for two hottest functions
    hottest = sorted(stats, key=lambda s: s.n_instructions, reverse=True)[:2]
    hot_funcs = [fn for fn in targets if fn.__qualname__ in {s.name for s in hottest}]
    for fn in hot_funcs:
        print_bytecode(fn)

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
