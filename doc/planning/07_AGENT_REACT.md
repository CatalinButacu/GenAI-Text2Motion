# 07 — Conversational agent: ReAct + Planning + Memory fit study

Status: **draft, not yet implemented**. Authored 2026-05-21.

## Why this doc exists

The long-term product (see [project_chat_product_vision.md] in memory) is a continuous chat over motion scenes: user types, the system parses → plans → generates → renders, but persistently — multi-turn edits, multi-actor, scene state held across turns. This doc evaluates whether the canonical "ReAct + Planning + Memory" agent pattern is the right wrapper around our existing pipeline, and sketches a minimal first slice.

## The verdict (one line)

**PARTIAL fit.** Memory is essential. Planning is overkill for the common case. A LangGraph-style state machine with a *small* ReAct sub-loop for ambiguous edits beats a free ReAct loop. Industry practice 2026 is graph-structured (>70% of production agents), not free ReAct.

## What ReAct buys us — and what it doesn't

| User intent                                          | Needs reasoning? | Routing  |
|------------------------------------------------------|------------------|----------|
| "generate a man walking"                             | no               | direct → M1→M2→M4→M6 |
| "render at 60fps"                                    | no               | direct → config patch |
| "what did I just generate?"                          | no               | direct → memory lookup |
| "make *him* kick harder" (3 actors on stage)         | yes (referent)   | ReAct sub-loop |
| "kick harder but keep the walk"                      | yes (compound)   | ReAct sub-loop |
| "first reposition B, then change his motion"         | yes (planning)   | optional `decomposePlan` tool |
| "the motion clipped through the wall — fix it"       | yes (repair)     | ReAct sub-loop + re-plan |

~80% of utterances route trivially. ReAct only earns its keep on the 20% that need disambiguation across turns.

## Where memory is load-bearing

Memory is what makes the chat product *work*. Without it, every turn is a fresh single-shot — defeats the point.

Three tiers, MemGPT-inspired:

1. **Core context (always in prompt)**: user prefs, persona, current scene summary.
2. **Recall memory (DB-backed)**: full chat history + agent traces. Retrievable.
3. **Archival memory (vector store)**: long-term preferences, named entities ("Bob the cowboy"), reusable motion clips by name.

The scene state itself — actor registry, current motion clip cache, last render config — is *episodic* and lives in core context.

## Failure modes (real, documented in 2025-2026)

- **Hallucinated tool args.** LLM emits `generateMotion(loopForever=True)` for an arg that doesn't exist. → Mitigate with strict pydantic schema + arg validation; reject and re-prompt.
- **Reflection loops.** Agent re-thinks without acting. → Cap iterations at N=6.
- **Latency.** Router emits ~3 tokens; ReAct emits 50+ per step × ~5 steps = 250+ tokens per turn = 3-5s on Gemini Flash, 10-20s on Claude Opus. → Route trivial intents *around* the agent.
- **Domain gap.** No verifiable reward for "is this motion good?" — Reflexion-style self-correction doesn't apply. The user is the oracle; design for human-in-the-loop preview + accept/reject.

## Proposed architecture

```
+--------------------------------------------------------------+
|                   LLM brain (Gemini 2.5 Flash)               |
|  Sees: system prompt + tool schemas + recent turns           |
|        + injected scene state (actor registry, last clip)    |
+--------------------------------------------------------------+
              | ToolCall(name, args)                   ^
              v                                        |
+--------------------------------------------------------------+
|                   Tool / Action layer                         |
|   parseScene(text)      -> SpacyParser.invoke()              |
|   planLayout(scene)     -> ScenePlanner.invoke()             |
|   generateMotion(...)   -> MotionGenerator.generate()        |
|   editClip(clipId,...)  -> MotionGenerator with priorClip    |
|   render(scene, clips)  -> renderSmplx2Video()               |
|   queryMemory(key)      -> SessionStore lookup               |
|   updateMemory(k, v)    -> SessionStore mutation             |
+--------------------------------------------------------------+
              |                                        ^
              v                                        |
+--------------------------------------------------------------+
|                   SessionStore (JSON-backed)                  |
|   scene:    ParsedScene                                       |
|   actors:   dict[actorId, ActorState]  (pos, motionClipId)    |
|   clips:    dict[clipId, MotionClip]   (LRU, last N)          |
|   history:  list[Turn]  (user msg + agent trace)              |
|   prefs:    UserPrefs   (fps, model, render quality)          |
+--------------------------------------------------------------+
```

Proposed repo layout (new top-level subpackage):

```
src/agent/
  __init__.py
  loop.py         # ReAct loop (only invoked for "needs reasoning" branch)
  router.py       # keyword/regex intent classifier — handles 80% of intents
  tools.py        # wrappers around existing module.invoke() functions
  store.py        # SessionStore dataclass + JSON persistence
  prompts.py      # system prompt + tool schemas
scripts/agent/
  chatDemo.py     # REPL: user line -> dispatch -> video
tests/
  test_agent.py   # mock LLM client, exercise router + tools + store
```

## Skeleton code

This is intent-conveying pseudocode, not finished — it imports our existing modules with the actual signatures.

```python
# src/agent/store.py
from __future__ import annotations
from dataclasses import dataclass, field
from src.modules.understanding.models import ParsedScene
from src.modules.motion.models import MotionClip


@dataclass(slots=True)
class ActorState:
    actorId: str
    position: tuple[float, float, float]
    motionClipId: str | None = None


@dataclass(slots=True)
class UserPrefs:
    fps: int = 30
    renderWidth: int = 1280
    renderHeight: int = 720


@dataclass(slots=True)
class Turn:
    userMsg: str
    agentTrace: list[dict]
    reply: str


@dataclass(slots=True)
class SessionStore:
    scene: ParsedScene | None = None
    actors: dict[str, ActorState] = field(default_factory=dict)
    clips: dict[str, MotionClip] = field(default_factory=dict)
    history: list[Turn] = field(default_factory=list)
    prefs: UserPrefs = field(default_factory=UserPrefs)
```

```python
# src/agent/tools.py
from __future__ import annotations
from dataclasses import dataclass
from src.modules import motion, planner, render, understanding
from src.modules.motion.generator import MotionGenerator
from src.shared.config import PipelineConfig
from .store import ActorState, SessionStore


@dataclass(slots=True)
class ToolResult:
    ok: bool
    payload: dict
    error: str | None = None


class ToolRegistry:

    def __init__(self, store: SessionStore, config: PipelineConfig) -> None:
        self.store = store
        self.config = config
        self.generator = MotionGenerator(config.motion)

    def parseScene(self, text: str) -> ToolResult:
        parsed = understanding.invoke(text, self.config.understanding)
        self.store.scene = parsed
        return ToolResult(ok=True, payload={"entityCount": len(parsed.entities)})

    def planLayout(self) -> ToolResult:
        if self.store.scene is None:
            return ToolResult(ok=False, payload={}, error="no scene parsed yet")
        planned = planner.invoke(self.store.scene, self.config.planner)
        for entity in planned.entities:
            self.store.actors[entity.actorId] = ActorState(
                actorId=entity.actorId,
                position=tuple(entity.position),
            )
        return ToolResult(ok=True, payload={"actors": list(self.store.actors)})

    def generateMotion(self, actorId: str, text: str) -> ToolResult:
        if actorId not in self.store.actors:
            return ToolResult(ok=False, payload={}, error=f"unknown actor: {actorId}")
        clip = self.generator.generate(text=text)
        clipId = f"clip_{len(self.store.clips)}"
        self.store.clips[clipId] = clip
        self.store.actors[actorId].motionClipId = clipId
        return ToolResult(ok=True, payload={"clipId": clipId})

    def render(self, outputPath: str) -> ToolResult:
        if not self.store.clips:
            return ToolResult(ok=False, payload={}, error="no clips to render")
        path = render.invoke(self.store.clips, outputPath, self.config.render)
        return ToolResult(ok=True, payload={"videoPath": path})

    def queryMemory(self, key: str) -> ToolResult:
        value = getattr(self.store, key, None)
        if value is None:
            return ToolResult(ok=False, payload={}, error=f"no such key: {key}")
        return ToolResult(ok=True, payload={"value": str(value)[:500]})
```

```python
# src/agent/router.py
from __future__ import annotations
from enum import Enum
import re


class Intent(Enum):
    NEW_SCENE = "newScene"
    EDIT_MOTION = "editMotion"
    QUERY = "query"
    RENDER = "render"
    PREFS = "prefs"
    AMBIGUOUS = "ambiguous"


NEW_RE = re.compile(r"\b(generate|create|new|make a|start with)\b", re.I)
EDIT_RE = re.compile(r"\b(change|edit|modify|make him|make her|make the|adjust)\b", re.I)
QUERY_RE = re.compile(r"\b(what|which|show me|how many|list)\b", re.I)
RENDER_RE = re.compile(r"\b(render|export|save|video)\b", re.I)


def classify(userMsg: str) -> Intent:
    msg = userMsg.strip().lower()
    if NEW_RE.search(msg):
        return Intent.NEW_SCENE
    if EDIT_RE.search(msg):
        return Intent.EDIT_MOTION
    if QUERY_RE.search(msg):
        return Intent.QUERY
    if RENDER_RE.search(msg):
        return Intent.RENDER
    return Intent.AMBIGUOUS
```

The ReAct loop itself lives in `loop.py` and is *only* invoked for `Intent.EDIT_MOTION` or `Intent.AMBIGUOUS`. The other intents are handled by deterministic dispatchers calling the tools directly — same code path, no LLM in the loop, instant response.

## First slice (one week)

**Goal**: "stateful single-actor edit chat" — 3-turn session: "a man walks" → "make him run instead" → "now stop after 2 seconds". Verify clip cache hits, state persists, only one new motion synthesized per edit (not full re-run).

| Day | Deliverable |
|-----|-------------|
| 1   | `src/agent/store.py` — SessionStore + JSON persistence to `runs/sessions/<id>.json` |
| 1-2 | `src/agent/tools.py` — wrap existing `invoke()`s into 5 tools above. Strict typed args. |
| 2   | `src/agent/router.py` — non-LLM intent classifier, routes 80% of intents to direct dispatchers |
| 3   | `src/agent/loop.py` — minimal ReAct loop, Gemini Flash backend, only called for ambiguous/edit intents |
| 4   | `scripts/agent/chatDemo.py` — REPL: read user line, dispatch, print video path or play |
| 5   | `tests/test_agent.py` — mock LLM client, exercise the 3-turn edit session. Verify clip cache reuse. |

**Defer to later weeks**: Reflexion-style self-correction, multi-actor referent resolution, planning decomposition, LangGraph migration (do that once tool surface is stable).

## Why not full LangGraph from day 1?

- LangGraph buys deterministic flow + retryable nodes + observable state. All useful, but it's a heavier framework dependency.
- For week 1 the value is in proving the *memory layer* carries the conversational UX. The orchestration framework can be swapped after we know the tool surface is right.
- Do migrate to LangGraph in week 2 if the simple router + sub-ReAct pattern feels like spaghetti or if we need pause-points (human-in-the-loop preview before render).

## Key references for the dissertation

- Yao et al. 2022 — original ReAct paper (https://arxiv.org/abs/2210.03629)
- Shinn et al. 2023 — Reflexion, verbal RL (https://arxiv.org/abs/2303.11366)
- Letta v1 blog (2025) — explicitly addresses why pure ReAct fails in production; rearchitecture lessons
- Motion-Agent (Wu, Zhao 2024) — closest published prior art; GPT-4 dispatcher over MotionLLM, multi-turn dialogue
- Iterative Motion Editing with Natural Language (Goel et al. SIGGRAPH 2024) — "motion editing operators" compiled from NL by an LLM
- LangGraph (langchain-ai) — current production-grade orchestration

## Open questions to resolve before implementing

1. Which LLM backend — Gemini 2.5 Flash (cheap, fast) vs Claude Sonnet (better at structured tool use)? Latency-sensitive: Gemini wins.
2. How do we surface "this motion is bad, redo" feedback — accept/reject button in REPL, or just "redo" command?
3. Do we expose `editClip` as a separate tool or fold it into `generateMotion(priorClipId=...)` with an optional arg?
4. JSON-backed sessions OR sqlite — JSON is fine for week 1; sqlite when multi-session search matters.
