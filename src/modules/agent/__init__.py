"""M5 -- Agent: natural-language action planning + closed-loop motion control.

Sits above M3 (motion) and consumes its streaming inference API. The agent
turns a high-level natural-language instruction ("walk to the red ball, then
sit down") into a sequence of atomic action primitives that MotionSSM can
generate, while monitoring termination conditions against avatar world state
to drive smooth transitions.

Submodules:

  - :mod:`planner` -- the fine-tuned LM (GPT-2-small base, MIT) that decomposes
    free-form instructions into structured JSON action plans. Trained on
    HumanML3D-derived (instruction, action_list) pairs with outlines-enforced
    grammar at inference.

  - :mod:`conditions` -- termination-predicate parser. Converts "distance(
    scene.tree) < 1.0" into a callable that the runner evaluates each frame.

  - :mod:`world_state` -- avatar pose / position / velocity tracker. The
    runner updates this from each motion frame; conditions read it.

  - :mod:`runner` -- closed-loop coordinator. Pulls actions from the planner's
    queue, drives :meth:`TextToMotionSSM.stream_step` per action, checks
    conditions per frame, and transitions via :meth:`StreamingState.carry_over`
    when a condition fires.

Design goals (dissertation):
  - **No hardcoding**: every action that ends up in the motion model has come
    from the trained planner, not a canned list.
  - **No external API**: planner + motion + render all self-hosted on a
    single consumer GPU. The "real NLP" claim is defended by training the
    planner from a vanilla GPT-2-small, not by prompting a hosted LLM.
  - **Streaming-native**: conditions monitored per-step; transitions exploit
    Mamba's hidden-state carryover for smooth motion-to-motion continuity.
"""
