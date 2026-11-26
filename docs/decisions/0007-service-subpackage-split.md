# ADR 0007 — Split of `app/service.py` into a sub-package

**Status:** Accepted (2026-11-26). Task A29-T7. The code move is task A29-T8.

**Related:** [ADR 0006](0006-step-runner-home.md) (same measurement method), task A28-T12 (the `steer` command).

## Context

`src/text2motion/app/service.py` has 723 lines. The size ratchet limit is 250 lines for a file and 30 for a function. The file holds the wire protocol, the steering state, the client, the server launcher and the server. Two splits into a sub-package `app/service/` were possible:

- A: wire protocol, session state, command handlers. This option names no home for the client, so the client is a fourth module.
- B: client and server. The shared wire protocol goes into the client module, and the server imports it.

| option | module        | content                                                                                                                                  |
| ------ | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| A      | `protocol.py` | `MotionStreamProtocol`, `Release`, `ServiceAddress`                                                                                      |
| A      | `state.py`    | `PendingSteer`, `StreamSteering`, steer constants, `LoadedInferencePipeline`, `InferencePipelineLoader`                                  |
| A      | `handlers.py` | `MotionInferenceServer`, `pending_request`, `process_memory_gb`, `record_inference`, `format_stats`                                      |
| A      | `client.py`   | `MotionInferenceClient`, `start_motion_server`, `await_motion_server`, `changed_identity_fields`, `motion_service_client`                |
| B      | `client.py`   | everything in A `protocol.py` and A `client.py`                                                                                          |
| B      | `server.py`   | everything in A `state.py` and A `handlers.py`                                                                                           |

## Measurement

One session, 2026-11-26, RTX 4050 laptop, no GPU job started. Three copies of `src` and `tests/streaming` in a temp folder: baseline, A, B. A script copied each top-level block of `service.py` unchanged into its module. `__init__.py` is empty in A and B, and the imports in `serving.py` and the streaming tests name the module. `ruff` removed unused imports and formatted the files; `ruff check` passed on `app` and `tests/streaming` in A and B. Each copy imported `text2motion` from its own `src`.

Import time: `python -X importtime`, cumulative time of the named module, median of 15, after one warm-up import for each target. The five targets ran in turn in each round, and the first target rotated.

| measure                                    | baseline              | A: protocol, state, handlers (+ client)                                                  | B: client and server        |
| ------------------------------------------ | --------------------- | ---------------------------------------------------------------------------------------- | --------------------------- |
| files                                      | 1                     | 4 modules + empty `__init__`                                                             | 2 modules + empty `__init__` |
| `.py` files directly in `app/` (limit 10)  | 10                    | 9                                                                                        | 9                           |
| file lines                                 | 723                   | protocol 72, state 126, client 222, handlers 333                                         | client 289, server 450      |
| largest file lines                         | 723                   | 333                                                                                      | 450                         |
| files above 250 lines                      | 1                     | 1                                                                                        | 2                           |
| largest function lines (`handle_generate`) | 122                   | 122                                                                                      | 122                         |
| functions above 30 lines                   | 5                     | 5                                                                                        | 5                           |
| imports between the new modules            | —                     | 3 statements, 7 names: client→protocol (3), handlers→protocol (1), handlers→state (3)    | 1 statement, 1 name: server→client (`MotionStreamProtocol`) |
| import cycle                               | —                     | none                                                                                     | none                        |
| import time, client side                   | 6727 ms (whole file)  | `client` 1835 ms (1775–2111)                                                             | `client` 1823 ms (1787–1936) |
| import time, server side                   | 6727 ms (6228–7215)   | `handlers` 6566 ms (6302–7555)                                                           | `server` 6706 ms (6367–7242) |
| torch loaded by the client side            | yes                   | yes                                                                                      | yes                         |
| `tests/streaming`                          | 37 passed, 110 s      | 37 passed, 109 s                                                                         | 37 passed, 115 s            |

Reading of the table:

- A split alone shortens no function. A29-T8 must do that in both options.
- Both options make the client import about 4.9 s faster, because it no longer loads the generation pipeline. A and B do not differ here: the ranges overlap.
- The client still loads torch in both options, through `motion/__init__.py` (see ADR 0006).
- Server import time does not change.
- An earlier timing pass in the same session did not rotate the order. It gave 3960 ms for the baseline in one loop and 6151 ms for the same module in the next loop, so the load of the machine changed. Its medians are not used. It gave the same direction: client 1874 ms (A) and 1816 ms (B). Its baseline test run passed 37 tests; the run then hit a 600 s time limit.

## Decision

Option A: `app/service/` with `protocol.py`, `state.py`, `handlers.py` and `client.py`, and an empty `__init__.py`.

- Largest file 333 lines against 450. One file stays above the limit against two.
- The client and the server both import the protocol from a neutral module. In B the server imports the client module.
- Import time and test result are equal, so they do not decide.
- The cost is 2 more modules and 6 more imported names between modules.

## Alternatives

1. B, client and server. Rejected: `server.py` is 450 lines and `client.py` is 289 lines, both above 250.
2. B with a third module for the protocol. Not measured. It was not one of the two options of the task.

## Consequences

- `handlers.py` has 333 lines. A29-T8 must move at least 83 lines out of it, and shorter functions can add lines. This needs one more module for part of the stream or statistics code. That module is not measured here.
- Imports change in `app/serving.py`, `tests/streaming/test_service.py`, `tests/streaming/test_warm_reuse.py`, `scripts/benchmarks/service_latency.py`, `scripts/qa/studio_controls_test.py` and `scripts/qa/studio_human_test.py`. The three scripts were not rewritten or run in this measurement.
- The names of `service.py` in the allow list of `tests/test_size_ratchet.py` move or leave.
- The wire protocol messages do not change.
