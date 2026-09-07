# Observation design experiment (Part 2, section 5)

## Setup

Both conditions use the same `ChessAgent`, the same shared ReAct loop, the same
repaired server (`artifacts/fix.patch`) and the same step limit (`STEPS=100`).
The only difference is what `format_chess_state` puts into the initial and
post-move observations, plus the one prompt line that refers to it:

- **Condition A — board state only.** The `legal_moves:` line is omitted from
  every `<chess_state>` observation, and the chess prompt tells the model to
  infer a legal UCI move from the board.
- **Condition B — board state plus legal moves.** The provided `legal_moves`
  line and prompt wording are restored. This is what the submitted source uses.

Each condition was run once with `deepseek/deepseek-v4-flash-0731` and once with
`openai/gpt-oss-120b`.

| trajectory | condition | model |
| --- | --- | --- |
| `artifacts/part3-no-legal-moves-trajectory.json` | A | deepseek-v4-flash |
| `artifacts/part3-no-legal-moves-gptoss-trajectory.json` | A | gpt-oss-120b |
| `artifacts/part3-legal-moves-trajectory.json` (= `part3-trajectory.json`) | B | deepseek-v4-flash |
| `artifacts/part3-legal-moves-gptoss-trajectory.json` | B | gpt-oss-120b |

**Counting rules.** A "`play_move` call" is a tool call the model emitted whose
function name is `play_move`. A call is "rejected as illegal" when the server
answered with `<chess_error>… is not legal in the current position.</chess_error>`.
Two other error classes are reported separately and are *not* counted as illegal
moves, because the server never saw a move in either case: one malformed tool
name (` play_move`, with a leading space, rejected by the harness as an unknown
tool) and one transient transport error. Counts are taken over
every step of the run, not over the final prompt, because compaction removes
older history from the prompt.

## Results

| condition | model | ReAct steps | `play_move` calls | illegal-move rejections | invalid-move rate | reached `game_over: true`? |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A — board only | deepseek-v4-flash | 32 | 21 | 1 | **4.8 %** | **yes** (mate on ply 38, "Black wins") |
| A — board only | gpt-oss-120b | 100 | 48 | 15 | **31.2 %** | no (hit the 100-step limit) |
| B — board + `legal_moves` | deepseek-v4-flash | 43 | 40 | 0 | **0.0 %** | **yes** (run returned at step 43 of 100) |
| B — board + `legal_moves` | gpt-oss-120b | 100 | 33 | 2 | **6.1 %** | no (hit the 100-step limit) |

Other, non-illegal errors: A/deepseek had one transient transport error
(`Server disconnected without sending a response.`) at call 10 of 21, which it
recovered from; B/deepseek had one ` play_move` unknown-tool error, likewise
recovered from and reissued correctly on the next step.

Both deepseek runs reached a terminal state; both gpt-oss runs exhausted the
step limit. `ChessAgent.finished` is set only when a `play_move` observation
carries `game_over: true`, and `run_chess_agent` writes the result file only if
`agent.run()` returns without raising `StepLimitError` — so a run that stops
short of its 100 steps is a run that ended the game. A/deepseek stopped at 32
and B/deepseek at 43.

Supporting numbers: steps in which the model returned no tool call at all were
11/32 (A/deepseek), 52/100 (A/gpt-oss), 3/43 (B/deepseek) and 67/100
(B/gpt-oss). Games reached move 19, 35, 39 and 32 respectively.

## Analysis

**Showing `legal_moves` cuts the invalid-move rate for both models**, from
31.2 % to 6.1 % for gpt-oss-120b and from 4.8 % to 0 % for deepseek-v4-flash.
The direction is the same for both, and the effect is much larger for the weaker
chess player.

The reason is that the two conditions ask the model for different things. In
condition B the task is *selection*: the answer is a token that appears verbatim
in the observation, so the model only has to pick one and copy it. In condition A
the task is *derivation*: the model has to read an ASCII board, reconstruct the
position, apply movement rules, check that its own king is not left in check,
and only then emit UCI. Every one of those steps can fail independently, and the
observation contains no signal that would catch a mistake before the server does.
The failures we saw match that: A/gpt-oss's rejections are moves that are
plausible in the abstract but not legal in the actual position (`e1f2`, `e1f1`
with the square occupied or attacked), i.e. board-reading errors rather than
formatting errors.

**The gap between the models is larger than the gap between the conditions.**
deepseek-v4-flash was already at 4.8 % without the list, so for it the list
mostly removed a residual error; gpt-oss-120b at 31.2 % was failing roughly one
move in three, and the list is what made it usable. That is what one would
expect if the list substitutes for exactly the capability the two models differ
in — reliable position tracking over a long context — and does nothing for the
capability they share.

**The list also changes how efficiently the step budget is spent.** In condition
B deepseek converted 40 of 43 steps into `play_move` calls, and got 39 legal
moves out of the run — a longer game (move 39) from fewer steps than either
gpt-oss run managed in 100. gpt-oss spent most of its steps producing no tool
call at all (52/100 in A, 67/100 in B), which is a separate, endpoint-level
problem: the SAIL endpoint frequently fails to parse this model's tool calls and
leaks the raw harmony markup into `message.content`. That inflates the step cost
per move for gpt-oss in both conditions and is why its condition-B run made
*fewer* calls than its condition-A run despite a lower error rate — it is noise
in the call count, not evidence against the list.

**Caveat on `game_over`.** The terminal-state column splits by model, not by
condition: both deepseek runs ended in a real game over, both gpt-oss runs ran
out of steps. That is a budget effect rather than a chess result — gpt-oss needs
about three ReAct steps per legal move on this endpoint (see the no-tool-call
counts above), so 100 steps buys it roughly 33 moves, which is not enough to
finish a game. The invalid-move rate is measured per call and is therefore the
comparison to read; it is unaffected by where a run was cut off.

Two further environment notes. The `ChessSandbox` default `deployment_timeout`
of 600 s is shorter than a game at ~40 s per White move; these runs used a
raised timeout (`--sandbox-timeout`, added locally to `run_chess_agent`) so the
sandbox would not expire mid-game. And `--result` has a hard default of
`artifacts/game-result.json` that `make run-chess-agent` never overrides, so each
of the four runs overwrote it: the `artifacts/game-result.json` submitted here is
the last run that finished a game, which is A/deepseek (mate on ply 38), not the
condition-B run that `artifacts/part3-trajectory.json` records.

**Takeaway.** Which information the observation carries is part of the action
space, not a cosmetic prompt choice. Moving legality checking from the model into
the observation converts an inference problem into a lookup, and buys back a
large fraction of the model's step budget from error recovery — most for the
model that needed it most.
