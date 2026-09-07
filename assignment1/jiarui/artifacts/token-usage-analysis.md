# Token usage: compaction vs. no compaction

Both runs solve the same vendored SWE-bench instance, `django__django-15368`,
with `deepseek/deepseek-v4-flash-0731`, the submit skill enabled, and a step
limit of 100. They differ only in whether compaction is configured.

| Run | Trajectory |
| --- | --- |
| No compaction | `artifacts/part1-swebench-trajectory.json` |
| Compaction (`COMPACT_THRESHOLD=6000`, `COMPACT_KEEP_RECENT=3`, `COMPACT_MAX_TOKENS=1024`) | `artifacts/part2-compaction-trajectory.json` |

Both produced a patch that passes the instance's tests: FAIL_TO_PASS 1/1 and
PASS_TO_PASS 29/29, `RESOLVED: yes`.

## Measurements

| | No compaction | Compaction | Change |
| --- | ---: | ---: | ---: |
| ReAct steps | 51 | 36 | −29% |
| Action prompt tokens | 721,298 | 185,442 | −74% |
| Action completion tokens | 11,832 | 10,591 | −10% |
| Compaction prompt tokens | 0 | 31,974 | — |
| Compaction completion tokens | 0 | 11,085 | — |
| **Total tokens** | **733,130** | **239,092** | **−67%** |
| Largest single prompt | 20,791 | 8,884 | −57% |
| First prompt | 1,457 | 1,457 | 0% |
| Compaction calls | 0 | 12 | — |

## What the numbers say

**The saving comes from the shape of the growth, not from doing less work.**
Both runs start at exactly 1,457 prompt tokens, because compaction cannot
trigger until there is history to compact. Without compaction the prompt grows
monotonically to 20,791 tokens, and since every step re-sends the whole
transcript, the total is roughly the area under that curve — quadratic in the
number of steps. Compaction caps the curve: the largest prompt is 8,884 tokens,
and the total falls by 74% even though the run still performed 34 `execute`
calls against the sandbox.

**Compaction pays for itself by a wide margin.** The 12 summarization calls cost
43,059 tokens, 18% of the compacted run's total. They removed 535,856 action
prompt tokens. Net saving is 494,038 tokens, about 12 times the overhead.

**Per-compaction reductions vary a lot**, from −70% to −4%:

```
step  8:  6,114 ->  2,456  (-60%)
step 14:  6,229 ->  3,011  (-52%)
step 21: 10,068 ->  8,106  (-19%)
step 22:  8,816 ->  8,465  ( -4%)
step 25:  6,195 ->  1,867  (-70%)
step 31:  6,055 ->  1,893  (-69%)
```

The size of a reduction is set by what the three retained steps happen to
contain, not by how much history was dropped. A step that read a large file
carries an observation truncated at 10,000 characters, so retaining three of
those leaves a floor of several thousand tokens that no amount of summarizing
can go below. That floor is why four compactions ended above the 6,000-token
threshold: the requirement to retain the most recent complete steps outranks
the threshold, so compaction reduces the prompt as far as it legitimately can
and no further. When the retained steps are short commands instead, the same
mechanism cuts 70%.

**Fewer steps was not a goal, and is probably not a compaction effect.** The
compacted run finished in 36 steps against 51. Most of that gap is the 13
steps the uncompacted run spent producing replies with no tool call at all,
which the harness discards; the compacted run had none. Model sampling is not
deterministic and these are single runs, so the step difference should not be
read as evidence that a shorter context makes the model more decisive. The
token measurements are the reliable part: they follow directly from prompt
size, which compaction controls by construction.

**One visible side effect.** The compacted run called `invoke_skill` twice
where the uncompacted run called it once. The skill's text was summarized away
after it was first read, so the agent fetched it again when it came time to
submit. That is compaction and progressive disclosure both working as intended:
the transcript is not treated as sacred, and anything dropped can be recovered
from its source.

## Caveat on the estimate

The threshold is compared against an estimate, `len(json.dumps(messages)) // 4`,
which counts only the message list. Measured against the `usage.prompt_tokens`
the API reports for the same requests, this underestimates in every one of 51
samples, averaging 0.88 of the true count and reaching 0.95 at the sizes that
matter here. The tool schemas sent alongside the messages are the bulk of the
gap. So `COMPACT_THRESHOLD=6000` fires at roughly 6,300 real tokens: compaction
is slightly later than the number suggests, never earlier.
