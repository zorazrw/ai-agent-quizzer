# Quizzer

Turn a filled-in assignment TODO into a short understanding quiz.

This repo is the quizzer tool plus exemplar assignment trees under `assignment1/`. Those trees are sample data, not the product: `original` is the unfilled scaffold, and `finished` is an exemplar submission.

From the repo root, defaults are `--finished assignment1/finished` and `--original assignment1/original`. Locate's LM filter and quiz both need a real `ANTHROPIC_API_KEY` in `.env` unless you pass `--no-lm-filter`.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
cp .env.example .env   # then put a real ANTHROPIC_API_KEY in .env
```

## Quick start

```bash
uv run quizzer run 1.1
uv run quizzer run 1.1 --questions 3 --format json -o quizzer/qa/1.1.json
```

Point at any other pair of trees with `--finished` and `--original`.

## Exemplar data

```
assignment1/
  original/    unfilled scaffold (TODO comments, stubs)
  finished/    exemplar filled-in submission
```

`finished` may contain a nested `.git`, `.venv`, or `.env` from a local assignment checkout. Those are gitignored so they are not uploaded. Do not commit API keys.

## Step 1. Locate: find before/after snippets

```bash
uv run quizzer locate 1.1
uv run quizzer locate 1.1.a --format json -o /tmp/1.1.json
uv run quizzer locate 1.2 --finished assignment1/finished --original assignment1/original
```

Markdown prints to **stdout**. `--format json -o PATH` writes the same data as JSON (step 2 reads that file). `1.1` includes `1.1.a` and `1.1.b`; `1.1.a` is only that sub-item.

After the TODO-comment match, locate asks the language model to keep only snippets (and only the lines) that match this module's instructions in `ASSIGNMENT.md`. That stops later work that lives in the same function — for example Part 2 compaction inside `build_prompt` — from leaking into a 1.1 quiz. Default spec file is `original/ASSIGNMENT.md`, then `finished/ASSIGNMENT.md`. Override with `--assignment PATH`. Use `--no-lm-filter` to skip the model (rule-based locate only; no API key).

```bash
uv run quizzer locate 1.1 --assignment assignment1/original/ASSIGNMENT.md
uv run quizzer locate 1.1 --no-lm-filter
```

Example (`uv run quizzer locate 1.2 --no-lm-filter`):

````text
# Module 1.2

Original: `/path/to/assignment1/original`
Finished: `/path/to/assignment1/finished`

## Spec

### 2. Run the ReAct loop

Implement the body of `Agent.run`.

> **TODO(1.2)**
> Run the ReAct loop. Orchestrate prompting, tool calls, and observations.
> Set `Agent.finished` when the task is done. If the agent exceeds
> `step_limit`, raise `StepLimitError`.

## Snippets

### `Agent.run` in `src/assignment/agent/base.py` (1.2)

```
            # TODO(1.2) Run the ReAct loop. ...
```

**Before** `src/assignment/agent/base.py:322-343`

```py
    def run(self) -> None:
        try:
            # TODO(1.2) Run the ReAct loop. ...
            raise NotImplementedError
        finally:
            ...
```

**After** `src/assignment/agent/base.py:427-464`

```py
    def run(self) -> None:
        try:
            while not self.finished:
                if self.steps_taken >= self.step_limit:
                    raise StepLimitError(...)
                self.maybe_compact_context()
                action = self.query_language_model()
                ...
        finally:
            ...
```
````

The JSON written by `-o /tmp/1.2.json` is the same locate result:

```json
{
  "module": "1.2",
  "finished_dir": "/path/to/assignment1/finished",
  "original_dir": "/path/to/assignment1/original",
  "spec": "### 2. Run the ReAct loop\n...",
  "assignment_md": "/path/to/assignment1/original/ASSIGNMENT.md",
  "snippets": [
    {
      "todo_ids": ["1.2"],
      "todo_comments": ["# TODO(1.2) Run the ReAct loop. ..."],
      "file": "src/assignment/agent/base.py",
      "symbol": "Agent.run",
      "before": {"start_line": 322, "end_line": 343, "code": "..."},
      "after": {"start_line": 427, "end_line": 464, "code": "..."}
    }
  ]
}
```

## Step 2. Quiz: generate questions, answers, and explanations

Needs a real `ANTHROPIC_API_KEY` in `.env` (same as locate's LM filter). Uses **claude-sonnet-4-6** (Anthropic Messages API) unless you pass `--model`.

```bash
uv run quizzer quiz /tmp/1.1.json
uv run quizzer quiz /tmp/1.1.json --questions 4
uv run quizzer quiz /tmp/1.1.json --format json -o /tmp/1.1-quiz.json
uv run quizzer quiz /tmp/1.1.json --model claude-sonnet-4-6
```

`--questions` / `-n` sets how many questions to write (default **2**).

## Public tests quiz

Rule-based; no API key. Reads `tests/` and asks which assignment module each test targets.

```bash
uv run quizzer testing
uv run quizzer testing --format json -o quizzer/qa/tests.json
```

Default tests dir is `<finished>/tests`. Override with `--tests PATH`.

## Read saved quizzes

JSON dumps in `quizzer/qa/` (for example `1.1.json`) can be read in the browser:

```bash
cd qa
python -m http.server
```

Then open http://127.0.0.1:8000/ — `index.html` lists every `*.json` in that folder.

## Tests

```bash
uv run pytest
```
