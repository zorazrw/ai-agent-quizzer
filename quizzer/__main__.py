"""CLI: `python -m quizzer locate 1.1` or `python -m quizzer run 1.1`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quizzer.locate import DEFAULT_FINISHED_DIR, DEFAULT_ORIGINAL_DIR
from quizzer.locate import format_markdown as format_locate
from quizzer.locate import locate_module
from quizzer.quiz import (
    DEFAULT_MODEL,
    DEFAULT_QUESTION_COUNT,
    format_quiz,
    generate_quiz,
    load_located,
)
from quizzer.testing_quiz import generate_testing_quiz


def _positive_question_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if count < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return count


def _add_tree_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--finished",
        default=DEFAULT_FINISHED_DIR,
        help=f"Filled-in tree (default: {DEFAULT_FINISHED_DIR})",
    )
    parser.add_argument(
        "--original",
        default=DEFAULT_ORIGINAL_DIR,
        help=f"Unfilled scaffold (default: {DEFAULT_ORIGINAL_DIR})",
    )
    parser.add_argument(
        "--assignment",
        type=Path,
        default=None,
        help="ASSIGNMENT.md with this module's instructions (default: original then finished)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--no-lm-filter",
        action="store_true",
        help="Skip the LM snippet filter (rule-based locate only)",
    )


def _add_question_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-n",
        "--questions",
        type=_positive_question_count,
        default=DEFAULT_QUESTION_COUNT,
        help=f"How many quiz questions to generate (default: {DEFAULT_QUESTION_COUNT})",
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("md", "json"), default="md")
    parser.add_argument("-o", "--out", type=Path)


def _locate_from_args(args: argparse.Namespace):
    return locate_module(
        args.module,
        args.finished,
        args.original,
        assignment_md=args.assignment,
        filter_with_lm=not args.no_lm_filter,
        model=args.model,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quizzer",
        description=(
            "Locate original vs finished TODO implementations, then quiz them "
            "with claude-sonnet-4-6."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    locate = sub.add_parser("locate", help="Step 1: extract before/after snippets")
    locate.add_argument("module")
    _add_tree_args(locate)
    _add_output_args(locate)

    quiz = sub.add_parser("quiz", help="Step 2: generate Q+A from locate JSON")
    quiz.add_argument("locate_json", type=Path)
    quiz.add_argument("--model", default=DEFAULT_MODEL)
    _add_question_arg(quiz)
    _add_output_args(quiz)

    run = sub.add_parser("run", help="Locate a module, then generate the quiz")
    run.add_argument("module")
    _add_tree_args(run)
    _add_question_arg(run)
    _add_output_args(run)
    run.add_argument(
        "--save-locate",
        type=Path,
        help="Write the locate JSON as well",
    )

    testing = sub.add_parser(
        "testing",
        help="Rule-based quiz: which module each public test targets",
    )
    testing.add_argument(
        "--finished",
        default=DEFAULT_FINISHED_DIR,
        help=f"Filled-in tree (default: {DEFAULT_FINISHED_DIR})",
    )
    testing.add_argument(
        "--original",
        default=DEFAULT_ORIGINAL_DIR,
        help=f"Unfilled scaffold (default: {DEFAULT_ORIGINAL_DIR})",
    )
    testing.add_argument(
        "--assignment",
        type=Path,
        default=None,
        help="ASSIGNMENT.md used to map tests to modules",
    )
    testing.add_argument(
        "--tests",
        type=Path,
        default=None,
        help="Public tests directory (default: <finished>/tests)",
    )
    _add_output_args(testing)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "locate":
            result = _locate_from_args(args)
            payload = result.to_dict()
            if args.out:
                args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            if args.format == "json":
                print(json.dumps(payload, indent=2))
            else:
                print(format_locate(result), end="")
            return 0

        if args.command == "testing":
            quiz = generate_testing_quiz(
                args.finished,
                args.original,
                tests_dir=args.tests,
                assignment_md=args.assignment,
            )
        elif args.command == "quiz":
            located = load_located(args.locate_json)
            quiz = generate_quiz(
                located, model=args.model, question_count=args.questions
            )
        else:
            located_result = _locate_from_args(args)
            if args.save_locate:
                args.save_locate.write_text(
                    json.dumps(located_result.to_dict(), indent=2), encoding="utf-8"
                )
            quiz = generate_quiz(
                located_result, model=args.model, question_count=args.questions
            )

        if args.out:
            args.out.write_text(json.dumps(quiz, indent=2), encoding="utf-8")
        print(json.dumps(quiz, indent=2) if args.format == "json" else format_quiz(quiz))
        return 0
    except (FileNotFoundError, LookupError, RuntimeError, ValueError) as exc:
        print(f"quizzer: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
