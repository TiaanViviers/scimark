from scimark.cli import build_parser


def test_build_parser_accepts_convert_math_mode() -> None:
    args = build_parser().parse_args(
        ["convert", "resources", "--out", "parsed", "--math-mode", "experimental"]
    )

    assert args.command == "convert"
    assert args.input_path == "resources"
    assert args.out == "parsed"
    assert args.math_mode == "experimental"


def test_build_parser_accepts_eval_math_command() -> None:
    args = build_parser().parse_args(
        ["eval-math", "resources", "--out", "math-eval", "--recursive", "--skip-convert"]
    )

    assert args.command == "eval-math"
    assert args.input_path == "resources"
    assert args.out == "math-eval"
    assert args.recursive is True
    assert args.skip_convert is True
