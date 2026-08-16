"""Loading local configuration from a .env file.

The behaviour worth protecting is precedence: an explicitly exported variable
must beat the file. Getting that backwards means an on-the-fly
``GROQ_API_KEY=... evalforge run`` is silently ignored in favour of a stale
file, which is an unpleasant way to lose half an hour.
"""

from __future__ import annotations

from pathlib import Path

from evalforge.config import load_env_file, parse_env_file


def test_simple_assignments_are_parsed() -> None:
    assert parse_env_file("GROQ_API_KEY=abc123\n") == {"GROQ_API_KEY": "abc123"}


def test_comments_and_blank_lines_are_ignored() -> None:
    text = "# a comment\n\n  \nKEY=value\n"

    assert parse_env_file(text) == {"KEY": "value"}


def test_an_export_prefix_is_accepted() -> None:
    assert parse_env_file("export KEY=value\n") == {"KEY": "value"}


def test_surrounding_quotes_are_stripped() -> None:
    text = "SINGLE='one'\nDOUBLE=\"two\"\n"

    assert parse_env_file(text) == {"SINGLE": "one", "DOUBLE": "two"}


def test_a_quoted_value_keeps_its_hash() -> None:
    # A key really can contain '#', so quoting must be respected literally.
    assert parse_env_file('KEY="abc#def"\n') == {"KEY": "abc#def"}


def test_an_unquoted_value_drops_a_trailing_comment() -> None:
    assert parse_env_file("KEY=value # explanation\n") == {"KEY": "value"}


def test_values_containing_equals_survive() -> None:
    assert parse_env_file("KEY=a=b=c\n") == {"KEY": "a=b=c"}


def test_empty_values_are_kept() -> None:
    assert parse_env_file("KEY=\n") == {"KEY": ""}


def test_lines_without_an_assignment_are_skipped() -> None:
    assert parse_env_file("this is not a setting\nKEY=value\n") == {"KEY": "value"}


def test_whitespace_around_names_and_values_is_trimmed() -> None:
    assert parse_env_file("  KEY  =  value  \n") == {"KEY": "value"}


# -- loading into an environment -----------------------------------------


def test_values_are_loaded_into_the_environment(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("GROQ_API_KEY=from-file\n", encoding="utf-8")
    environ: dict[str, str] = {}

    applied = load_env_file(path, environ=environ)

    assert applied == ("GROQ_API_KEY",)
    assert environ["GROQ_API_KEY"] == "from-file"


def test_an_existing_variable_is_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("GROQ_API_KEY=from-file\n", encoding="utf-8")
    environ = {"GROQ_API_KEY": "from-shell"}

    applied = load_env_file(path, environ=environ)

    # An explicit export must win, or overriding a key for one run is impossible.
    assert applied == ()
    assert environ["GROQ_API_KEY"] == "from-shell"


def test_override_is_available_when_asked_for(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("KEY=from-file\n", encoding="utf-8")
    environ = {"KEY": "from-shell"}

    load_env_file(path, environ=environ, override=True)

    assert environ["KEY"] == "from-file"


def test_an_empty_existing_value_is_treated_as_unset(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("KEY=from-file\n", encoding="utf-8")
    environ = {"KEY": ""}

    load_env_file(path, environ=environ)

    assert environ["KEY"] == "from-file"


def test_a_missing_file_is_not_an_error(tmp_path: Path) -> None:
    environ: dict[str, str] = {}

    # CI sets real environment variables and has no .env; that is normal.
    assert load_env_file(tmp_path / "absent", environ=environ) == ()
    assert environ == {}


def test_an_unreadable_file_does_not_stop_a_run(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_bytes(b"\xff\xfe not utf-8 \xff")
    environ: dict[str, str] = {}

    assert load_env_file(path, environ=environ) == ()
