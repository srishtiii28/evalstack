"""The human-labelled set a judge is measured against.

Small and deliberate rather than large and casual. A gold set is the only thing
standing between "the judge said so" and "we checked", so every item carries the
reasoning behind its label — a gold set nobody can audit is just a second
unvalidated opinion.

``author_family`` exists so the same set can drive the self-preference probe:
without knowing who produced each output, you cannot ask whether the judge is
kinder to its own kind.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from evalforge.hashing import content_hash


class JudgeExample(BaseModel):
    """One item a judge is asked about, with the verdict a human gave it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    agent_summary: str = ""
    expected_behaviour: str = ""
    human_verdict: str = Field(min_length=1)
    #: Why the human decided that. Present so a disputed label can be argued
    #: with rather than merely counted.
    rationale: str = ""
    #: Which model family produced the output, for the self-preference probe.
    author_family: str = ""


class GoldSet(BaseModel):
    """A versioned collection of human-labelled examples."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    examples: tuple[JudgeExample, ...] = Field(min_length=1)

    @property
    def content_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))

    @property
    def verdicts(self) -> tuple[str, ...]:
        return tuple(example.human_verdict for example in self.examples)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted({example.human_verdict for example in self.examples}))

    def label_balance(self) -> dict[str, int]:
        """How many examples carry each label.

        Worth looking at before trusting an agreement score: a set that is 90%
        one label makes raw accuracy meaningless and κ noisy.
        """
        counts: dict[str, int] = {}
        for example in self.examples:
            counts[example.human_verdict] = counts.get(example.human_verdict, 0) + 1
        return dict(sorted(counts.items()))

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = "".join(f"{example.model_dump_json()}\n" for example in self.examples)
        path.write_text(lines, encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path, *, name: str = "", version: str = "v1") -> Self:
        if not path.is_file():
            raise FileNotFoundError(f"no gold set at {path}")
        examples = tuple(
            JudgeExample.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if not examples:
            raise ValueError(f"gold set at {path} contains no examples")
        return cls(name=name or path.stem, version=version, examples=examples)
