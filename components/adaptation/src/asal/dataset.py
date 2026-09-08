"""Canonical dataset format, hashing and deterministic splits.

A fine-tuning dataset must be addressable: two runs that claim to use "v2" must
be able to prove they used the same v2. That is what the content hash is for.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass(slots=True)
class Example:
    """One supervised example. ``prompt`` in, ``completion`` expected out."""

    example_id: str
    prompt: str
    completion: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        payload = json.dumps({"p": self.prompt, "c": self.completion},
                             sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Example:
        return cls(example_id=d["example_id"], prompt=d["prompt"],
                   completion=d["completion"], tags=list(d.get("tags", [])),
                   metadata=dict(d.get("metadata", {})))


@dataclass(slots=True)
class Dataset:
    version: str
    examples: list[Example] = field(default_factory=list)
    parent_version: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    @property
    def content_hash(self) -> str:
        """Order-independent digest over example content."""
        joined = "|".join(sorted(e.content_hash for e in self.examples))
        return hashlib.sha256(joined.encode()).hexdigest()

    def __len__(self) -> int:
        return len(self.examples)

    def split(self, *, eval_fraction: float = 0.2,
              seed: int = 1234) -> tuple[Dataset, Dataset]:
        """Hash-based split, so membership is stable as the dataset grows.

        A shuffle would move examples between train and eval whenever new data
        arrives, silently invalidating every comparison against an earlier
        adapter. That failure is invisible unless you look for it.
        """
        train, held = [], []
        for e in self.examples:
            h = hashlib.sha256(f"{seed}:{e.example_id}".encode()).hexdigest()
            bucket = int(h[:8], 16) / 0xFFFFFFFF
            (held if bucket < eval_fraction else train).append(e)
        return (
            Dataset(f"{self.version}:train", train, self.parent_version,
                    notes="train split"),
            Dataset(f"{self.version}:eval", held, self.parent_version,
                    notes="eval split"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "version": self.version,
                "parent_version": self.parent_version,
                "created_at": self.created_at, "notes": self.notes,
                "content_hash": self.content_hash, "count": len(self.examples),
                "examples": [e.to_dict() for e in self.examples]}

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return p

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        d = json.loads(Path(path).read_text())
        return cls(version=d["version"],
                   examples=[Example.from_dict(e) for e in d["examples"]],
                   parent_version=d.get("parent_version"),
                   created_at=d.get("created_at", ""),
                   notes=d.get("notes", ""))

    def to_jsonl(self) -> str:
        """mlx-lm expects one JSON object per line with a `text` field."""
        return "\n".join(
            json.dumps({"text": f"{e.prompt}\n{e.completion}"}, sort_keys=True)
            for e in self.examples
        )


def seed_dataset() -> Dataset:
    """A small cycle-1 dataset: structured extraction from short notices."""
    rows = [
        ("Ticket 4417 was closed by the platform team on Tuesday.",
         '{"ticket": 4417, "status": "closed", "team": "platform"}', ["closed"]),
        ("Ticket 5120 remains open and is assigned to the storage team.",
         '{"ticket": 5120, "status": "open", "team": "storage"}', ["open"]),
        ("Ticket 3301 was escalated to the networking team yesterday.",
         '{"ticket": 3301, "status": "escalated", "team": "networking"}',
         ["escalated"]),
        ("Ticket 2288 is blocked pending review by the security team.",
         '{"ticket": 2288, "status": "blocked", "team": "security"}',
         ["blocked"]),
        ("Ticket 9004 was closed by the storage team this morning.",
         '{"ticket": 9004, "status": "closed", "team": "storage"}', ["closed"]),
        ("Ticket 1150 is open with the platform team.",
         '{"ticket": 1150, "status": "open", "team": "platform"}', ["open"]),
        ("Ticket 7732 was escalated to the security team.",
         '{"ticket": 7732, "status": "escalated", "team": "security"}',
         ["escalated"]),
        ("Ticket 6600 is blocked awaiting the networking team.",
         '{"ticket": 6600, "status": "blocked", "team": "networking"}',
         ["blocked"]),
    ]
    return Dataset(
        version="v1",
        examples=[Example(f"v1-{i:03d}", p, c, t)
                  for i, (p, c, t) in enumerate(rows)],
        notes="cycle 1: ticket status extraction",
    )


__all__ = ["Dataset", "Example", "SCHEMA_VERSION", "seed_dataset"]
