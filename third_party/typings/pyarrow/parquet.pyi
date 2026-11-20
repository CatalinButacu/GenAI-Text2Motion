from collections.abc import Iterator
from pathlib import Path
from typing import Any


class ParquetFile:
    def __init__(self, source: str | Path) -> None: ...
    def iter_batches(self, batch_size: int = ..., columns: list[str] | None = ...) -> Iterator[Any]: ...
