"""Small, dependency-free Named Binary Tag reader.

Only reading is implemented deliberately: this sandbox must never rewrite a world or
invoke a Minecraft data fixer.
"""

from __future__ import annotations

import gzip
import io
import struct
from typing import BinaryIO, Any


class NBTError(ValueError):
    pass


class NBTReader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream

    def _read(self, size: int) -> bytes:
        value = self.stream.read(size)
        if len(value) != size:
            raise NBTError(f"Unexpected end of NBT data (wanted {size} bytes)")
        return value

    def _number(self, fmt: str) -> int | float:
        return struct.unpack(">" + fmt, self._read(struct.calcsize(fmt)))[0]

    def _string(self) -> str:
        size = self._number("H")
        return self._read(size).decode("utf-8", errors="replace")

    def payload(self, tag: int) -> Any:
        if tag == 1:
            return self._number("b")
        if tag == 2:
            return self._number("h")
        if tag == 3:
            return self._number("i")
        if tag == 4:
            return self._number("q")
        if tag == 5:
            return self._number("f")
        if tag == 6:
            return self._number("d")
        if tag == 7:
            return self._read(self._number("i"))
        if tag == 8:
            return self._string()
        if tag == 9:
            item_tag = self._number("B")
            size = self._number("i")
            if size < 0:
                raise NBTError("Negative NBT list size")
            return [self.payload(item_tag) for _ in range(size)]
        if tag == 10:
            result: dict[str, Any] = {}
            while True:
                child_tag = self._number("B")
                if child_tag == 0:
                    return result
                # Keep these reads separate: assignment evaluates its right-hand side
                # before the subscription target in Python.
                child_name = self._string()
                result[child_name] = self.payload(child_tag)
        if tag == 11:
            size = self._number("i")
            return [self._number("i") for _ in range(size)]
        if tag == 12:
            size = self._number("i")
            return [self._number("q") for _ in range(size)]
        raise NBTError(f"Unsupported NBT tag type {tag}")

    def root(self) -> tuple[str, Any]:
        tag = self._number("B")
        if tag == 0:
            raise NBTError("NBT root cannot be TAG_End")
        name = self._string()
        return name, self.payload(tag)


def loads(data: bytes, *, compressed: bool = False) -> dict[str, Any]:
    if compressed:
        data = gzip.decompress(data)
    _, root = NBTReader(io.BytesIO(data)).root()
    if not isinstance(root, dict):
        raise NBTError("NBT root is not a compound")
    return root
