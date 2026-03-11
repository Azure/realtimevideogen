from __future__ import annotations

from unittest.mock import MagicMock


class NumPyMock(MagicMock):
    @property
    def __version__(self) -> str:
        return "1.26.4"

    @property
    def __name__(self) -> str:
        return "numpy"

    def get_sub_modules(self) -> dict[str, MagicMock]:
        return {
            "numpy": self,
            "numpy.typing": MagicMock(),
            "numpy.core": MagicMock(),
            "numpy.core.numeric": MagicMock(),
            "numpy.linalg": MagicMock(),
        }
