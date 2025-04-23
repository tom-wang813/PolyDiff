# PolyDiff/dataset/text.py
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Iterator, List

import torch
from torch.utils.data import get_worker_info

from PolyDiff.configs import model_config
from PolyDiff.dataset.base import BaseIterableDataset, register_dataset

PAD = model_config.PAD_TOKEN_ID


# --------------------------------------------------------------------------- #
@register_dataset("text_diffusion")
class TextDiffusionDataset(BaseIterableDataset):
    """
    Streaming text corpus → token ids

    Parameters
    ----------
    data_dir : str | Path
        Root directory containing `<split>* .txt`.
    split : str
        Dataset split prefix, e.g. `"train"` / `"val"` / `"test"`.
    tokenizer : Callable[[str], List[int]]
        Any callable mapping string → list[int].
    shuffle_files : bool, default True
        Randomise file order each epoch.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        tokenizer: Callable[[str], List[int]],
        *,
        shuffle_files: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.shuffle_files = shuffle_files

        self.files = sorted(self.data_dir.glob(f"{split}*.txt"))
        if not self.files:
            raise FileNotFoundError(f"No '{split}*.txt' in {data_dir}")

    # ------------------------------------------------------------------ #
    # iterator implementation
    # ------------------------------------------------------------------ #
    def _iter_data(self) -> Iterator[torch.Tensor]:
        files = self.files.copy()
        if self.shuffle_files:
            random.shuffle(files)

        w = get_worker_info()
        if w is not None:                              # split files across workers
            files = files[w.id :: w.num_workers]

        for file in files:
            yield from self._iter_file(file)

    # ----------------------------- helpers ----------------------------- #
    def _iter_file(self, file: Path) -> Iterator[torch.Tensor]:
        with file.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.rstrip("\n")
                if not line:
                    continue
                ids = self.tokenizer(
                    line,
                    truncation=True,
                    max_length=model_config.MAX_SEQ_LENGTH,
                )
                yield torch.tensor(ids, dtype=torch.long)


