from __future__ import annotations

from typing import Dict, Any, Optional, TextIO
import csv, datetime, pathlib, os


class CSVLogger:
    """
    >>> logger = CSVLogger("runs/exp1_metrics.csv")
    >>> logger({"epoch":0,"train_loss":1.23})
    """

    def __init__(self, file_path: str, newline: str = "") -> None:
        self.path = pathlib.Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.fp: TextIO = self.path.open("a", newline=newline, encoding="utf-8")
        self.writer: Optional[csv.writer] = None   # 延遲到第一次寫入決定欄位順序

    # ------------------------------------------------------------------
    def __call__(self, metrics: Dict[str, Any]) -> None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row_dict = {"time": ts, **metrics}

        if self.writer is None:           
            self.fieldnames = list(row_dict.keys())
            self.writer = csv.DictWriter(self.fp, fieldnames=self.fieldnames)
            if self.fp.tell() == 0:       
                self.writer.writeheader()

        self.writer.writerow(row_dict)
        self.fp.flush()                  

    # ------------------------------------------------------------------
    def close(self) -> None:
        if not self.fp.closed:
            self.fp.close()

    def __del__(self):
        self.close()
