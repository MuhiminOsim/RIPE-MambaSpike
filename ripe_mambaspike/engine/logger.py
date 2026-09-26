"""CSV metric logging."""

import csv
import os

__all__ = ["CSVLogger"]


class CSVLogger:
    """Append one row per epoch to ``<run_dir>/log.csv``.

    The header is written only when the file is created, so a resumed run
    appends to the existing history instead of truncating it.
    """

    def __init__(self, run_dir: str, fieldnames, filename: str = "log.csv"):
        self.path = os.path.join(run_dir, filename)
        self.fieldnames = list(fieldnames)
        os.makedirs(run_dir, exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as handle:
                csv.DictWriter(handle, fieldnames=self.fieldnames).writeheader()

    def append(self, row: dict):
        """Write one row; missing keys are blank, unknown keys are dropped."""
        clean = {key: row.get(key, "") for key in self.fieldnames}
        with open(self.path, "a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.fieldnames).writerow(clean)
