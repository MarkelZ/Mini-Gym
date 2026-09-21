"""Tiny CSV logger for training metrics."""

import csv
import os


class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def log(self, **row):
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()
