"""Chunked CSV reader; optional ID-to-path mapping replaces a dataset registry."""

import csv
from collections.abc import Iterator, Mapping
from pathlib import Path

import pandas as pd


class CsvDatasetReader:
    def __init__(self, paths: Mapping[str, str | Path] | None = None) -> None:
        self._paths = dict(paths or {})

    def _path(self, dataset_id: str) -> Path:
        return Path(self._paths.get(dataset_id, dataset_id)).resolve()

    def total_rows(self, dataset_id: str) -> int:
        with self._path(dataset_id).open(newline="", encoding="utf-8-sig") as handle:
            rows = (
                row
                for row in csv.reader(handle)
                if row and not (len(row) == 1 and not row[0].strip())
            )
            header = next(rows, [])
            if len(set(header)) != len(header) or not (
                {"customer_id", "customerID"} & set(header)
            ):
                raise ValueError(
                    "CSV requires unique columns and customer_id or customerID"
                )
            if {"customer_id", "customerID"} <= set(header):
                raise ValueError("CSV must contain one unambiguous customer ID column")
            return sum(1 for _ in rows)

    def chunks(self, dataset_id: str, chunk_size: int) -> Iterator[pd.DataFrame]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        with pd.read_csv(
            self._path(dataset_id),
            chunksize=chunk_size,
            dtype={"customer_id": "string", "customerID": "string"},
        ) as reader:
            for frame in reader:
                if "customer_id" not in frame and "customerID" in frame:
                    frame = frame.rename(columns={"customerID": "customer_id"})
                # Prepared train/val files include the label; inference never consumes it.
                yield frame.drop(columns=["Churn"], errors="ignore")
