"""Create token-safe batches from Spark DataFrame rows."""

from collections.abc import Iterable, Iterator
from functools import cache, partial
from typing import Any

import tiktoken
from pyspark.sql import DataFrame, Row
from via_mle.logging import get_logger

from feedbecky.config.base_config import conf
from feedbecky.llm_processing.batch_models import BatchItem
from feedbecky.utils.timing import log_duration

logger = get_logger(logger_name=__name__)

BatchLimits = tuple[int, int, int | None]
Separators = tuple[str, str]


@cache
def _get_encoding(encoding_name: str) -> Any:
    """Return a cached tokenizer encoding."""
    return tiktoken.get_encoding(encoding_name)


class BatchCreator:
    """Build batches constrained by token and row limits."""

    def __init__(self, config: Any) -> None:
        """Initialize the batch creator with component configuration."""
        self.config = config
        self.environment_config = conf

    @log_duration
    def execute(
        self,
        dataset: DataFrame,
        fixed_tokens: int,
    ) -> list[BatchItem]:
        """Returns one BatchItem per approximate, token-safe batch."""
        self.fixed_tokens = fixed_tokens
        self.token_budget = (
            self.environment_config.current_env_deployments[self.config.gpt_version][1]
            / 60
            * self.config.time_per_window
            * 0.9
            - self.fixed_tokens
        )
        self.n_batch_rows = getattr(self.config, "n_batch_rows", None)
        self.variable_token_count_col = self.config.variable_token_count_col
        self.column_separator = getattr(self.config, "column_separator", " ")
        self.row_separator = getattr(self.config, "row_separator", "\n")
        self.unique_key_col = getattr(self.config, "unique_key_col", None)

        if self.token_budget <= 0:
            raise ValueError(
                f"fixed_tokens ({self.fixed_tokens}) must be smaller than the token limit "
                f"(token_budget={self.token_budget:.0f} after subtracting fixed tokens)"
            )
        value_indexes = list(range(len(dataset.columns)))
        assign_batches = partial(
            self._assign_batches,
            value_indexes=value_indexes,
            encoding=self._get_encoding(),
        )
        logger.info(
            "Collecting rows for batch creation: columns=%s, token_budget=%.2f, max_rows_per_batch=%s",
            dataset.columns,
            self.token_budget,
            self.n_batch_rows,
        )
        dataset_rows = dataset.toLocalIterator()
        batch_tuples = list(assign_batches(dataset_rows))
        logger.info("Created %d batch tuples", len(batch_tuples))
        batches = [
            BatchItem(
                batch_id=batch_id,
                batch_rows=batch_rows,
                batch_tokens=batch_tokens,
                row_separator=row_separator,
                id_list=id_list,
                rows=rows,
            )
            for batch_id, (batch_rows, batch_tokens, row_separator, id_list, rows) in enumerate(
                batch_tuples, 1
            )
        ]
        logger.info("Created %d batches", len(batches))
        return batches

    def _assign_batches(
        self,
        rows: Iterable[Row],
        value_indexes: list[int],
        encoding: Any,
    ) -> Iterator[tuple[int, int, str, list[str], list[str]]]:
        """Group rows into batches within token and row limits."""
        batch_tokens = 0
        batch: list[str] = []
        id_list: list[str] = []

        for row in rows:
            values = list(row)
            row_text = self.column_separator.join(str(values[index]) for index in value_indexes)
            token_text = self.row_separator.join(
                str(values[index]) for index in value_indexes if values[index] is not None
            )
            remaining_tokens = self._apply_output_token_policy(len(encoding.encode(token_text)))

            while True:
                chunk_tokens = min(remaining_tokens, self.token_budget)
                exceeds_rows = self.n_batch_rows is not None and len(batch) >= self.n_batch_rows
                exceeds_tokens = bool(batch) and batch_tokens + chunk_tokens > self.token_budget

                if exceeds_rows or exceeds_tokens:
                    yield self._make_tuple(
                        batch_tokens,
                        self.fixed_tokens,
                        batch,
                        self.row_separator,
                        id_list,
                    )
                    batch_tokens = 0
                    batch = []
                    id_list = []
                    continue

                batch.append(row_text)
                if self.unique_key_col is not None:
                    id_list.append(str(row[self.unique_key_col]))
                batch_tokens += chunk_tokens
                remaining_tokens -= chunk_tokens

                if remaining_tokens <= 0:
                    break

        if batch:
            yield self._make_tuple(
                batch_tokens, self.fixed_tokens, batch, self.row_separator, id_list
            )

    def _get_encoding(self) -> Any:
        """Return the tokenizer configured for this component."""
        encoding_name = getattr(getattr(self.config, "encoding", None), "name", "cl100k_base")
        return _get_encoding(encoding_name)

    def _apply_output_token_policy(self, input_tokens: int) -> int:
        """Add configured output tokens to the input token estimate."""
        if getattr(self.config, "variable_output_token_method", "fixed") == "input":
            return input_tokens * 2
        return input_tokens + getattr(self.config, "variable_output_token_count", 0)

    def _make_tuple(
        self,
        batch_tokens: int,
        fixed_tokens: int,
        batch: list[str],
        row_separator: str,
        id_list: list[str],
    ) -> tuple[int, int, str, list[str], list[str]]:
        """Create a batch tuple containing metadata and serialized rows."""
        return (
            len(batch),
            batch_tokens + fixed_tokens,
            row_separator,
            id_list,
            batch,
        )
