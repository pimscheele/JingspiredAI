"""Example data loader."""

from typing import Any

from pyspark.sql import DataFrame, SparkSession


def example_loader(
    spark: SparkSession,
    config: dict[str, Any],
) -> DataFrame:
    """Load a limited set of example records."""
    return spark.table(config["input_table_path"]).limit(10)


def example_loader_2(
    spark: SparkSession,
    config: dict[str, Any],
) -> DataFrame:
    """Load a limited set of example records."""
    return spark.table(config["input_table_path"]).limit(10)
