"""Orchestrate data loading, LLM inference, post-processing, and export."""

from __future__ import annotations

import importlib
import time
from typing import Any

from feedbecky.config.base_config import conf
from feedbecky.llm_processing.batch_processing import StandardProcessor
from feedbecky.utils.timing import log_duration
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import concat_ws, current_timestamp, format_string, lit, xxhash64
from via_mle.logging import get_logger

logger = get_logger(logger_name=__name__)


class GenericGenAIPipeline:
    """Run a configured generative-AI component pipeline."""

    def __init__(
        self,
        usecase: str,
        component: str,
        sub_usecase: str | None = None,
    ) -> None:
        """Initialize a pipeline for the selected use case and component."""
        self.spark = SparkSession.builder.getOrCreate()
        self.usecase = usecase
        self.usecase_output_path = self.usecase
        self.usecase_folder = self.usecase
        self.component = component
        self.sub_usecase = sub_usecase
        self.datasets: list[DataFrame] = []
        self.granularity_values: dict[str, Any] = {}
        self.invocation_params: dict[str, Any] = {}
        self.output_table_path: str = ""
        self._start_time = time.perf_counter()

        if sub_usecase:
            self.usecase_folder = f"{self.usecase}.{self.sub_usecase}"
            self.usecase_output_path = f"{self.usecase}_{self.sub_usecase}"

        module_prefix = f"feedbecky.use_cases.{self.usecase_folder}"

        config_module = importlib.import_module(
            f"feedbecky.config.use_cases.{self.usecase_folder}.config"
        )
        config_class = getattr(config_module, self.component)

        self.config = config_class()
        self.environment_config = conf

        prompt_module = importlib.import_module(f"{module_prefix}.prompt")
        self.prompt = getattr(prompt_module.PromptBase, self.config.prompt)

        pydantic_module = importlib.import_module(f"{module_prefix}.pydantic")
        self.pydantic = getattr(pydantic_module, self.config.pydantic)

        self.data_module = importlib.import_module(f"{module_prefix}.data")

        self.batch_processor: StandardProcessor = StandardProcessor(
            spark=self.spark, config=self.config
        )

    def prepare(self) -> None:
        """Loads the data, sets the invocation parameters, initializes the prompt template, creates a hash key, adds token count columns, and initializes the LLM."""
        self.load_data()
        self._set_invocation_params()
        self._init_prompt_template()
        self._create_hash_key()
        self._add_token_count_cols()
        self._init_llm()

    def predict(self) -> None:
        """Processes the batches to generate LLM classifications and combines the results."""
        self.output, self.total_api_cost = self.batch_processor.process_batches(
            input_sdf=self.input_sdf,
            invocation_params=self.invocation_params,
            llm_chain=self.llm_chain,
            fixed_tokens=self.fixed_tokens,
        )

    def export(self) -> None:
        """Exports the final results to the configured output table."""
        if self.output:
            self.output_sdf = self.spark.createDataFrame(self.output)
            self.post_processing()
            self.export_data()
        else:
            logger.warning("No data available to export")

    def execute(self) -> None:
        """Executes the entire pipeline: prepare, predict, and export."""
        self._create_config_dict()
        for granularity_values in self._load_granularity():
            if granularity_values:
                logger.info(f"Running granularity {granularity_values}")
            self.granularity_values = granularity_values
            self.prepare()
            input_is_empty = self.input_sdf.isEmpty()
            if input_is_empty:
                logger.warning("No data available for processing. Skipping this granularity.")
                continue
            self.predict()
            self.export()

        try:
            logger.info(
                f"Pipeline execution completed in {time.perf_counter() - self._start_time:.2f} seconds. Total tokens used: {self.total_api_cost:.2f}"
            )
        except Exception:
            logger.warning("Cost calculation failed. Check your pipeline!")

    def _create_config_dict(self) -> None:
        """Creates a configuration dictionary that combines environment and component configurations."""
        self.config_dict = {
            **self.environment_config.model_dump(),
            **self.config.model_dump(),
        }

        self.output_table_path = (
            f"{self.environment_config.uc_catalog}."
            f"{self.environment_config.uc_schema}."
            f"feedbecky_{self.config.output_table_name}"
        ).lower()
        self.config_dict["output_table_path_1"] = self.output_table_path

    def _load_granularity(self) -> list[dict[str, Any]]:
        """Loads the configured granularity values, or one unfiltered run."""
        if getattr(self.config, "granularity", False) is not True:
            return [{}]

        loader_name = self.config.granularity_loader
        loader = getattr(self.data_module, loader_name)
        self.granularity_sdf = loader(
            spark=self.spark,
            config=self.config_dict,
        )
        return [row.asDict(recursive=True) for row in self.granularity_sdf.collect()]

    @log_duration
    def load_data(self) -> Any:
        """Loads the input data based on the configured data loaders."""
        self.datasets = []
        for attr_name in dir(self.config):
            if attr_name.startswith("input_loader_"):
                loader_name = getattr(self.config, attr_name)
                logger.info(f"Attempting to load {loader_name}...")
                loader = getattr(self.data_module, loader_name)
                dataset_sdf = loader(
                    spark=self.spark,
                    config=self.config_dict,
                )
                for column, value in self.granularity_values.items():
                    if column in dataset_sdf.columns:
                        dataset_sdf = dataset_sdf.filter(dataset_sdf[column].eqNullSafe(value))
                    else:
                        logger.warning(
                            f"Column '{column}' not found in dataset '{loader_name}'. Skipping filter for this column."
                        )
                self.datasets.append(dataset_sdf)

    def _init_prompt_template(self) -> None:
        """Initializes the prompt template using the configured prompt and sets the fixed token count."""
        formatted_prompt = self.prompt_template.format(**self.invocation_params)
        self.prompt_template_tokens = len(self.config.encoding.encode(formatted_prompt))
        self.fixed_tokens = self.prompt_template_tokens + self.config.fixed_output_tokens

    def _init_llm(self) -> None:
        """Sets up the LLM settings and initializes the LLM chains."""
        self.llm: AzureChatOpenAI = AzureChatOpenAI(
            azure_ad_token_provider=conf.token_provider,
            azure_endpoint=self.config.openai_endpoint,
            api_version=self.config.open_api_version,  # type: ignore[call-arg]
            azure_deployment=self.environment_config.current_env_deployments[
                self.config.gpt_version
            ][0],
            #   reasoning_effort=self.config.reasoning_effort, # Disabled as it is incompatible with gpt 6.
            #   verbosity=self.config.verbosity, # Disabled as it is incompatible with gpt 6.
        )
        self.structured_llm: Any = self.llm.with_structured_output(
            self.pydantic, method="json_schema"
        )
        self.llm_chain: Any = self.prompt_template | self.structured_llm

    def _create_hash_key(self) -> None:
        """Creates a short unique hash key for each row in the input DataFrame based on the primary key columns."""
        if getattr(self.config, "unique_key_col", None):
            if self.config.unique_key_col not in self.datasets[0].columns:
                self.datasets[0] = self.datasets[0].withColumn(
                    self.config.unique_key_col, concat_ws("", *self.config.primary_key_cols)
                )
        if getattr(self.config, "hash_primary_key_cols", True):
            self.datasets[0] = self.datasets[0].withColumn(
                self.config.unique_key_col,
                format_string("%016x", xxhash64(self.config.unique_key_col)),
            )

    def _add_token_count_cols(self) -> None:
        """Select input columns for token counting during batch construction."""
        llm_processing_cols = list(self.config.llm_processing_input_cols)
        if getattr(self.config, "unique_key_col", None):
            llm_processing_cols += [self.config.unique_key_col]
        self.input_sdf = self.datasets[0].select(*llm_processing_cols)

    def _set_invocation_params(self) -> None:
        """Sets the invocation parameters for the prompt template, including granularity values and external input parameters."""
        external_input_params: dict[str, str] = {}
        for sdf in self.datasets[1:]:
            external_rows = sdf.collect()
            for row in external_rows:
                for column_name, value in row.asDict().items():
                    external_input_params[column_name] = (
                        external_input_params.get(column_name, "") + f"{value}\n"
                    )

        external_input_params = {k: v.rstrip("\n") for k, v in external_input_params.items()}
        input_placeholder = {"input": ""}
        self.invocation_params = self.granularity_values | external_input_params | input_placeholder

        self.prompt_template = PromptTemplate.from_template(self.prompt)
        expected_keys = self.prompt_template.input_variables
        for key in expected_keys:
            if key not in self.invocation_params:
                logger.warning(
                    "Prompt invocation parameter '%s' not found in invocation params. Setting it to an empty string.",
                    key,
                )
                self.invocation_params[key] = ""

    def post_processing(self) -> None:
        """Applies post-processing steps to the output DataFrame, including joining with input data, adding granularity columns, dropping specified columns, and applying custom post-processing if configured."""
        # If configured, join the output with the input dataset
        if getattr(self.config, "join_input", False) is True:
            self.output_sdf = self.output_sdf.join(
                self.datasets[0], on=self.config.unique_key_col, how="inner"
            )
        # If configured, add granularity columns to the output
        else:
            if getattr(self.config, "output_granularity", False):
                for column, value in self.granularity_values.items():
                    if column not in self.output_sdf.columns:
                        self.output_sdf = self.output_sdf.withColumn(column, lit(value))
        # If configured, drop any specified output columns
        if getattr(self.config, "output_columns_to_drop", None):
            self.output_sdf = self.output_sdf.drop(*self.config.output_columns_to_drop)
        # If configured, unpivot specified columns
        if getattr(self.config, "unpivot_columns", None):
            unpivot_columns = self.config.unpivot_columns
            id_vars = [col for col in self.output_sdf.columns if col not in unpivot_columns]
            self.output_sdf = self.output_sdf.selectExpr(
                *id_vars,
                f"stack({len(unpivot_columns)}, {', '.join([f"'{col}', {col}" for col in unpivot_columns])}) as (component, output)",
            )
        # If configured, apply custom post processing
        if getattr(self.config, "custom_post_processor", None) is not None:
            post_processer_name = self.config.custom_post_processor
            post_processor = getattr(self.data_module, post_processer_name)
            self.output_sdf = post_processor(
                self.spark,
                self.input_sdf,
                self.output_sdf,
                self.config_dict,
            )
        if getattr(self.config, "drop_duplicates", True):
            self.output_sdf = self.output_sdf.dropDuplicates()

    def export_data(self) -> None:
        """Exports the final results to the configured output table."""
        try:
            self.output_sdf = self.output_sdf.withColumn("run_date", current_timestamp())

            logger.info(
                f"Initiating write operation to {self.output_table_path} with columns: {self.output_sdf.columns} "
                f"and rows: {self.output_sdf.count()}"
            )
            table_exists = self.spark.catalog.tableExists(self.output_table_path)
            schema_option = (
                "mergeSchema"
                if (table_exists and self.config.write_mode == "append")
                else "overwriteSchema"
            )
            writer = (
                self.output_sdf.write.format("delta")
                .options(header="true")
                .mode(self.config.write_mode)
                .option(schema_option, "true")
            )
            if self.config.partition_output_cols:
                writer = writer.partitionBy(*self.config.partition_output_cols)
            writer.saveAsTable(self.output_table_path)
            logger.info(f"Successfully exported rows to {self.output_table_path}")

        except Exception as e:
            logger.error(f"Failed to write table: {self.output_table_path} due to error: {e}")
            raise
