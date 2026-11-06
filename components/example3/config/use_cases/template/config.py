"""Configuration for the MSD notes classification pipeline."""

from pydantic import Field

from feedbecky.config.base_config import Base


class ComponentTemplate(Base):
    """Template of a use case config."""

    # The name of the prompt inside the PromptBase class in src/feedbecky/use_case/example/prompt.py
    prompt: str = "example_prompt"

    # The name of the pydantic class inside src/feedbecky/use_case/example/pydantic_class.py
    pydantic: str = "ExamplePydanticClassBatch"

    # The name of the pyspark loader inside src/feedbecky/use_case/example/loader.py
    input_loader_1: str = "example_loader"

    # Input table path
    input_table_path: str = "cieas_studio_p.feedbecky.example_input_table"

    # LLM settings
    reasoning_effort: str = "minimal"
    verbosity: str = "low"
    gpt_version: str = "gpt5nano"

    # Input settings
    input_col: str = "example_notes"

    # The Primary Key columns for the input data
    primary_key_cols: list[str] = Field(
        default_factory=lambda: ["example_id", "example_business_contact_number"]
    )

    # The columns sent to the LLM (the unique_key_col is added automatically)
    llm_processing_input_cols: list[str] = Field(default_factory=lambda: ["example_notes"])

    # The ID column built from primary_key_cols; must match the ID field in the pydantic class
    unique_key_col: str = "id"

    # Whether LLM outputs are matched to input IDs (missing IDs are retried)
    output_id: bool = True

    # Expected number of output tokens per request, added to the prompt token count
    fixed_output_tokens: int = 0

    # Export table name (the table where the processed output will be stored, schema and catalog are added automatically)
    output_table_name: str = "example_output_table"

    # Delta write mode for the output table ("append" or "overwrite")
    write_mode: str = "append"

    # Columns to partition the output table by (empty list for no partitioning)
    partition_output_cols: list[str] = Field(default_factory=list)

    # How may times should the LLM retry in case of failure
    max_retries: int = 3

    # Maximum number of threads to use for batch processing
    max_threads: int = 300

    # Number of rows to process in each batch (set to None to process all rows at once)
    n_batch_rows: int | None = 5
