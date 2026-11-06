"""This module provides functionality for managing and initializing application configuration in different environments."""

import os
from typing import Any

import yaml
from databricks.sdk import WorkspaceClient


def is_running_in_databricks() -> bool:
    """Checks if the code is running in a Databricks environment.

    Returns:
        bool: True if running in a Databricks environment, False otherwise.
    """
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_conf_for_current_env(
    env_dict: dict[str, Any],
) -> Any:
    """Get the configuration for the current environment.

    Args:
        env_dict (Dict[str, Callable[[], Any]]): A dictionary where keys are environment
            names (e.g., "local", "studio") and values are callables that return the
            configuration for that environment.

    Returns:
        Any: The configuration for the current environment returned by the callable
        in `env_dict`.

    Raises:
        ValueError: If the current environment is not defined in the `env_dict`.
    """
    if "ENV" not in os.environ:
        # "studio" here refers to Analytics Studio
        current_env = "studio" if is_running_in_databricks() else "local"
    else:
        current_env = str(os.environ.get("ENV")).lower()

    if current_env not in env_dict:
        raise ValueError(f"Invalid environment '{current_env}' specified in ENV.")

    return env_dict[current_env]()


# TODO: Check the ENV versus the databricks studio names as a sanity test!
# E.g. -> in via-xp, running an interactive cluster without ENV properly set causing issues
# -> should be "xp" but ENV not set causing "local" to be picked up instead.


def oai_config_init_tasks(config: Any, strict: bool = False) -> None:
    """Perform openai-specific initialization tasks for the application configuration.

    This function sets up necessary environment variables and loads additional
    configuration data from a YAML file if the environment is set to "xp", "qa", or "pr".

    Args:
        config (Base): The configuration object containing environment-specific settings.
        strict (bool): If True, raises an error when `oai_config_file` is not set in the configuration.

    Raises:
        RuntimeError:
            - If the YAML configuration file does not exist.
            - If there is an error while loading or parsing the YAML configuration file.
    """
    os.environ["SSL_CERT_FILE"] = config.ssl_cert_file

    if getattr(config, "oai_config_file", None) is None:
        if strict:
            raise RuntimeError("oai_config_file is not set in the configuration.")
        return

    if config.ENV.lower() in ["xp", "qa", "pr"]:
        if not os.path.exists(config.oai_config_file):
            raise RuntimeError(f"Configuration file {config.oai_config_file} does not exist.")

        with open(config.oai_config_file) as file:
            try:
                config_data = yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise RuntimeError(f"Error loading YAML file {config.oai_config_file}: {e}") from e

        os.environ["AZURE_CLIENT_ID"] = config_data["AZURE_CLIENT_ID"]
        os.environ["AZURE_TENANT_ID"] = config_data["AZURE_TENANT_ID"]

        dbutils = WorkspaceClient().dbutils
        os.environ["AZURE_CLIENT_SECRET"] = dbutils.secrets.get(
            scope=config.SPN_SECRET_SCOPE, key=config.SPN_SECRET_KEY
        )
