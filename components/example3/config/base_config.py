"""All configuration done through this config file.

This file is both an example as well as a few functioning preset
values for some utility functions.
"""

from typing import Any

import tiktoken
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource
from tiktoken.core import Encoding

from feedbecky.config.helpers import (
    get_conf_for_current_env,
    oai_config_init_tasks,
)


# NOTE: The provided XP, QA and PROD configs are just templates and for further customisations / granular settings
# a DSPE or AS specific class can also be made as appropriate!
class Base(BaseSettings):
    """Base settings."""

    # Use case settings
    team: str = "via"
    use_case: str = "feedbecky"

    # Azure settings
    subscription_id: str = ""
    tenant_id: str = "3a15904d-3fd9-4256-a753-beb05cdf0c6d"
    ingestion_endpoint: str = (
        "https://westeurope-5.in.applicationinsights.azure.com/"  # don't change this!
    )

    # Certificate paths and OAI settings paths
    ssl_cert_file: str = "/Workspace/Shared/certificates/ABN_AMRO_CERTS.pem"
    oai_config_file: str = "/Workspace/Shared/oai_configurations/config.yml"

    # ONLY allow to override default pydantic class attributes if class is instantiated
    # i.e. not from dotenv files or unintentionally from environment variables!
    # See: https://docs.pydantic.dev/latest/concepts/pydantic_settings/#field-value-priority

    # Common config settings
    encoding: Encoding = tiktoken.get_encoding("cl100k_base")
    # OpenAI settings
    open_api_version: str = "2025-04-01-preview"
    bearer_token_scope: str = (
        "https://28e8a4bf-2d66-4d5b-aef5-248993885557.abnamro.onmicrosoft.com/.default"
    )
    openai_endpoint: str = (
        "https://aziag-prd.nl.eu.abnamro.com/maap-cognitive-services/openai-services/v1"
    )
    variable_token_count_col: str = "variable_token_count"
    token_provider: Any = None

    time_per_window: int = 60
    requests_per_minute: int = 100

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize the sources for settings initialization.

        Configures Pydantic settings to only allow overriding default class attributes
        when the class is explicitly instantiated. This prevents unintended configuration
        from dotenv files or environment variables.

        Args:
            settings_cls: The settings class being configured.
            init_settings: Settings provided during class initialization.
            env_settings: Settings from environment variables.
            dotenv_settings: Settings from dotenv files.
            file_secret_settings: Settings from secret files.

        Returns:
            A tuple containing only init_settings as the configuration source.

        See: https://docs.pydantic.dev/latest/concepts/pydantic_settings/#field-value-priority
        """
        return (init_settings,)


class Local(Base):
    """Local specific configs."""

    # General
    ENV: str = "LOCAL"
    mlflow_model_alias: str = ENV.lower()
    databricks_hostname: str = ""
    workspace: str = ""

    # UC table settings
    uc_catalog: str = "dspe_viaxp1_p"
    uc_schema: str = "viaxp1_p"
    output_table_path: str = ""
    model_registry: str = ""

    # Azure resource settings
    instrumentation_key: str = ""
    APP_INSIGHTS_CONNECTION_STRING: str = ""
    resource_group: str = ""

    # Empty but should be initialised
    SPN_SECRET_SCOPE: str = ""
    SPN_SECRET_KEY: str = ""

    registry_uri: str = ""
    experiment_name: str = ""
    experiment_path: str = ""
    mlflow_model_name: str = ""

    current_env_deployments: dict[str, tuple[str, int]] = {
        "gpt5": ("fbkygpt5xpqa", 100000),
        "gpt5mini": ("fbky5minixpqa", 100000),
        "gpt5nano": ("fbky5nanoxpqa", 100000),
    }


class ASBase(Base):
    """Analytics Studio specific configs."""

    ENV: str = "STUDIO"
    mlflow_model_alias: str = ENV.lower()
    databricks_hostname: str = ""
    workspace: str = ""

    # UC table settings
    uc_catalog: str = "cieas_studio_p"
    uc_schema: str = "feedbecky"
    output_table_path: str = ""
    model_registry: str = ""

    # Azure resource settings
    instrumentation_key: str = ""
    APP_INSIGHTS_CONNECTION_STRING: str = ""
    resource_group: str = ""

    # Empty but should be initialised
    SPN_SECRET_SCOPE: str = ""
    SPN_SECRET_KEY: str = ""

    registry_uri: str = ""
    experiment_name: str = ""
    experiment_path: str = ""
    mlflow_model_name: str = ""

    current_env_deployments: dict[str, tuple[str, int]] = {
        "gpt5": ("ciegpt5", 100000),
        "gpt5mini": ("cie5mini", 100000),
        "gpt5nano": ("cie5nano", 150000),
    }


class DSPE(Base):
    """DSPE specific configs."""

    # Security groups and SPNs
    security_group_names: list[str] = ["DSPE-VIA-DS", "DSPE-VIA-MLE"]
    service_principal_names: list[str] = [
        "fcc6a177-52d0-44ba-8028-6bcbb2149906",  # dspe-viaxp1-p-spn
        "a5fad5a7-626c-442e-a326-ec29dd780a88",  # dspe-viaqa1-p-spn
        "ffae395c-9ed4-4439-abb6-81175d85640b",  # dspe-viapr1-p-spn
    ]

    # Azure OpenAI settings
    ABN_CREDENTIAL: str = (
        "https://28e8a4bf-2d66-4d5b-aef5-248993885557.abnamro.onmicrosoft.com/.default"
    )
    AZURE_OPENAI_ENDPOINT: str = (
        "https://aziag-prd.nl.eu.abnamro.com/maap-cognitive-services/openai-services/v1"
    )
    DEPLOYMENT_NAME: str = ""
    TOKENS_PER_MINUTE: int = 0

    # UC settings
    registry_uri: str = "databricks-uc"
    common_model_registry: str = "dspe_common1_p.model_registry_via"

    @property
    def experiment_name(self) -> str:
        """The name of the MLflow experiment."""
        return f"{self.team}_{self.use_case}"

    @property
    def experiment_path(self) -> str:
        """The path of the MLflow experiment."""
        return f"/Workspace/Shared/{self.team}/{self.use_case}"

    # DDS paths (UC)
    dds_catalog_name: str = "cbdsap_p"
    dds_input_schema_name: str = "dspe_outbound"
    dds_output_schema_name: str = "dspe_inbound"
    dds_train_table_path: str = f"{dds_catalog_name}.{dds_input_schema_name}.<INSERT_TABLE_NAME>"
    dds_predict_table_path: str = f"{dds_catalog_name}.{dds_input_schema_name}.<INSERT_TABLE_NAME>"
    dds_output_schema: str = f"{dds_catalog_name}.{dds_output_schema_name}"


class XP(DSPE):
    """XP specific configs."""

    # General
    ENV: str = "XP"
    mlflow_model_alias: str = ENV.lower()
    databricks_hostname: str = "adb-4042418083290321.1.azuredatabricks.net"
    workspace: str = f"https://{databricks_hostname}/"

    @property
    def mlflow_model_name(self) -> str:
        """The name of the MLflow model."""
        return f"{self.team}_{self.use_case}_model_{self.ENV}"

    # UC table settings
    uc_catalog: str = "dspe_viaxp1_p"
    uc_schema: str = "viaxp1_p"

    @property
    def model_registry(self) -> str:
        """The model registry path in UC format."""
        return f"{self.uc_catalog}.{self.uc_schema}"

    @property
    def model_path(self) -> str:
        """The full path of the MLflow model in UC format."""
        return f"{self.model_registry}.{self.mlflow_model_name}"

    # Azure resource settings
    instrumentation_key: str = ""
    APP_INSIGHTS_CONNECTION_STRING: str = ""
    resource_group: str = ""

    # SPN secret details
    SPN_SECRET_SCOPE: str = "dspe-viaxp1-p-secret-scope"
    SPN_SECRET_KEY: str = "dspe-viaxp1-p-spn-secret"

    current_env_deployments: dict[str, tuple[str, int]] = {
        "gpt5": ("fbkygpt5xpqa", 100000),
        "gpt5mini": ("fbky5minixpqa", 100000),
        "gpt5nano": ("fbky5nanoxpqa", 100000),
    }


class QA(DSPE):
    """QA specific configs."""

    # General
    ENV: str = "QA"
    mlflow_model_alias: str = "champion"
    databricks_hostname: str = "adb-2756955058793107.7.azuredatabricks.net"
    workspace: str = f"https://{databricks_hostname}/"

    @property
    def mlflow_model_name(self) -> str:
        """The name of the MLflow model."""
        return f"{self.team}_{self.use_case}_model_{self.ENV}"

    # UC table settings
    uc_catalog: str = "dspe_viaqa1_p"
    uc_schema: str = "viaqa1_p"

    @property
    def model_registry(self) -> str:
        """The model registry path in UC format."""
        return self.common_model_registry

    @property
    def model_path(self) -> str:
        """The full path of the MLflow model in UC format."""
        return f"{self.model_registry}.{self.mlflow_model_name}"

    # Azure resource settings
    instrumentation_key: str = ""
    APP_INSIGHTS_CONNECTION_STRING: str = ""
    resource_group: str = ""

    # SPN secret details
    SPN_SECRET_SCOPE: str = "dspe-viaqa1-p-secret-scope"
    SPN_SECRET_KEY: str = "dspe-viaqa1-p-spn-secret"

    current_env_deployments: dict[str, tuple[str, int]] = {
        "gpt5": ("fbkygpt5xpqa", 100000),
        "gpt5mini": ("fbky5minixpqa", 100000),
        "gpt5nano": ("fbky5nanoxpqa", 100000),
    }


class Prod(DSPE):
    """Prod specific configs."""

    # General
    ENV: str = "PR"
    mlflow_model_alias: str = "champion"
    databricks_hostname: str = "adb-1846028197963591.11.azuredatabricks.net"
    workspace: str = f"https://{databricks_hostname}/"

    @property
    def mlflow_model_name(self) -> str:
        """The name of the MLflow model."""
        return f"{self.team}_{self.use_case}_model_{self.ENV}"

    # UC table settings
    uc_catalog: str = "dspe_viapr1_p"
    uc_schema: str = "viapr1_p"

    @property
    def model_registry(self) -> str:
        """The model registry path in UC format."""
        return self.common_model_registry

    @property
    def model_path(self) -> str:
        """The full path of the MLflow model in UC format."""
        return f"{self.model_registry}.{self.mlflow_model_name}"

    # Azure resource settings
    instrumentation_key: str = ""
    APP_INSIGHTS_CONNECTION_STRING: str = ""
    resource_group: str = ""

    # SPN secret details
    SPN_SECRET_SCOPE: str = "dspe-viapr1-p-secret-scope"
    SPN_SECRET_KEY: str = "dspe-viapr1-p-spn-secret"

    current_env_deployments: dict[str, tuple[str, int]] = {
        "gpt5": ("fbkygpt5", 200000),
        "gpt5mini": ("fbky5mini", 400000),
        "gpt5nano": ("fbky5nano", 200000),
    }


######################## CONFIG INIT STEPS ########################

env_dict: dict[str, type[Local] | type[ASBase] | type[XP] | type[QA] | type[Prod]] = {
    "local": Local,
    "studio": ASBase,
    "xp": XP,
    "qa": QA,
    "pr": Prod,
}
conf = get_conf_for_current_env(env_dict=env_dict)
oai_config_init_tasks(config=conf)
conf.token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    conf.bearer_token_scope,
)
