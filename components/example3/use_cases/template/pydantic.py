"""Example pydantic class."""

from pydantic import BaseModel, Field


class ExamplePydanticClass(BaseModel):
    """Example pydantic class."""

    id: str = Field(..., description="A composite primary key for a record")
    translation: str = Field(
        ...,
        description="The translated text in English",
    )


class ExamplePydanticClassBatch(BaseModel):
    """Represents a batch of example translation results.

    Attributes:
        result (list[ExamplePydanticClass]): List of example translation results.
    """

    result: list[ExamplePydanticClass] = Field(
        ...,
        description="Batch example translation results",
    )
