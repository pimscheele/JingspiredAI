"""Prompt templates for example."""


class PromptBase:
    """Class to hold all prompts across the project."""

    example_prompt: str = """Translate the following text to English:
    {input}.
    Return only the example_id and the translated text.
    """

    example_prompt_2: str = """Translate the following text to English:
    {input}.
    Return only the example_id and the translated text.
    """
