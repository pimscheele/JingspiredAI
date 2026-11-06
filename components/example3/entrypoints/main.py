"""This script serves as the entrypoint for training a model."""
# We want to keep this script to run as an executable - and as short as possible (NO LOGIC ALLOWED).
# Thus the actual logic is defined in model/train.py, where we're only importing the main function to be run.
# This also allows us to use the same function in tests to reach coverage, and exempt this entrypoint from coverage.

from feedbecky.model.feedbecky_generic_genai_pipeline import GenericGenAIPipeline
from feedbecky.utils.parsing import parse_args

args = parse_args(
    arg=["--usecase", "--sub_usecase", "--component"],
    required=[True, False, True],
    type=[str, str, str],
)

# Main function execution (Convert Namespace to dict for kwargs)
kwargs = {k: v for k, v in vars(args).items() if v is not None}  # Filter out None values

pipeline = GenericGenAIPipeline(**kwargs)
pipeline.execute()
