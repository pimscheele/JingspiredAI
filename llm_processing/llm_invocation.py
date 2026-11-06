"""Invoke structured-output OpenAI chains and normalize their results."""

from math import ceil
from typing import Any

from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel
from via_mle.logging import get_logger

from feedbecky.llm_processing.rate_limiter import RateLimiter

logger = get_logger(logger_name=__name__)


class OpenAIAPI:
    """Provide guarded OpenAI calls for batch processing."""

    def api_call(
        self,
        estimated_tokens: float | int,
        llm_chain: Any,
        invocation_params: dict[str, Any],
        current_batch: int = 0,
        rate_limiter: RateLimiter | None = None,
    ) -> tuple[bool, list[dict[str, Any]] | Exception, dict[str, Any]]:
        """Calls the OpenAI API for processing feedback for any of the standard components.

        Expected structured output: BaseModel with `.result -> list[Model|dict]`,
        which will be normalized to list[dict].

        Returns usage data for multithreading scenarios.

        Args:
            estimated_tokens (float | int): Estimated request token count.
            llm_chain (Any): LangChain chain used for invocation.
            invocation_params (dict[str, Any]): Parameters passed to the chain.
            input_text (str | None): Combined feedback payload.
            input_topics (list[str] | None): Optional consolidated topics context.
            current_batch (int): Batch index for logging and rate_limiter bookkeeping.
            rate_limiter (RateLimiter | None): Optional request and token limiter.

        Returns:
            tuple[bool, list[dict[str, Any]] | Exception, dict[str, Any]]:
                Success flag, parsed result or exception, and usage payload.
        """
        try:
            if rate_limiter:
                safe_tokens = ceil(estimated_tokens)
                rate_limiter.acquire(batch_id=current_batch, token_estimate=safe_tokens)
            with get_openai_callback() as cb:
                resp = llm_chain.invoke(invocation_params)

                batch_result = self._extract_result_list(resp)

                usage = self._build_usage_payload(cb)
            return True, batch_result, usage
        except Exception as e:
            logger.error(f"An error occurred during API call. Exception Type: {type(e).__name__}")
            return False, e, {}

    def _extract_result_list(self, resp: Any) -> list[dict[str, Any]]:
        """Strictly extract list[dict] from BaseModel.result."""
        if not isinstance(resp, BaseModel) or not hasattr(resp, "result"):
            raise TypeError(
                f"Expected BaseModel with 'result' (list) for components; got {type(resp)}"
            )
        items = getattr(resp, "result", None)
        if items is None:
            return []
        out: list[dict[str, Any]] = []
        for x in items:
            if isinstance(x, BaseModel) and hasattr(x, "model_dump"):
                out.append(x.model_dump())
            elif isinstance(x, dict):
                out.append(x)
            else:
                raise TypeError(f"Items in 'result' must be BaseModel or dict; got {type(x)}")
        return out

    def _build_usage_payload(self, callback: Any) -> dict[str, Any]:
        """Builds a usage payload from a LangChain OpenAI callback."""
        return {
            "total_tokens": getattr(callback, "total_tokens", 0.0) or 0.0,
        }
