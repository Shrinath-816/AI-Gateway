"""
llm_client.py
=============

Async client wrapper around OpenRouter's OpenAI-compatible Chat
Completions API.

Why wrap the SDK instead of calling it directly from route handlers:
- keeps provider-specific details (base_url, headers, usage-field
  quirks) in one place, so switching providers later touches one file
- lets us enforce a request timeout centrally
- gives us one place to normalize streaming vs. non-streaming usage
  extraction

OpenRouter-specific behavior this code relies on (verified against
OpenRouter's docs at the time this was written):
- OpenRouter is called via the OpenAI Python SDK, pointed at
  base_url="https://openrouter.ai/api/v1".
- OpenRouter *always* returns usage information automatically, for
  both streaming and non-streaming requests — for streaming, usage
  arrives in the final SSE chunk before [DONE]. Unlike raw OpenAI's
  API, this does not require passing `stream_options={"include_usage": True}`
  (that parameter is accepted but has no effect on OpenRouter, per
  their docs — usage is on by default).
- Because usage arrives in the *final* chunk, a streaming caller only
  gets accurate token counts once the stream is fully consumed — not
  from any intermediate chunk.
"""

from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from app.config import Settings
from app.schemas import UsageInfo


class LLMClient:
    """
    Thin async wrapper around an OpenAI-compatible client configured
    for OpenRouter.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Args:
            settings: Application settings containing the OpenRouter
                API key, base URL, and request timeout.
        """
        self._client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=settings.request_timeout_seconds,
            default_headers={
                # Optional but recommended by OpenRouter for usage
                # attribution on their dashboard/leaderboard. Harmless
                # to omit; kept here as an example of provider-specific
                # header handling living in the client, not the routes.
                "HTTP-Referer": settings.app_referrer,
                "X-Title": settings.app_title,
            },
        )
        self._default_max_tokens = settings.default_max_tokens

    async def complete(self, model: str, prompt: str, max_tokens: int | None = None) -> tuple[str, UsageInfo]:
        """
        Non-streaming completion — waits for the full response before
        returning. Useful for tooling, evaluation harnesses, or any
        caller that doesn't need incremental output.

        Args:
            model: OpenRouter model slug to call.
            prompt: The user's prompt.
            max_tokens: Optional max output tokens override.

        Returns:
            A tuple of (generated_text, usage_info).
        """
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or self._default_max_tokens,
        )

        content = response.choices[0].message.content or ""
        usage = UsageInfo(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
        return content, usage

    async def stream(
        self, model: str, prompt: str, max_tokens: int | None = None
    ) -> AsyncGenerator[str, None]:
        """
        Streaming completion — yields text deltas as they arrive.

        Note on usage: this generator yields *text only*. Token usage
        for a streamed request is not known until the stream completes
        (it arrives in the final chunk), so callers that need usage
        must consume this generator via `stream_with_usage` instead if
        they need the numbers, or track the final chunk themselves.
        See `stream_with_usage` below for the common case of "stream
        the text AND get usage once it's done."

        Args:
            model: OpenRouter model slug to call.
            prompt: The user's prompt.
            max_tokens: Optional max output tokens override.

        Yields:
            Successive text chunks as they are generated.
        """
        async for chunk, _usage in self._stream_chunks(model, prompt, max_tokens):
            if chunk:
                yield chunk

    async def stream_with_usage(
        self, model: str, prompt: str, max_tokens: int | None = None
    ) -> AsyncGenerator[tuple[str | None, UsageInfo | None], None]:
        """
        Streaming completion that also surfaces the usage chunk when it
        arrives.

        Yields:
            Tuples of (text_delta_or_None, usage_or_None). Every chunk
            before the last one yields (text, None). The final chunk
            (OpenRouter's usage-bearing chunk) yields (None, usage) —
            by that point OpenRouter's `choices` delta is content-free,
            it only carries the finish_reason and usage.
        """
        async for chunk, usage in self._stream_chunks(model, prompt, max_tokens):
            yield chunk, usage

    async def _stream_chunks(
        self, model: str, prompt: str, max_tokens: int | None
    ) -> AsyncGenerator[tuple[str | None, UsageInfo | None], None]:
        """
        Internal shared implementation for streaming. Opens the SSE
        stream and yields (text_delta, usage) pairs, exactly one of
        which is non-None per yielded pair.
        """
        stream = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or self._default_max_tokens,
            stream=True,
        )

        async for event in stream:
            # OpenRouter's final SSE chunk carries usage and has an
            # empty/content-free delta rather than new text.
            if getattr(event, "usage", None) is not None:
                usage = UsageInfo(
                    prompt_tokens=event.usage.prompt_tokens,
                    completion_tokens=event.usage.completion_tokens,
                    total_tokens=event.usage.total_tokens,
                )
                yield None, usage
                continue

            delta = event.choices[0].delta.content if event.choices else None
            if delta:
                yield delta, None

    async def close(self) -> None:
        """Release the underlying HTTP client's connections cleanly."""
        await self._client.close()
