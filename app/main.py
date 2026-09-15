"""
main.py
=======

FastAPI application entry point for The AI Gateway (Week 1 project).

Endpoints:
    POST /chat        - streaming chat completion (Server-Sent Events)
    POST /chat/sync    - non-streaming chat completion (full response at once)
    GET  /usage        - summary of all requests served since process start
    GET  /health       - basic liveness check

Run locally with:
    uvicorn app.main:app --reload

See README.md for full setup instructions.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.llm_client import LLMClient
from app.model_router import ModelTierRouter
from app.pricing import CostCalculator
from app.schemas import ChatRequest, ChatResponse, UsageInfo
from app.usage_tracker import RequestTimer, UsageTracker

# --------------------------------------------------------------------------
# Application state
# --------------------------------------------------------------------------
#
# These components are created once per process (not per request) because
# they are either stateless (router, cost calculator) or intentionally
# hold cross-request state (usage tracker) or manage a connection pool
# (LLM client). Creating a new LLMClient per request would defeat HTTP
# connection reuse and is a common performance mistake in async services.

router = ModelTierRouter()
cost_calculator = CostCalculator()
usage_tracker = UsageTracker()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan hook: create the LLM client on startup (reading
    settings once, failing fast if OPENROUTER_API_KEY is missing) and
    close its underlying HTTP connections cleanly on shutdown.
    """
    settings = get_settings()
    app.state.llm_client = LLMClient(settings)
    yield
    await app.state.llm_client.close()


app = FastAPI(
    title="The AI Gateway",
    description="Week 1 project: streaming LLM backend with usage and cost tracking.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_llm_client(request: Request) -> LLMClient:
    """FastAPI dependency: retrieve the shared LLMClient from app state."""
    return request.app.state.llm_client


@app.get("/health")
async def health() -> dict[str, str]:
    """Basic liveness check — does not call the LLM provider."""
    return {"status": "ok"}


@app.get("/usage")
async def usage_summary() -> dict[str, float | int]:
    """
    Return aggregate usage across every request served since the
    process started. Backed by the in-memory UsageTracker — restart
    the process and this resets, which is fine for Week 1's scope.
    """
    return {
        "total_requests": len(usage_tracker.all_logs()),
        "total_tokens": usage_tracker.total_tokens(),
        "total_cost_usd": usage_tracker.total_cost_usd(),
    }


@app.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(
    chat_request: ChatRequest,
    llm_client: LLMClient = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    """
    Non-streaming chat completion.

    Waits for the full model response, then returns it in one JSON
    payload along with usage and cost. Useful for callers that don't
    need incremental output — e.g. batch jobs, evaluation harnesses,
    or simple tooling.
    """
    model = router.select_model(chat_request.complexity)

    timer = RequestTimer()
    timer.start()
    content, usage = await llm_client.complete(
        model=model,
        prompt=chat_request.prompt,
        max_tokens=chat_request.max_tokens,
    )
    latency_ms = timer.stop()

    cost_usd = cost_calculator.calculate(
        model=model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )
    usage_tracker.record(
        model=model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )

    return ChatResponse(
        model=model,
        content=content,
        usage=usage,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )


@app.post("/chat")
async def chat_stream(
    chat_request: ChatRequest,
    request: Request,
    llm_client: LLMClient = Depends(get_llm_client),
) -> StreamingResponse:
    """
    Streaming chat completion via Server-Sent Events (SSE).

    Forwards text chunks to the client as they arrive from the
    provider, instead of buffering the full response server-side.
    This is what gives the client fast time-to-first-token instead of
    waiting for the entire generation to finish.

    Usage and cost are only known once the stream completes (the
    provider's usage data arrives in the final chunk), so they are
    recorded into the UsageTracker at the end of the generator rather
    than returned in the HTTP response itself — a streaming response
    has already started sending bytes to the client by the time we'd
    know the final cost, so there's no clean place to return it inline.
    The /usage endpoint exposes the aggregate after the fact.
    """
    model = router.select_model(chat_request.complexity)

    async def event_generator() -> AsyncIterator[str]:
        timer = RequestTimer()
        timer.start()

        try:
            async for text_delta, usage in llm_client.stream_with_usage(
                model=model,
                prompt=chat_request.prompt,
                max_tokens=chat_request.max_tokens,
            ):
                # Stop consuming/streaming if the client has gone away —
                # no point paying for tokens nobody will read.
                if await request.is_disconnected():
                    break

                if text_delta is not None:
                    # SSE wire format: each event is "data: <payload>\n\n"
                    yield f"data: {text_delta}\n\n"

                if usage is not None:
                    latency_ms = timer.stop()
                    cost_usd = cost_calculator.calculate(
                        model=model,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                    )
                    usage_tracker.record(
                        model=model,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        cost_usd=cost_usd,
                        latency_ms=latency_ms,
                    )
        finally:
            # Standard SSE termination signal so clients using
            # EventSource-style parsing know the stream is complete.
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
