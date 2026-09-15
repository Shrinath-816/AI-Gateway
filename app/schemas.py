"""
schemas.py
==========

Pydantic models describing the shapes of data moving in and out of the
API. Keeping these separate from route handlers and business logic
means the "contract" of the API is defined in exactly one place, and
FastAPI can auto-generate accurate OpenAPI docs from it.
"""

from enum import Enum

from pydantic import BaseModel, Field


class TaskComplexity(str, Enum):
    """
    A coarse hint the caller provides describing how demanding the task
    is. This is the input to the model tier router (see model_router.py).

    Why an enum instead of a free-text string: it constrains callers to
    a known, finite set of tiers that the router actually knows how to
    handle, and FastAPI will reject invalid values automatically with a
    422 response instead of that failure surfacing deep inside routing
    logic.
    """

    SIMPLE = "simple"       # classification, extraction, short factual Q&A
    STANDARD = "standard"   # general chat, everyday reasoning
    COMPLEX = "complex"     # multi-step reasoning, coding, long-form analysis


class ChatRequest(BaseModel):
    """
    Incoming request body for both /chat and /chat/sync.

    Attributes:
        prompt: The user's message to the model. Kept as a single string
            for Week 1 simplicity; a real chat app would send full
            conversation history instead (message list), which we will
            introduce once memory/state is covered in Week 4.
        complexity: Hint used to select which model tier handles this
            request. Defaults to STANDARD if the caller doesn't know or
            care.
        max_tokens: Optional override for the maximum number of tokens
            the model may generate in its response.
    """

    prompt: str = Field(..., min_length=1, description="User's prompt to the LLM")
    complexity: TaskComplexity = Field(
        default=TaskComplexity.STANDARD,
        description="Task complexity hint used for model tier routing",
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Optional max output tokens override",
    )


class UsageInfo(BaseModel):
    """
    Token usage for a single request, as reported by the provider.

    Attributes:
        prompt_tokens: Tokens consumed by the input (prompt + system message).
        completion_tokens: Tokens generated in the model's response.
        total_tokens: Sum of the two — this is what providers bill on.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    """
    Response body for the non-streaming /chat/sync endpoint.

    Attributes:
        model: The actual model slug that served this request (useful to
            confirm the tier router picked what you expected).
        content: The full generated text.
        usage: Token usage for this request.
        cost_usd: Computed cost of this request in US dollars.
        latency_ms: Total wall-clock time for the request, in milliseconds.
    """

    model: str
    content: str
    usage: UsageInfo
    cost_usd: float
    latency_ms: float
