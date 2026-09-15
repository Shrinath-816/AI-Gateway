"""
The AI Gateway
===============

A streaming, non-blocking FastAPI backend that sits in front of an LLM
provider (OpenRouter) and adds the things a raw provider SDK does not
give you for free:

- async, non-blocking request handling
- Server-Sent Events (SSE) token streaming to the client
- accurate per-request token usage capture
- cost calculation per request
- model tiering (routing a request to a cheap/fast vs. a strong/expensive
  model based on task complexity)

This is Week 1's project in a 10-week AI Engineering self-study program.
See README.md for setup and usage instructions.
"""
