"""
Utilities for constructing OpenAI-compatible async clients in local-tunnel
development setups.

The main goal is to avoid accidentally sending localhost/private-network Ollama
traffic through system HTTP proxies such as Clash. In that setup, httpx's
default trust_env=True can cause immediate connection failures even though
curl --noproxy localhost works.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

import httpx
from openai import AsyncOpenAI


def should_bypass_env_proxy(base_url: str | None) -> bool:
    """
    Return True when the target looks like a local/private endpoint and should
    ignore HTTP(S)_PROXY-style environment variables.
    """
    if not base_url:
        return False

    hostname = urlparse(base_url).hostname
    if not hostname:
        return False

    if hostname == "localhost":
        return True

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False

    return any(
        (
            address.is_loopback,
            address.is_private,
            address.is_link_local,
            address.is_reserved,
        )
    )


def build_async_openai_client(
    *,
    api_key: str,
    base_url: str | None = None,
    max_retries: int = 0,
) -> AsyncOpenAI:
    """
    Build an AsyncOpenAI client with sane defaults for local/private Ollama
    endpoints while leaving public OpenAI usage unchanged.
    """
    client_kwargs: dict = {
        "api_key": api_key,
        "max_retries": max_retries,
    }
    if base_url:
        client_kwargs["base_url"] = base_url

    if should_bypass_env_proxy(base_url):
        client_kwargs["http_client"] = httpx.AsyncClient(trust_env=False)

    return AsyncOpenAI(**client_kwargs)
