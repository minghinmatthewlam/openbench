"""Strict Gateway Bench profiles and privacy-safe route evidence."""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping
from typing import Any


GATEWAYS = frozenset({"cloudflare", "concentrate", "openrouter", "ramp", "vercel"})

_OPENROUTER_ENDPOINTS = {
    "openai_chat": "https://openrouter.ai/api/v1/chat/completions",
    "openai_responses": "https://openrouter.ai/api/v1/responses",
}
_VERCEL_ENDPOINTS = {
    "openai_chat": "https://ai-gateway.vercel.sh/v1/chat/completions",
    "openai_responses": "https://ai-gateway.vercel.sh/v1/responses",
}
_CONCENTRATE_ENDPOINTS = {
    "openai_chat": "https://api.concentrate.ai/v1/chat/completions",
    "openai_responses": "https://api.concentrate.ai/v1/responses",
}
_RAMP_ENDPOINTS = {
    "openai_responses": "https://router-api.ramp.com/v1/responses",
}
# Ramp returns the served model but no upstream-provider field. Admit only
# model IDs whose ownership is unambiguous instead of trusting the profile's
# requested_provider declaration for arbitrary catalog entries.
_RAMP_MODEL_PROVIDERS = {
    "gpt-5.6-sol": "openai",
}
_CONCENTRATE_PROVIDER_SLUGS = {
    "moonshotai": "moonshot",
}
_OPENROUTER_PROVIDER_SLUGS = {
    "deepseek": "DeepSeek",
    "zai": "Z.AI",
}
_PROVIDER_IDENTITY_ALIASES = {
    "moonshot ai": "moonshotai",
    "z.ai": "zai",
    "z-ai": "zai",
}
_MODEL_ID_ALIASES = {
    "deepseek-v4-flash-0731": "deepseek-v4-flash-20260731",
}
_CLOUDFLARE_REST_ENDPOINT_RE = re.compile(
    r"https://api\.cloudflare\.com/client/v4/accounts/"
    r"(?P<account_id>[0-9a-fA-F]{32})/ai/v1/"
    r"(?P<operation>chat/completions|responses)"
)
_GATEWAY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_CACHE_KEYS = frozenset({
    "cache",
    "cache_control",
    "cache_key",
    "caching",
    "prompt_cache_breakpoint",
    "prompt_cache_key",
    "prompt_cache_options",
    "prompt_cache_retention",
})
_SCHEMA_KEYS = frozenset({
    "input_schema",
    "json_schema",
    "parameters",
    "schema",
})
_DATED_REVISION_RE = re.compile(
    r"-(?:\d{4}-\d{2}-\d{2}|\d{8})$"
)


class GatewayProfileError(ValueError):
    """Raised when a gateway profile cannot satisfy its benchmark contract."""


def models_match(requested: str, observed: str, mode: str) -> bool:
    """Compare an observed route without overstating revision equivalence."""
    if mode == "exact_revision":
        return observed == requested
    if mode not in {"model_family", "rolling_alias"}:
        raise GatewayProfileError(f"unsupported model_match mode: {mode}")
    if mode == "rolling_alias":
        return model_evidence_consistent(requested, observed, mode)
    return _model_alias(requested) == _model_alias(observed)


def model_evidence_consistent(first: str, second: str, mode: str) -> bool:
    """Accept alias resolution while rejecting two contradictory snapshots."""
    if mode == "exact_revision":
        return first == second
    if mode not in {"model_family", "rolling_alias"}:
        raise GatewayProfileError(f"unsupported model_match mode: {mode}")
    if not _providers_compatible(first, second):
        return False
    first_id = _model_id(first)
    second_id = _model_id(second)
    if first_id == second_id:
        return True
    first_revision = _dated_revision(first_id)
    second_revision = _dated_revision(second_id)
    if first_revision and second_revision:
        first_base, first_date = first_revision
        second_base, second_date = second_revision
        return (
            first_base == second_base
            and first_date == second_date
        )
    return _model_alias(first_id) == _model_alias(second_id)


def _model_id(value: str) -> str:
    model_id = value.casefold().rsplit("/", 1)[-1]
    return _MODEL_ID_ALIASES.get(model_id, model_id)


def _model_provider(value: str) -> str | None:
    normalized = value.casefold()
    if "/" not in normalized:
        return None
    return normalized.rsplit("/", 1)[0]


def _provider_identity(value: str) -> str:
    normalized = value.strip().casefold()
    return _PROVIDER_IDENTITY_ALIASES.get(normalized, normalized)


def _persisted_provider(value: str) -> str:
    stripped = value.strip()
    return _PROVIDER_IDENTITY_ALIASES.get(stripped.casefold(), stripped)


def _providers_compatible(first: str, second: str) -> bool:
    first_provider = _model_provider(first)
    second_provider = _model_provider(second)
    return (
        first_provider is None
        or second_provider is None
        or _provider_identity(first_provider)
        == _provider_identity(second_provider)
    )


def model_provider_matches(value: str, expected_provider: str) -> bool:
    """Allow an unqualified model ID, but reject a contradictory qualifier."""
    provider = _model_provider(value)
    return (
        provider is None
        or _provider_identity(provider) == _provider_identity(expected_provider)
    )


def concrete_model_revision(value: str) -> str | None:
    """Return a normalized dated model ID, or None for a rolling alias."""
    model_id = _model_id(value)
    return model_id if _DATED_REVISION_RE.search(model_id) else None


def _dated_revision(value: str) -> tuple[str, str] | None:
    match = _DATED_REVISION_RE.search(value)
    if match is None:
        return None
    return value[:match.start()], match.group(0)[1:].replace("-", "")


def _model_alias(value: str) -> str:
    return _DATED_REVISION_RE.sub("", _model_id(value))


def _concentrate_provider_slug(value: str) -> str:
    provider = _provider_identity(value)
    return _CONCENTRATE_PROVIDER_SLUGS.get(provider, provider)


def _openrouter_provider_slug(value: str) -> str:
    provider = _provider_identity(value)
    return _OPENROUTER_PROVIDER_SLUGS.get(provider, provider)


def validate_arm(
    *,
    route_kind: str,
    gateway: str | None,
    gateway_id: str | None = None,
    endpoint: str,
    protocol: str,
    requested_model: str,
    requested_provider: str,
    allow_private_endpoint: bool = False,
) -> None:
    """Validate profile-specific, nonsecret arm fields."""
    if route_kind == "direct":
        if gateway is not None:
            raise GatewayProfileError("direct arm must not declare gateway")
        if gateway_id is not None:
            raise GatewayProfileError("direct arm must not declare gateway_id")
        return
    if gateway is None:
        raise GatewayProfileError("gateway arm requires gateway")
    if gateway not in GATEWAYS:
        raise GatewayProfileError(
            f"gateway must be one of: {', '.join(sorted(GATEWAYS))}"
        )
    if gateway != "cloudflare" and gateway_id is not None:
        raise GatewayProfileError(
            "gateway_id is supported only for cloudflare managed gateway arms"
        )
    if gateway == "cloudflare" and (
        gateway_id is None or _GATEWAY_ID_RE.fullmatch(gateway_id) is None
    ):
        raise GatewayProfileError(
            "cloudflare managed gateway arm requires a valid gateway_id"
        )
    if gateway == "concentrate":
        expected_endpoint = _CONCENTRATE_ENDPOINTS.get(protocol)
        if endpoint != expected_endpoint:
            raise GatewayProfileError(
                "concentrate endpoint must be "
                f"{expected_endpoint} for protocol {protocol}"
            )
        if (
            _model_provider(requested_model)
            != _concentrate_provider_slug(requested_provider)
        ):
            raise GatewayProfileError(
                "concentrate requested_model must be provider-qualified with "
                "the Concentrate provider slug for requested_provider"
            )
    if gateway == "cloudflare":
        rest_match = _CLOUDFLARE_REST_ENDPOINT_RE.fullmatch(endpoint)
        expected_operation = (
            "responses" if protocol == "openai_responses" else "chat/completions"
        )
        if rest_match is None or rest_match.group("operation") != expected_operation:
            raise GatewayProfileError(
                "cloudflare managed endpoint must be "
                "https://api.cloudflare.com/client/v4/accounts/"
                "{32-hex-account-id}/ai/v1/{chat/completions|responses}"
            )
        if (
            _model_provider(requested_model) is None
            or not model_provider_matches(requested_model, requested_provider)
        ):
            raise GatewayProfileError(
                "cloudflare requested_model must be provider-qualified with "
                "requested_provider"
            )
    if gateway == "ramp":
        expected_endpoint = _RAMP_ENDPOINTS.get(protocol)
        if endpoint != expected_endpoint:
            raise GatewayProfileError(
                "ramp endpoint must be "
                f"{expected_endpoint} for protocol {protocol}"
            )
        expected_provider = _RAMP_MODEL_PROVIDERS.get(_model_id(requested_model))
        if expected_provider != _provider_identity(requested_provider):
            raise GatewayProfileError(
                "ramp requested_model must have an admitted unambiguous "
                "provider mapping"
            )

    if allow_private_endpoint:
        return
    if gateway == "openrouter" and endpoint != _OPENROUTER_ENDPOINTS.get(protocol):
        raise GatewayProfileError(
            f"openrouter endpoint must be {_OPENROUTER_ENDPOINTS.get(protocol)}"
        )
    if gateway == "vercel":
        if endpoint != _VERCEL_ENDPOINTS.get(protocol):
            raise GatewayProfileError(
                f"vercel endpoint must be {_VERCEL_ENDPOINTS.get(protocol)}"
            )
        if "/" not in requested_model:
            raise GatewayProfileError(
                "vercel requested_model must be a provider-qualified model ID"
            )
def _strip_cache_controls(value: Any) -> None:
    if isinstance(value, dict):
        for key in list(value):
            normalized = str(key).lower().replace("-", "_")
            if normalized in _CACHE_KEYS:
                value.pop(key)
            elif normalized not in _SCHEMA_KEYS:
                _strip_cache_controls(value[key])
    elif isinstance(value, list):
        for item in value:
            _strip_cache_controls(item)


def strip_cache_controls(payload: dict[str, Any]) -> None:
    """Remove request cache controls without rewriting tool/data schemas."""
    _strip_cache_controls(payload)


def shape_provider_body(
    payload: dict[str, Any],
    *,
    requested_provider: str,
) -> None:
    """Apply provider-native request fields shared by direct and gateway arms."""
    if requested_provider.casefold() != "deepseek":
        return
    max_output_tokens = payload.pop("max_completion_tokens", None)
    if max_output_tokens is not None:
        payload["max_tokens"] = max_output_tokens
    payload.pop("seed", None)


def shape_body(
    payload: dict[str, Any],
    *,
    gateway: str,
    requested_provider: str,
) -> None:
    """Replace caller-controlled gateway routing and cache policy in-place."""
    strip_cache_controls(payload)
    if gateway == "openrouter":
        routing_keys = (
            "provider", "providerOptions", "plugins", "router",
            "session_id", "conversation_id", "models", "order", "sort",
            "caching",
        )
        for key in routing_keys:
            payload.pop(key, None)
        payload["provider"] = {
            "only": [_openrouter_provider_slug(requested_provider)],
            "allow_fallbacks": False,
        }
        return
    if gateway == "vercel":
        payload.pop("provider", None)
        for key in ("models", "order", "sort", "caching"):
            payload.pop(key, None)
        payload["providerOptions"] = {
            "gateway": {"only": [requested_provider]},
        }
        return
    if gateway == "cloudflare":
        for key in (
            "provider", "providerOptions", "models", "order", "sort", "caching",
            "router", "plugins", "routes", "fallback",
        ):
            payload.pop(key, None)
        return
    if gateway == "concentrate":
        for key in (
            "provider", "providerOptions", "models", "order", "sort",
            "router", "plugins", "routes", "fallback", "fallbacks",
        ):
            payload.pop(key, None)
        seed = payload.get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool):
            payload["seed"] = str(seed)
        payload["routing"] = {
            "providers": [_concentrate_provider_slug(requested_provider)],
            "models": [],
        }
        return
    if gateway == "ramp":
        for key in (
            "provider", "providerOptions", "models", "order", "sort",
            "router", "plugins", "routes", "fallback", "fallbacks",
        ):
            payload.pop(key, None)
        return
    raise GatewayProfileError(f"unsupported gateway profile: {gateway}")


def request_headers(
    *,
    gateway: str | None,
    secret: str,
    gateway_id: str | None = None,
) -> dict[str, str]:
    """Return authoritative auth and gateway control headers."""
    headers = {"Authorization": f"Bearer {secret}"}
    if gateway is None:
        return headers
    if gateway == "openrouter":
        headers.update({
            "X-OpenRouter-Metadata": "enabled",
            "X-OpenRouter-Cache": "false",
        })
    elif gateway == "cloudflare":
        if gateway_id is None or _GATEWAY_ID_RE.fullmatch(gateway_id) is None:
            raise GatewayProfileError(
                "cloudflare managed requests require a valid gateway_id"
            )
        headers.update({
            "cf-aig-gateway-id": gateway_id,
            "cf-aig-skip-cache": "true",
            "cf-aig-max-attempts": "1",
            "cf-aig-collect-log-payload": "false",
        })
    elif gateway not in {"concentrate", "ramp", "vercel"}:
        raise GatewayProfileError(f"unsupported gateway profile: {gateway}")
    return headers


def blocked_request_header(name: str) -> bool:
    """Return whether an inbound header could alter managed gateway behavior."""
    normalized = name.lower()
    return (
        normalized == "x-openrouter-metadata"
        or normalized.startswith("x-openrouter-cache")
        or normalized.startswith("cf-aig-")
    )


def _identifier(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _finite_nonnegative_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0 or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value


def _clean_openrouter_attempts(value: Any) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(value, list):
        return [], False
    attempts = []
    for item in value:
        if not isinstance(item, dict):
            return attempts, False
        attempt: dict[str, Any] = {}
        provider = _identifier(item.get("provider"))
        model = _identifier(item.get("model"))
        status = _integer(item.get("status"))
        if provider is not None:
            attempt["provider"] = _persisted_provider(provider)
        if model is not None:
            attempt["model"] = model
        if status is not None:
            attempt["status"] = status
        if not attempt:
            return attempts, False
        attempts.append(attempt)
    return attempts, True


@dataclasses.dataclass(slots=True)
class GatewayEvidence:
    """Accumulate only privacy-safe routing metadata for one streamed response."""

    gateway: str
    requested_model: str
    requested_provider: str
    allowed_models: tuple[str, ...]
    allowed_providers: tuple[str, ...]
    model_match: str = "exact_revision"
    response_headers: Mapping[str, str] = dataclasses.field(default_factory=dict)
    metadata_seen: bool = False
    metadata_requested_model: str | None = None
    served_model: str | None = None
    provider: str | None = None
    attempts: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    attempts_present: bool = False
    attempts_malformed: bool = False
    profile_reasons: list[str] = dataclasses.field(default_factory=list)
    safe_metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    _dated_model_ids: set[str] = dataclasses.field(
        default_factory=set, repr=False
    )

    def observe(self, obj: Mapping[str, Any]) -> bool:
        top_model = _identifier(obj.get("model"))
        top_provider = _identifier(obj.get("provider"))
        if self.gateway == "openrouter":
            if top_model is not None:
                self._set_model(top_model)
            if top_provider is not None:
                self._set_provider(top_provider)
            return self._observe_openrouter(obj)
        elif self.gateway == "vercel":
            return self._observe_vercel(obj, top_model, top_provider)
        elif self.gateway == "cloudflare":
            return self._observe_cloudflare(top_model)
        elif self.gateway == "concentrate":
            return self._observe_concentrate(top_model)
        elif self.gateway == "ramp":
            return self._observe_ramp(top_model, top_provider)
        return False

    def _observe_cloudflare(self, top_model: str | None) -> bool:
        if top_model is None:
            return False
        self.metadata_seen = True
        requested_provider = _model_provider(self.requested_model)
        if requested_provider is not None:
            self._set_provider(requested_provider)
            if not model_provider_matches(top_model, requested_provider):
                self.profile_reasons.append("provider_conflict")
        self._set_model(top_model)
        return True

    def _observe_concentrate(self, top_model: str | None) -> bool:
        if top_model is None:
            return False
        self.metadata_seen = True
        provider = _model_provider(top_model)
        if provider is None:
            self.profile_reasons.append("unqualified_served_model")
        else:
            self._set_provider(provider)
        self._set_model(top_model)
        return True

    def _observe_ramp(
        self,
        top_model: str | None,
        top_provider: str | None,
    ) -> bool:
        """Keep only route identity the hosted Router returns itself."""
        if top_model is None and top_provider is None:
            return False
        self.metadata_seen = True
        if top_model is not None:
            self._set_model(top_model)
        if top_provider is not None:
            self._set_provider(top_provider)
        elif top_model is not None:
            mapped_provider = _RAMP_MODEL_PROVIDERS.get(_model_id(top_model))
            if mapped_provider is not None:
                self._set_provider(mapped_provider)
        return True

    def _set_provider(self, value: str) -> None:
        if (
            self.provider is not None
            and _provider_identity(self.provider) != _provider_identity(value)
        ):
            self.profile_reasons.append("provider_conflict")
        self.provider = _persisted_provider(value)

    def _set_model(self, value: str) -> None:
        self._record_model(value)
        if (
            self.served_model is not None
            and not model_evidence_consistent(
                self.served_model, value, self.model_match
            )
        ):
            self.profile_reasons.append("served_model_conflict")
        self.served_model = value

    def _record_model(self, value: str) -> None:
        if self.model_match != "rolling_alias":
            return
        revision = concrete_model_revision(value)
        if revision is not None:
            if any(
                not model_evidence_consistent(
                    existing, revision, self.model_match
                )
                for existing in self._dated_model_ids
            ):
                self.profile_reasons.append("served_model_conflict")
            self._dated_model_ids.add(revision)

    def _observe_openrouter(self, obj: Mapping[str, Any]) -> bool:
        usage = obj.get("usage")
        if isinstance(usage, Mapping):
            self.safe_metadata.pop("cost", None)
            cost = _finite_nonnegative_number(usage.get("cost"))
            if cost is not None:
                self.safe_metadata["cost"] = cost

        metadata = obj.get("openrouter_metadata")
        if not isinstance(metadata, dict):
            return False
        self.metadata_seen = True
        requested = _identifier(metadata.get("requested"))
        if requested is not None:
            self.metadata_requested_model = requested
        if "attempts" in metadata:
            raw_attempts = metadata.get("attempts")
            if isinstance(raw_attempts, list) and not raw_attempts:
                self.attempts_present = False
                self.attempts = []
            else:
                self.attempts_present = True
                self.attempts, valid = _clean_openrouter_attempts(raw_attempts)
                self.attempts_malformed = not valid
                for attempt in self.attempts:
                    model = attempt.get("model")
                    if isinstance(model, str):
                        self._record_model(model)
        endpoints = metadata.get("endpoints")
        available = endpoints.get("available") if isinstance(endpoints, dict) else None
        if isinstance(available, list):
            for endpoint in available:
                if isinstance(endpoint, dict) and endpoint.get("selected") is True:
                    provider = _identifier(endpoint.get("provider"))
                    if provider is not None:
                        self._set_provider(provider)
                    model = _identifier(endpoint.get("model"))
                    if model is not None:
                        self._record_model(model)
                        if not models_match(
                            self.requested_model, model, self.model_match
                        ):
                            self.profile_reasons.append("served_model_conflict")
                    break
        return True

    @staticmethod
    def _vercel_metadata(obj: Mapping[str, Any]) -> Mapping[str, Any] | None:
        containers = [obj]
        choices = obj.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, Mapping):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, Mapping):
                    containers.append(delta)
        for container in containers:
            for key in ("providerMetadata", "provider_metadata"):
                provider_metadata = container.get(key)
                if isinstance(provider_metadata, Mapping):
                    gateway = provider_metadata.get("gateway")
                    if isinstance(gateway, Mapping):
                        return gateway
        gateway = obj.get("gateway")
        return gateway if isinstance(gateway, Mapping) else None

    def _observe_vercel(
        self,
        obj: Mapping[str, Any],
        top_model: str | None,
        top_provider: str | None,
    ) -> bool:
        if top_model is not None:
            self._record_model(top_model)
        metadata = self._vercel_metadata(obj)
        if metadata is None:
            return False
        self.metadata_seen = True
        routing = metadata.get("routing")
        evidence = routing if isinstance(routing, Mapping) else metadata

        if "originalModelId" in evidence:
            self.metadata_requested_model = _identifier(
                evidence.get("originalModelId")
            )
        else:
            self.metadata_requested_model = self.requested_model

        final_provider = _identifier(evidence.get("finalProvider"))
        resolved_provider = _identifier(evidence.get("resolvedProvider"))
        if resolved_provider is not None:
            self._set_provider(resolved_provider)
        if final_provider is not None:
            self._set_provider(final_provider)
        elif resolved_provider is None and top_provider is not None:
            self._set_provider(top_provider)

        canonical_model = _identifier(evidence.get("canonicalSlug"))
        provider_api_model = _identifier(evidence.get("resolvedProviderApiModelId"))
        for model in (canonical_model, provider_api_model):
            if model is None:
                continue
            self._record_model(model)
            if not models_match(
                self.requested_model, model, self.model_match
            ):
                self.profile_reasons.append("served_model_conflict")
        resolved_model = canonical_model or provider_api_model
        if resolved_model is not None:
            self._set_model(resolved_model)
        elif top_model is not None:
            self._set_model(top_model)
        if (
            top_model is not None
            and resolved_model is not None
            and not model_evidence_consistent(
                top_model, resolved_model, self.model_match
            )
        ):
            self.profile_reasons.append("served_model_conflict")

        counts = [
            _integer(evidence.get(key))
            for key in ("modelAttemptCount", "totalProviderAttemptCount")
            if key in evidence
        ]
        if not counts:
            self.profile_reasons.append("missing_attempt_count")
        elif any(count != 1 for count in counts):
            self.profile_reasons.append("multiple_attempts")

        model_attempts = evidence.get("modelAttempts")
        if isinstance(model_attempts, Mapping):
            model_attempts = [model_attempts]
        if not isinstance(model_attempts, list) or not model_attempts:
            self.profile_reasons.append("missing_model_attempts")
            return True
        if len(model_attempts) != 1:
            self.profile_reasons.append("multiple_attempts")
        provider_attempts = []
        self.attempts = []
        for model_attempt in model_attempts:
            if not isinstance(model_attempt, Mapping):
                self.attempts_malformed = True
                continue
            model_attempt_model = (
                _identifier(model_attempt.get("canonicalSlug"))
                or resolved_model
            )
            if model_attempt_model is not None:
                self._record_model(model_attempt_model)
            if (
                model_attempt_model is not None
                and resolved_model is not None
                and not model_evidence_consistent(
                    model_attempt_model, resolved_model, self.model_match
                )
            ):
                self.profile_reasons.append("served_model_conflict")
            raw = model_attempt.get("providerAttempts")
            if not isinstance(raw, list):
                self.attempts_malformed = True
                continue
            provider_attempts.extend(
                (attempt, model_attempt_model, model_attempt.get("success"))
                for attempt in raw
            )
        self.attempts_present = True
        if len(provider_attempts) != 1:
            self.profile_reasons.append("multiple_attempts")
        successful_attempts = 0
        for raw, model_attempt_model, model_attempt_success in provider_attempts:
            if not isinstance(raw, Mapping):
                self.attempts_malformed = True
                continue
            provider = _identifier(raw.get("provider"))
            model = (
                _identifier(raw.get("resolvedProviderApiModelId"))
                or _identifier(raw.get("providerApiModelId"))
                or model_attempt_model
                or resolved_model
            )
            status = _integer(raw.get("statusCode"))
            provider_attempt_success = raw.get("success")
            if (
                provider_attempt_success is not None
                and not isinstance(provider_attempt_success, bool)
            ):
                self.attempts_malformed = True
            if (
                model_attempt_success is not None
                and not isinstance(model_attempt_success, bool)
            ):
                self.attempts_malformed = True
            if (
                isinstance(provider_attempt_success, bool)
                and isinstance(model_attempt_success, bool)
                and provider_attempt_success != model_attempt_success
            ):
                self.attempts_malformed = True
            if status is None and provider_attempt_success is True:
                status = 200
            status_success = status is not None and 200 <= status < 300
            if provider_attempt_success is True and not status_success:
                self.attempts_malformed = True
            if provider_attempt_success is False and status_success:
                self.attempts_malformed = True
            successful = (
                status_success
                and provider_attempt_success is not False
                and model_attempt_success is not False
            )
            if (
                model_attempt_success is False
                and provider_attempt_success is True
            ):
                self.attempts_malformed = True
            if successful:
                successful_attempts += 1
            attempt = {}
            if provider is not None:
                attempt["provider"] = _persisted_provider(provider)
            if model is not None:
                attempt["model"] = model
                self._record_model(model)
            if status is not None:
                attempt["status"] = status
            self.attempts.append(attempt)
            if not provider or not model or status is None:
                self.attempts_malformed = True
        if len(self.attempts) != 1:
            self.profile_reasons.append("multiple_attempts")
        if successful_attempts != 1:
            self.profile_reasons.append("missing_successful_attempt")

        for key in ("generationId", "cost", "marketCost"):
            value = metadata.get(key)
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                self.safe_metadata[key] = value
        return True

    def route_reasons(self) -> list[str]:
        reasons = list(self.profile_reasons)
        required = (
            (self.metadata_seen, f"missing_{self.gateway}_metadata"),
            (self.served_model, "missing_served_model"),
            (self.provider, "missing_provider"),
        )
        reasons.extend(reason for value, reason in required if not value)
        if (
            self.served_model
            and not models_match(
                self.requested_model, self.served_model, self.model_match
            )
        ):
            reasons.append("served_model_not_allowed")
        if (
            self.provider
            and self.requested_provider
            and not self._provider_matches(
                self.provider, self.requested_provider
            )
        ):
            reasons.append("provider_conflict")
        allowed = {
            _provider_identity(
                _concentrate_provider_slug(provider)
                if self.gateway == "concentrate"
                else provider
            )
            for provider in self.allowed_providers
        }
        if (
            self.provider
            and allowed
            and _provider_identity(self.provider) not in allowed
        ):
            reasons.append("provider_not_allowed")
        if self.gateway in {"openrouter", "vercel"}:
            if not self.metadata_requested_model:
                reasons.append("missing_metadata_requested_model")
            elif self.metadata_requested_model != self.requested_model:
                reasons.append("requested_model_conflict")
        if self.gateway == "vercel" and not self.attempts_present:
            reasons.append("missing_attempt_evidence")
        if self.attempts_malformed:
            reasons.append("malformed_attempts")
        if (
            self.attempts_present
            and self.attempts
            and not any(
                isinstance(attempt.get("status"), int)
                and 200 <= attempt["status"] < 300
                for attempt in self.attempts
            )
        ):
            reasons.append("missing_successful_attempt")
        for attempt in self.attempts:
            provider = _identifier(attempt.get("provider"))
            model = _identifier(attempt.get("model"))
            status = _integer(attempt.get("status"))
            if provider is None:
                reasons.append("missing_attempt_provider")
            elif (
                self.provider
                and _provider_identity(provider)
                != _provider_identity(self.provider)
            ):
                reasons.append("fallback_attempt")
            elif allowed and _provider_identity(provider) not in allowed:
                reasons.append("attempt_provider_not_allowed")
            if model is None:
                reasons.append("missing_attempt_model")
            elif not models_match(
                self.requested_model, model, self.model_match
            ):
                reasons.append("fallback_attempt")
            if status is None:
                reasons.append("missing_attempt_status")
            elif not 200 <= status < 300:
                reasons.append("unsuccessful_attempt")
        return list(dict.fromkeys(reasons))

    def _provider_matches(self, observed: str, expected: str) -> bool:
        expected_slug = (
            _concentrate_provider_slug(expected)
            if self.gateway == "concentrate"
            else expected
        )
        return _provider_identity(observed) == _provider_identity(expected_slug)
