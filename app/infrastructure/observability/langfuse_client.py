from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Protocol


class SupportsObservation(Protocol):
    def update(self, **kwargs: Any) -> Any:
        ...

    def end(self) -> Any:
        ...


class SupportsLangfuseClient(Protocol):
    def start_as_current_observation(
        self,
        **kwargs: Any,
    ) -> AbstractContextManager[SupportsObservation]:
        ...

    def start_observation(self, **kwargs: Any) -> SupportsObservation:
        ...

    def flush(self) -> Any:
        ...


@dataclass(slots=True)
class ObservationHandle:
    observation: SupportsObservation
    trace_id: str | None = None

    def update(self, **kwargs: Any) -> None:
        self.observation.update(**kwargs)

    def end(self) -> None:
        self.observation.end()


@dataclass(slots=True)
class RootObservationHandle:
    observation: SupportsObservation
    observation_context: AbstractContextManager[SupportsObservation]
    attributes_context: AbstractContextManager[object]
    trace_id: str | None = None

    def update(self, **kwargs: Any) -> None:
        self.observation.update(**kwargs)

    def close(self) -> None:
        try:
            self.observation_context.__exit__(None, None, None)
        finally:
            self.attributes_context.__exit__(None, None, None)


def _build_default_langfuse_client(
    *,
    public_key: str,
    secret_key: str,
    base_url: str,
    environment: str,
    release: str | None,
    sample_rate: float,
) -> SupportsLangfuseClient:
    from langfuse import Langfuse

    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=base_url,
        environment=environment,
        release=release,
        sample_rate=sample_rate,
    )


def _build_default_attribute_context(**kwargs: Any) -> AbstractContextManager[object]:
    from langfuse import propagate_attributes

    return propagate_attributes(**kwargs)


class LangfuseClient:
    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        base_url: str,
        environment: str,
        release: str | None,
        sample_rate: float,
        client_factory=_build_default_langfuse_client,
        attribute_context_factory=_build_default_attribute_context,
    ) -> None:
        self._client = client_factory(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
            environment=environment,
            release=release,
            sample_rate=sample_rate,
        )
        self._attribute_context_factory = attribute_context_factory

    def start_root_observation(
        self,
        *,
        name: str,
        input_payload: dict[str, object],
        metadata: dict[str, object],
        user_id: str | None,
        session_id: str | None,
        environment: str,
        release: str | None,
    ) -> RootObservationHandle:
        attributes: dict[str, object] = {}
        if user_id is not None:
            attributes["user_id"] = user_id
        if session_id is not None:
            attributes["session_id"] = session_id
        if metadata:
            attributes["metadata"] = metadata
        if release is not None:
            attributes["version"] = release
        attributes["trace_name"] = name
        attributes["environment"] = environment
        attributes_context = (
            self._attribute_context_factory(**attributes)
            if attributes
            else nullcontext()
        )
        attributes_context.__enter__()
        observation_context = self._client.start_as_current_observation(
            as_type="span",
            name=name,
            input=input_payload,
            metadata=metadata,
            version=release,
        )
        observation = observation_context.__enter__()
        return RootObservationHandle(
            observation=observation,
            observation_context=observation_context,
            attributes_context=attributes_context,
            trace_id=self._extract_trace_id(observation),
        )

    def start_observation(
        self,
        *,
        name: str,
        as_type: str = "span",
        input_payload: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
        version: str | None = None,
        model: str | None = None,
        model_parameters: dict[str, object] | None = None,
    ) -> ObservationHandle:
        observation = self._client.start_observation(
            name=name,
            as_type=as_type,
            input=input_payload,
            metadata=metadata,
            version=version,
            model=model,
            model_parameters=model_parameters,
        )
        return ObservationHandle(
            observation=observation,
            trace_id=self._extract_trace_id(observation),
        )

    def flush(self) -> None:
        self._client.flush()

    def _extract_trace_id(self, observation: object) -> str | None:
        for candidate in (
            getattr(observation, "trace_id", None),
            getattr(getattr(observation, "trace", None), "id", None),
            getattr(getattr(observation, "trace", None), "trace_id", None),
        ):
            if isinstance(candidate, str):
                normalized = candidate.strip()
                if normalized:
                    return normalized
        return None
