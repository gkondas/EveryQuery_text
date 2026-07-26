"""Prompt styles — the pluggable seam between a patient's event stream and the LLM prompt.

``EQ_llm_predict`` supports two serializations of the same tensorized MEDS history:

``event_stream`` (default)
    EveryQuery's own format: a chronological list of raw code strings, one per line, with a
    Yes/No question appended.  Open-vocabulary and lossless with respect to *which* codes
    occurred.  See :mod:`every_query.llm_baseline.serialize`.

``llm4healthcare``
    A faithful reproduction of the prompt from https://github.com/yhzhu99/llm4healthcare —
    feature-major rows over a fixed clinical panel, with units, reference ranges, few-shot
    examples and a float response.  Lossy by construction (only panel variables are shown)
    but directly comparable to that paper's published numbers.  See
    :mod:`every_query.llm_baseline.serialize_l4h`.

Both satisfy :class:`PromptStyle`, so the CLI, the dry-run printer and the predictor all work
against the protocol rather than against either module's free functions.  A style owns the
whole path from a subject's dense arrays to the final user turn, because the two formats
disagree about *where* the query appears: ``event_stream`` appends it after the history (so a
group's rows share a token prefix), while ``llm4healthcare`` places it in the task-description
paragraph near the top (so they do not — see the throughput note in ``__main__``).
"""

import logging
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Protocol

import numpy as np
from omegaconf import DictConfig, OmegaConf

from every_query.llm_baseline import serialize, serialize_l4h
from every_query.llm_baseline.serialize import SerializedHistory
from every_query.llm_baseline.serialize_l4h import L4HConfig, Panel, ValueDecoder, load_panel

logger = logging.getLogger(__name__)

PANELS = files("every_query") / "llm_baseline" / "panels"


class PromptStyle(Protocol):
    """One serialization of a patient history into a chat prompt."""

    name: str
    template_version: str
    #: ``"yes_no"`` (repeated-sampling vote) or ``"float"`` (parse a probability directly).
    response_format: str

    def template_hash(self) -> str:
        """Short fingerprint of every constant and knob that affects the rendered prompt."""
        ...

    def system_prompt(self) -> str:
        """The system turn."""
        ...

    def history_from_dense(
        self,
        dense: dict[str, np.ndarray],
        static_codes: list[int] | None,
        index_to_code: dict[int, str],
    ) -> SerializedHistory:
        """Render the query-independent portion of the prompt for one subject window."""
        ...

    def user_prompt(self, history_text: str, query_code: str, duration_days: float) -> str:
        """Assemble the full user turn for one ``(history, query, horizon)`` row."""
        ...

    def payload_fields(self) -> dict:
        """Style-specific fields recorded in output metadata and folded into the fingerprint."""
        ...


@dataclass
class EventStreamStyle:
    """EveryQuery's native code-stream serialization (:mod:`serialize`)."""

    max_events: int
    code_descriptions: dict[str, str] | None = None
    code_descriptions_path: str | None = None

    name: str = "event_stream"
    template_version: str = serialize.PROMPT_TEMPLATE_VERSION
    response_format: str = "yes_no"

    def template_hash(self) -> str:
        return serialize.prompt_template_hash()

    def system_prompt(self) -> str:
        return serialize.SYSTEM_PROMPT

    def history_from_dense(self, dense, static_codes, index_to_code) -> SerializedHistory:
        events = serialize.events_from_jnrt_dense(dense, index_to_code)
        return serialize.serialize_history(
            events, max_events=self.max_events, code_descriptions=self.code_descriptions
        )

    def user_prompt(self, history_text: str, query_code: str, duration_days: float) -> str:
        question = serialize.serialize_question(query_code, duration_days, self.code_descriptions)
        return serialize.build_user_prompt(history_text, question)

    def payload_fields(self) -> dict:
        return {
            "max_events": int(self.max_events),
            "code_descriptions_used": self.code_descriptions is not None,
            "code_descriptions_path": self.code_descriptions_path,
        }


@dataclass
class LLM4HealthcareStyle:
    """The llm4healthcare panel prompt (:mod:`serialize_l4h`)."""

    panel: Panel
    decoder: ValueDecoder
    cfg: L4HConfig
    panel_path: str | None = None

    name: str = "llm4healthcare"
    template_version: str = serialize_l4h.PROMPT_TEMPLATE_VERSION

    @property
    def response_format(self) -> str:
        return self.cfg.response_format

    def template_hash(self) -> str:
        return serialize_l4h.prompt_template_hash(self.panel, self.cfg)

    def system_prompt(self) -> str:
        return serialize_l4h.SYSTEM_PROMPT

    def history_from_dense(self, dense, static_codes, index_to_code) -> SerializedHistory:
        return serialize_l4h.serialize_history(
            dense, static_codes, index_to_code, self.panel, self.decoder, self.cfg
        )

    def user_prompt(self, history_text: str, query_code: str, duration_days: float) -> str:
        return serialize_l4h.build_user_prompt(history_text, query_code, duration_days, self.panel, self.cfg)

    def payload_fields(self) -> dict:
        return {
            "panel": self.panel.name,
            "panel_path": self.panel_path,
            "panel_digest": self.panel.spec_digest(),
            "form": self.cfg.form,
            "impute": self.cfg.impute,
            "unit": self.cfg.unit,
            "reference_range": self.cfg.reference_range,
            "max_visits": int(self.cfg.max_visits),
            "value_precision": int(self.cfg.value_precision),
            "record_time_mode": self.cfg.record_time_mode,
            "n_shot": len(self.cfg.examples),
            "value_decoder_codes": len(self.decoder.stats),
        }


def _resolve_panel_path(spec: str) -> Path:
    """Resolve a panel spec: a bare name selects a bundled panel, a path is used as given."""
    candidate = Path(spec)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return candidate
    bundled = Path(str(PANELS)) / f"{spec}.yaml"
    if bundled.exists():
        return bundled
    available = sorted(p.stem for p in Path(str(PANELS)).glob("*.yaml"))
    raise FileNotFoundError(
        f"Panel spec {spec!r} not found — pass a path to a .yaml, or one of the bundled panels: {available}"
    )


def _load_examples(spec) -> tuple[str, ...]:
    """Load few-shot example blocks from a list of paths or inline strings."""
    if not spec:
        return ()
    out = []
    for entry in spec:
        path = Path(str(entry))
        out.append(path.read_text() if path.exists() else str(entry))
    return tuple(out)


def build_style(cfg: DictConfig, code_descriptions: dict[str, str] | None) -> PromptStyle:
    """Construct the :class:`PromptStyle` selected by ``cfg.prompt_style``."""
    match cfg.prompt_style:
        case "event_stream":
            return EventStreamStyle(
                max_events=int(cfg.max_events),
                code_descriptions=code_descriptions,
                code_descriptions_path=cfg.code_descriptions,
            )
        case "llm4healthcare":
            panel_path = _resolve_panel_path(str(cfg.l4h.panel))
            panel = load_panel(panel_path)
            decoder = ValueDecoder.from_codes_parquet(
                Path(cfg.tensorized_cohort_dir) / "metadata" / "codes.parquet"
            )
            l4h_cfg = L4HConfig(
                form=cfg.l4h.form,
                impute=cfg.l4h.impute,
                unit=bool(cfg.l4h.unit),
                reference_range=bool(cfg.l4h.reference_range),
                max_visits=int(cfg.l4h.max_visits),
                value_precision=int(cfg.l4h.value_precision),
                record_time_mode=cfg.l4h.record_time_mode,
                response_format=cfg.l4h.response_format,
                examples=_load_examples(OmegaConf.to_container(cfg.l4h.examples)),
            )
            style = LLM4HealthcareStyle(panel=panel, decoder=decoder, cfg=l4h_cfg, panel_path=str(panel_path))
            logger.info(
                f"Prompt style 'llm4healthcare': panel {panel.name} "
                f"({len(panel.features)} features) from {panel_path}; "
                f"value decoder covers {len(decoder.stats)} binned codes; "
                f"form={l4h_cfg.form} impute={l4h_cfg.impute} "
                f"response_format={l4h_cfg.response_format} n_shot={len(l4h_cfg.examples)}"
            )
            return style
        case _:
            raise ValueError(
                f"Unknown prompt_style {cfg.prompt_style!r} — expected 'event_stream' or 'llm4healthcare'."
            )
