from __future__ import annotations

from dataclasses import dataclass, field

from app.config import get_settings


@dataclass
class TurnDecision:
    """Outcome of rule evaluation for one chat turn (before/after generation)."""

    action: str
    confidence: float  # 0.0 .. 1.0
    rules_fired: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)


def evaluate_turn(
    *,
    user_content: str,
    memory_hit_count: int,
    llm_succeeded: bool,
    user_embedding_ok: bool,
) -> TurnDecision:
    """
    Rule-based scoring for the assistant turn. Extensible: add predicates and
    combine confidence with caps/floors.
    """
    settings = get_settings()
    rules: list[str] = []
    ctx: dict = {
        "memory_hit_count": memory_hit_count,
        "memory_top_k": settings.memory_top_k,
        "llm_succeeded": llm_succeeded,
        "user_embedding_ok": user_embedding_ok,
        "has_openai_key": bool(settings.openai_api_key),
    }

    text = user_content.strip()
    lower = text.lower()

    if not settings.openai_api_key:
        rules.append("no_openai_key")
        return TurnDecision(
            action="respond_placeholder",
            confidence=0.25,
            rules_fired=rules,
            context=ctx,
        )

    if len(text) < 12:
        rules.append("short_user_input")

    if any(w in lower for w in ("help", "stuck", "error", "how do i")):
        rules.append("help_intent")

    if memory_hit_count >= max(3, settings.memory_top_k // 2):
        rules.append("strong_memory_context")

    if not user_embedding_ok:
        rules.append("user_embedding_missing")

    if llm_succeeded:
        base = 0.72
        base += min(0.18, memory_hit_count * 0.03)
        if "strong_memory_context" in rules:
            base += 0.05
        if "short_user_input" in rules:
            base -= 0.08
        if "user_embedding_missing" in rules:
            base -= 0.06
        if "help_intent" in rules:
            base += 0.02
        conf = max(0.15, min(0.95, base))
        rules.append("llm_primary")
        return TurnDecision(
            action="respond_llm",
            confidence=round(conf, 4),
            rules_fired=rules,
            context=ctx,
        )

    rules.append("llm_failed_fallback")
    conf = 0.38
    if memory_hit_count >= 2:
        conf += 0.05
    return TurnDecision(
        action="respond_placeholder",
        confidence=round(min(0.55, conf), 4),
        rules_fired=rules,
        context=ctx,
    )
