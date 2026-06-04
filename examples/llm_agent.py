"""Arena PokerKit — Level 5 runtime-LLM agent (Competition Edition).

Claude (or any OpenAI-compatible model) decides every action. Falls back to
the enhanced L4 heuristic on tight deadlines or API failures.

Key improvements over baseline llm_agent.py:
  - Competition-grade system prompt with GTO ranges, texture sizing, 3-bet game
  - Default model: claude-sonnet-4-6 (strongest available)
  - Haiku fallback available via --model haiku for development runs
  - Better context compaction (street history, pot geometry, SPR)
  - Reasoning validation improved

CLI:
    uv run examples/llm_agent.py
    uv run examples/llm_agent.py --dry-run --mock-llm
    uv run examples/llm_agent.py --model haiku    # cheaper for testing
    uv run examples/llm_agent.py --max-hands 50   # preview run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

from agent import (  # type: ignore
    _build_reasoning,
    _detect_position,
    _board_texture,
    _has_draw,
    decide as heuristic_decide,
    retrieve_solver_context,
    run_live_benchmark,
)


def _compute_spr(stack: int, pot: int) -> float:
    return stack / max(pot, 1)


# ─── Competition-grade system prompt ──────────────────────────────────────────

SYSTEM_PROMPT = """You are a No-Limit Texas Hold'em poker agent competing for a $2500 prize at dev.fun Arena Poker Eval.

Your strategy is based on Nathan "BlackRain79" Williams' TAG (Tight-Aggressive) system from "Massive Profit at the Micros" — proven to crush passive, calling-station opponents.

You face 5 reference bots. Adjust to each:
- "anchor-checkcall" / calling station: VALUE BET RELENTLESSLY every street. NEVER bluff. They will call with any pair.
- "anchor-fold" / tight-passive bot: C-BET EVERYTHING. Represent scare cards. They fold too much.
- "anchor-random" / random bot: treat as calling station — value bet, don't bluff.
- DeepCFR / solver bot: play standard TAG fundamentals, don't get fancy.
- Unknown: default to value-heavy TAG, minimal bluffing.

═══════════════════════════════════════════
PREFLOP — BlackRain79 TAG Rules
═══════════════════════════════════════════

OPENING RANGE (top 20%, 6-max) — ALWAYS raise 3x BB, NEVER limp:
  Pairs: 22-AA (all of them)
  Suited aces: A2s-AKs
  Suited broadways/connectors: KQs, KJs, KTs, QJs, QTs, JTs, J9s, T9s, T8s, 98s, 87s, 76s, 65s
  Offsuit: AKo, AQo, AJo, ATo, KQo, KJo, QJo
  FOLD EVERYTHING ELSE preflop — discipline is the foundation of this strategy.

3-BET SIZING: 3x their raise if IN POSITION, 4x their raise if OUT OF POSITION.
3-BET VALUE HANDS (re-raise vs open): AA, KK, QQ, JJ, AKs, AKo, AQs
  ⚠ EXCEPTION: vs EP (early position) raiser → FLAT CALL instead of 3-betting (they likely have AA/KK/QQ)
  ⚠ EXCEPTION: vs player who folds to 3-bets often → flat call to keep them in pot

FLAT CALL range (call vs single raise, too good to fold, not 3-bet hands):
  Pairs 22-TT, AJo/ATo/KQo/KJo/QJo, suited connectors (JTs-65s), suited aces (A2s-A9s)
  Minimize OOP flat calls — fold marginal hands when out of position.

vs 4-BET: ONLY continue with AA, KK, AK, QQ. Fold everything else.
  "At the lower limits, when they 4-bet you, they have a monster. Only continue with the nuts." — BR79

═══════════════════════════════════════════
POSTFLOP — BlackRain79 TAG Rules
═══════════════════════════════════════════

C-BET (continuation bet) RULES — flop, ~2/3 of the time:
  ✅ C-bet with: ANY pair, ANY draw, ace-high IP, king-high IP, queen-high IP
  ✅ C-bet sizing: 1/2 pot on dry boards, 2/3 pot on wet/coordinated boards
  ❌ Do NOT c-bet: OOP with nothing on wet boards vs calling stations — just give up
  ❌ Do NOT c-bet vs multiple opponents unless you have a strong hand

FAST-PLAY STRONG HANDS (CRITICAL BR79 principle):
  When facing a bet with top pair top kicker or better → RAISE IMMEDIATELY, do not call.
  "Don't slow play at the micros. Fast-play your big hands. Build the pot for them." — BR79

FLAT CALL on flop (facing a bet):
  ✅ Call with: top pair weak kicker, middle pair, flush draw, OESD
  ❌ Fold: bottom pair, weak gutshot, nothing — avoid Fancy Play Syndrome

TURN (double barrel):
  ✅ Barrel again (~50% frequency) with: solid made hand OR strong draw (flush draw, OESD)
  ✅ Bluff barrel ONLY when: dry/uncoordinated flop + A or K appears on turn + opponent is tight/passive
  ❌ Do NOT double barrel vs calling stations with nothing — they will call

RIVER (triple barrel):
  ✅ Triple barrel only with very strong hand (top pair top kicker or better)
  ✅ Value bet thin (top pair, even middle pair) when action slowed earlier (a street was checked)
  ❌ NEVER bluff calling stations on the river — they always call

═══════════════════════════════════════════
KEY EXPLOITATION PRINCIPLE
═══════════════════════════════════════════

"The reason why most people struggle at the micros is that they run big bluffs against
calling stations. These players simply aren't going to fold their pair. Stop trying to
bluff them and just get value when you have a hand." — BR79

vs calling station opponents: value bet top pair, middle pair, even bottom pair for 2-3 streets.
vs tight/passive opponents: bluff freely when they show weakness, represent the scare cards.

═══════════════════════════════════════════
OUTPUT FORMAT — CRITICAL
═══════════════════════════════════════════

⚠ DO NOT write any analysis, explanation, or reasoning text. Output ONLY the JSON.

The ENTIRE response must be this single JSON object and nothing else:

{"action": "<fold|check|call|bet|raise|all-in>", "amount": <int or omit>, "message": "<≤500 chars>", "reasoning": "<≤150 chars YAML flow>"}

Rules:
1. action MUST be in allowedActions.availableActions
2. For bet/raise/all-in: amount = total chips committed this street (within betRange/raiseRange min-max)
3. For fold/check/call: OMIT amount field entirely
4. reasoning: YAML flow style ≤150 chars — {vr: "range", ke: "XX% eq", bf: [dry|FD-h|paired], pp: "pos plan", sr: "sizing"}
5. message: one short sentence, never reveal hole cards
6. NO prose before or after the JSON. The JSON is your entire response.
"""


# ─── Mock LLM ─────────────────────────────────────────────────────────────────

class _MockLLMResponse:
    def __init__(self, text: str) -> None:
        class _Block:
            def __init__(self, t: str) -> None:
                self.type = "text"
                self.text = t
        self.content = [_Block(text)]


class _MockAnthropic:
    def __init__(self, *_, **__) -> None:
        self.messages = self

    def create(self, **kwargs) -> _MockLLMResponse:
        text = json.dumps({
            "action": "call",
            "message": "mock LLM: calling for pot odds",
            "reasoning": '{vr: "std", ke: "55% eq", bf: [dry], pp: "IP call", sr: "po 25% covered"}',
        })
        return _MockLLMResponse(text)


_MOCK_LLM = False


def _maybe_mock_anthropic_module():
    if not _MOCK_LLM:
        return None
    class _Mod:
        Anthropic = _MockAnthropic
    return _Mod()


# ─── LLM provider call ────────────────────────────────────────────────────────

# Model aliases for convenience
_MODEL_ALIASES = {
    "sonnet":  "claude-sonnet-4-6",
    "opus":    "claude-opus-4-6",
    "haiku":   "claude-haiku-4-5-20251001",
    "sonnet45": "claude-sonnet-4-5",
}


def _resolve_model(model_hint: Optional[str]) -> Optional[str]:
    if model_hint is None:
        return None
    return _MODEL_ALIASES.get(model_hint.lower(), model_hint)


def _call_llm(system: str, user: str, max_tokens: int,
              model_hint: Optional[str], mock_mod=None) -> Optional[str]:
    """Model-agnostic LLM call. Returns text on success, None on failure."""
    resolved_model = _resolve_model(model_hint)

    if mock_mod is not None:
        client = mock_mod.Anthropic()
        resp = client.messages.create(
            model="mock", max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}])
        return "".join(getattr(b, "text", "") for b in resp.content
                       if getattr(b, "type", None) == "text").strip()

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # type: ignore
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=resolved_model or "claude-sonnet-4-6",
                max_tokens=max_tokens,
                system=system,
                messages=[
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": "{"},  # prefill — forces JSON-only output
                ])
            raw = "".join(getattr(b, "text", "") for b in resp.content
                          if getattr(b, "type", None) == "text").strip()
            return "{" + raw  # re-attach the prefill character
        except Exception as e:
            print(f"[arena-pokerkit] Anthropic call failed: {e}", file=sys.stderr)
            return None

    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI()
            resp = client.chat.completions.create(
                model=resolved_model or "gpt-4o",
                max_completion_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ])
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[arena-pokerkit] OpenAI call failed: {e}", file=sys.stderr)
            return None

    return None


# ─── Table state compactor ────────────────────────────────────────────────────

def _compact_table(table: dict) -> dict:
    """Compress table state into a minimal, information-dense prompt payload."""
    allowed = table.get("allowedActions") or {}
    self_num = table.get("selfSeatNumber")
    seats = table.get("seats") or []
    self_seat = next((s for s in seats if s.get("seatNumber") == self_num), {})
    bb = max(int(table.get("bigBlindChips") or 20), 1)
    pot = int(table.get("potChips") or 0)
    stack = int(self_seat.get("stackChips") or 0)
    board = list(table.get("boardCards") or [])

    # Board analysis for context
    texture = _board_texture(board)
    hole = list(self_seat.get("holeCards") or [])
    draws = _has_draw(hole, board) if board else {}
    spr = _compute_spr(stack, pot)
    position = _detect_position(table)

    opponents = []
    for s in seats:
        if s.get("seatNumber") != self_num:
            opp_stack = int(s.get("stackChips") or 0)
            opponents.append({
                "seat": s.get("seatNumber"),
                "stack": opp_stack,
                "stackBB": round(opp_stack / bb, 1),
                "bet": s.get("currentBetChips"),
                "status": s.get("status"),
            })

    compact = {
        "street": table.get("street"),
        "position": position,
        "pot": pot,
        "potBB": round(pot / bb, 1),
        "spr": round(spr, 1),
        "board": board,
        "boardTexture": texture.get("type"),
        "hero": {
            "seat": self_num,
            "hole": hole,
            "stack": stack,
            "stackBB": round(stack / bb, 1),
            "currentBet": self_seat.get("currentBetChips"),
        },
        "opponents": opponents,
        "actions": {
            "available": allowed.get("availableActions"),
            "callChips": allowed.get("callChips"),
            "callToAmount": allowed.get("callToAmount"),
            "betRange": allowed.get("betRange"),
            "raiseRange": allowed.get("raiseRange"),
            "callPotOdds": (
                round(allowed.get("callChips", 0) / max(pot + allowed.get("callChips", 0), 1), 3)
                if allowed.get("callChips") else 0
            ),
        },
        "bb": bb,
    }

    # Add draw info if relevant
    if draws:
        draw_summary = []
        if draws.get("flush_draw"):
            draw_summary.append("flush_draw")
        if draws.get("oesd"):
            draw_summary.append("oesd")
        if draws.get("gutshot"):
            draw_summary.append("gutshot")
        if draws.get("backdoor_flush"):
            draw_summary.append("backdoor_flush")
        if draw_summary:
            compact["draws"] = draw_summary
            compact["drawEquity"] = draws.get("draw_equity")

    # Last 6 events for street context
    events = (table.get("recentEvents") or [])[-6:]
    if events:
        compact["recentEvents"] = [
            {"type": e.get("type"), "summary": e.get("summary")}
            for e in events
        ]

    return compact


# ─── JSON parsing ─────────────────────────────────────────────────────────────

def _strip_code_fences(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    nl = s.find("\n")
    if nl >= 0:
        s = s[nl + 1:]
    else:
        s = s.lstrip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    if s.rstrip().endswith("```"):
        s = s.rstrip()[:-3]
    return s.strip()


def _extract_balanced_json(text: str) -> Optional[str]:
    """Find all balanced {...} blocks and return the last one containing 'action'."""
    results = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                results.append(text[start:i + 1])
    # Prefer the last block that looks like an action object
    for candidate in reversed(results):
        if '"action"' in candidate:
            return candidate
    return results[-1] if results else None


def _parse_action_json(text: str) -> Optional[dict]:
    s = _strip_code_fences(text)
    obj: Any = None
    try:
        obj = json.loads(s)
    except Exception:
        chunk = _extract_balanced_json(s)
        if chunk is None:
            return None
        try:
            obj = json.loads(chunk)
        except Exception:
            return None
    if not isinstance(obj, dict):
        return None
    if "action" not in obj or "message" not in obj or "reasoning" not in obj:
        return None
    obj["message"] = str(obj["message"])[:500]
    obj["reasoning"] = str(obj["reasoning"])
    return obj


def _validate_against_allowed(action: dict, table: dict) -> dict:
    allowed = table.get("allowedActions") or {}
    available = set(allowed.get("availableActions") or [])
    name = action.get("action")

    if name not in available:
        if "check" in available:
            action["action"] = "check"
            action.pop("amount", None)
        else:
            action["action"] = "fold"
            action.pop("amount", None)
        return action

    if name in ("fold", "check", "call"):
        action.pop("amount", None)
        return action

    try:
        amount = int(action.get("amount") or 0)
    except (TypeError, ValueError):
        amount = 0

    if name == "bet":
        rng = allowed.get("betRange") or {}
    elif name == "raise":
        rng = allowed.get("raiseRange") or {}
    else:  # all-in
        rng = {"min": allowed.get("allInToAmount"), "max": allowed.get("allInToAmount")}

    lo = int((rng or {}).get("min") or amount or 0)
    hi = int((rng or {}).get("max") or amount or lo)
    if lo and hi:
        amount = max(lo, min(amount or lo, hi))
    action["amount"] = amount
    return action


# ─── Main LLM decide ─────────────────────────────────────────────────────────

_FALLBACK_REASONING = '{vr: "std", ke: "legal", pp: "pot control"}'


def llm_decide(table: dict, deadline_s: float = 10.0,
               model: Optional[str] = None,
               max_tokens: int = 600,
               research_context: Optional[dict] = None) -> dict:
    """Call LLM for action. Falls back to L4 heuristic on failure/timeout."""
    mock_mod = _maybe_mock_anthropic_module()
    has_provider = (
        mock_mod is not None
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )

    if not has_provider or deadline_s < 3.0:
        return heuristic_decide(table, deadline_s=deadline_s,
                                research_context=research_context)

    compact = _compact_table(table)
    prompt = "TABLE STATE:\n" + json.dumps(compact, separators=(",", ":"), indent=None)

    if research_context:
        prompt += "\n\nAUTO-RESEARCH CONTEXT:\n" + json.dumps(research_context, separators=(",", ":"))

    prompt += "\n\nRespond with ONLY the JSON action object on the last line."

    text = _call_llm(SYSTEM_PROMPT, prompt, max_tokens, model, mock_mod)
    if not text:
        return heuristic_decide(table, deadline_s=deadline_s, research_context=research_context)

    action = _parse_action_json(text)
    if action is None:
        print(f"[arena-pokerkit] LLM parse failed, using heuristic. LLM output: {text[:200]}",
              file=sys.stderr)
        return heuristic_decide(table, deadline_s=deadline_s, research_context=research_context)

    action = _validate_against_allowed(action, table)

    # Validate reasoning
    reasoning = action.get("reasoning", "") or ""
    valid_reasoning = (
        reasoning.startswith("{") and
        reasoning.endswith("}") and
        len(reasoning) <= 150 and
        all(k in reasoning for k in ("vr:", "ke:", "pp:"))
    )
    if not valid_reasoning:
        allowed = table.get("allowedActions") or {}
        action["reasoning"] = _build_reasoning(
            action.get("action", "fold"),
            0.0, 0.0, table, allowed,
        )

    return action


def decide(table: dict, deadline_s: float = 10.0,
           research_context: Optional[dict] = None) -> dict:
    """Module-level decide() — forwards to llm_decide() with defaults.
    This lets `./pokerkit run --agent examples/llm_agent.py` work."""
    return llm_decide(table, deadline_s=deadline_s,
                      research_context=research_context)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    global _MOCK_LLM
    parser = argparse.ArgumentParser(
        description="Arena PokerKit — Level 5 LLM agent (competition edition)")
    parser.add_argument("--competition-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-scenario", choices=("instant", "queued", "stale"),
                        default="instant")
    parser.add_argument("--mock-llm", action="store_true",
                        help="Mock LLM for offline testing of the parse/validate path")
    parser.add_argument("--max-hands", type=int, default=0)
    parser.add_argument("--model", default=None,
                        help="Model to use: sonnet (default), haiku, opus, or full model string")
    parser.add_argument("--handle", default="adah-llm-agent")
    parser.add_argument("--name", default="Adah LLM Agent")
    parser.add_argument("--quote", default="Claude thinks, probability wins")
    args = parser.parse_args(argv)

    _MOCK_LLM = bool(args.mock_llm)

    def _decide(table: dict, deadline_s: float = 10.0,
                research_context: Optional[dict] = None) -> dict:
        return llm_decide(table, deadline_s=deadline_s,
                          model=args.model, research_context=research_context)

    if args.dry_run:
        from mock import run_mock_benchmark
        return run_mock_benchmark(args, decide_fn=_decide,
                                  retrieve_solver_context=retrieve_solver_context)
    return run_live_benchmark(args, decide_fn=_decide)


if __name__ == "__main__":
    sys.exit(main())
