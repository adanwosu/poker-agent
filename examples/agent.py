"""Arena PokerKit — BlackRain79 TAG Strategy Agent (Competition Edition).

Strategy source: "Massive Profit at the Micros" by Nathan "BlackRain79" Williams.

Core philosophy:
  - Play tight (top 20% of hands) and aggressive (always raise, never limp)
  - Fast-play strong hands — never slow play at micro/low stakes
  - Value bet relentlessly vs calling stations; don't try to bluff them
  - Fold with nothing — avoid Fancy Play Syndrome
  - Position is everything — play tighter OOP, wider IP

Preflop:
  - Open-raise 3x BB, top 20% of hands
  - 3-bet premiums (AA/KK/QQ/JJ/AK/AQs): 3x IP, 4x OOP
  - Flat call: small pairs, suited connectors, offsuit broadways
  - Don't 3-bet vs EP raise (they usually have a strong hand)
  - vs 4-bet: only continue with AA/KK/AK/QQ

Postflop:
  - C-bet ~2/3 of flops vs 1 opponent (1/2 to 2/3 pot)
  - C-bet with: any pair, any draw, Ax/Kx high in position
  - Fast-play TPTK+ (raise, not call, when facing a bet)
  - Flat call: top pair weak kicker, middle pair, draws
  - Fold: nothing on the board (no pair, no draw)
  - Double barrel turn ~50% (only with made hand or good draw)
  - Triple barrel river only with TPTK+
  - Value bet thin (top/mid pair) when pot was checked earlier

Opponent exploitation:
  - Calling stations: value bet every street, NEVER bluff
  - Folder types: c-bet everything, represent scare cards
  - Unknown/regulars: standard TAG approach
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Optional

from dotenv import load_dotenv

from arena_client import (
    ArenaClient,
    ArenaError,
    DEFAULT_BASE,
    append_iteration,
    assert_endpoints,
    fetch_introspection,
    load_or_register,
    load_state,
    resolve_terminal_phases,
    save_state,
)

try:
    from treys import Card as TreysCard, Evaluator as TreysEvaluator, Deck as TreysDeck
    _HAS_TREYS = True
except Exception:
    _HAS_TREYS = False


POLL_INTERVAL = 1.0
POLL_JITTER = 0.5
STATUS_REFRESH_S = 8.0

# ─── BlackRain79 Hand Ranges ──────────────────────────────────────────────────

# Top 20% of hands for 6-max — BlackRain79's recommended opening range.
_BR79_OPEN_RANGE: set[str] = {
    # All pocket pairs
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    # Suited aces
    "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    # Suited kings
    "KQs", "KJs", "KTs", "K9s", "K8s",
    # Suited queens — added Q9s
    "QJs", "QTs", "Q9s",
    # Suited jacks
    "JTs", "J9s",
    # Suited tens
    "T9s",
    # Suited nines
    "98s",
    # Suited eights
    "87s",
    # Offsuit premiums
    "AKo", "AQo", "AJo", "ATo",
    "KQo", "KJo",
    "QJo",
}

# 3-bet value range — re-raise with these vs an open.
# BlackRain79: "strong premium hands" = AA, KK, QQ, JJ + AK, AQs
_BR79_3BET_HANDS: set[str] = {
    "AA", "KK", "QQ", "JJ", "AKs", "AKo", "AQs",
}

# Flat call range — too good to fold vs a raise, not strong enough to 3-bet.
# BlackRain79: "Small/Mid pairs, Broadways (AJ/AT/KQ/KJ/QJ), Suited connectors"
_BR79_FLAT_HANDS: set[str] = {
    "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    "AJo", "ATo", "KQo", "KJo", "QJo",
    "KQs", "KJs", "KTs", "QJs", "QTs",
    "JTs", "T9s", "98s", "87s",
    "K9s", "K8s", "Q9s",
    "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
}

# Only continue vs 4-bet with these — BlackRain79: "only with the nuts"
_BR79_4BET_CONTINUE: set[str] = {"AA", "KK", "AKo", "AKs", "QQ"}

# Preflop equity table for fallback (when treys unavailable)
_PREFLOP_EQUITY = {
    "AA": 0.85, "KK": 0.82, "QQ": 0.80, "JJ": 0.77, "TT": 0.75,
    "99": 0.72, "88": 0.69, "77": 0.66, "66": 0.63, "55": 0.60,
    "44": 0.57, "33": 0.54, "22": 0.50,
    "AKs": 0.67, "AQs": 0.66, "AJs": 0.65, "ATs": 0.64, "A9s": 0.62,
    "A8s": 0.61, "A7s": 0.60, "A6s": 0.59, "A5s": 0.58, "A4s": 0.57,
    "A3s": 0.56, "A2s": 0.55,
    "KQs": 0.63, "KJs": 0.62, "KTs": 0.61, "K9s": 0.59,
    "QJs": 0.60, "QTs": 0.59, "JTs": 0.58, "J9s": 0.56,
    "T9s": 0.54, "T8s": 0.53, "98s": 0.52, "97s": 0.51,
    "87s": 0.52, "86s": 0.50, "76s": 0.50, "75s": 0.49, "65s": 0.49,
    "K8s": 0.57, "Q9s": 0.57,
    "AKo": 0.65, "AQo": 0.64, "AJo": 0.63, "ATo": 0.62,
    "KQo": 0.61, "KJo": 0.60, "QJo": 0.58,
}


# ─── Hand classification ──────────────────────────────────────────────────────

def _hand_class(hole: list[str]) -> str:
    ranks = "23456789TJQKA"
    if len(hole) != 2:
        return ""
    r1, s1 = hole[0][0].upper(), hole[0][-1].lower()
    r2, s2 = hole[1][0].upper(), hole[1][-1].lower()
    if r1 not in ranks or r2 not in ranks:
        return ""
    if ranks.index(r1) < ranks.index(r2):
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    if r1 == r2:
        return r1 + r2
    return f"{r1}{r2}{'s' if s1 == s2 else 'o'}"


# ─── Opponent type detection ──────────────────────────────────────────────────

def _detect_opponent_type(table: dict) -> str:
    """Detect opponent style from handle/name. This is key for BR79 exploitation:
    - 'caller': value bet everything, NEVER bluff
    - 'folder': c-bet aggressively, represent scare cards
    - 'random': treat as caller (safer)
    - 'regular': standard TAG response
    """
    self_num = table.get("selfSeatNumber")
    seats = table.get("seats") or []
    types = []
    for s in seats:
        if s.get("seatNumber") == self_num:
            continue
        if s.get("status") in ("folded", "out", "bust"):
            continue
        handle = (s.get("agentHandle") or "").lower()
        name = (s.get("agentName") or "").lower()
        combined = handle + " " + name
        if any(x in combined for x in ["checkcall", "check-call", "call", "passive"]):
            types.append("caller")
        elif any(x in combined for x in ["fold", "tight", "anchor-fold"]):
            types.append("folder")
        elif any(x in combined for x in ["random", "rng", "rand"]):
            types.append("random")
        elif any(x in combined for x in ["cfr", "deepcfr", "solver", "bot-poker", "pokerkit"]):
            types.append("regular")
        else:
            types.append("unknown")
    # If any opponent is a caller, be conservative (don't bluff)
    if "caller" in types or "random" in types:
        return "caller"
    if "folder" in types and "regular" not in types:
        return "folder"
    if "regular" in types:
        return "regular"
    return "caller"  # default: assume calling station (safer)


# ─── Position detection ───────────────────────────────────────────────────────

def _detect_position(table: dict) -> str:
    self_num = table.get("selfSeatNumber", 0)
    seats = table.get("seats") or []
    bb_chips = int(table.get("bigBlindChips") or 20)
    sb_chips = int(table.get("smallBlindChips") or 10)

    active = [s for s in seats if s.get("status") not in ("folded", "out", "bust")]
    active_nums = sorted(s.get("seatNumber", 0) for s in active)
    n = len(active_nums)
    if n < 2 or self_num not in active_nums:
        return "IP"

    btn_num = table.get("buttonSeatNumber") or table.get("dealerSeatNumber")

    if not btn_num and (table.get("street") or "") == "Preflop":
        sb_seat = None
        bb_seat = None
        for s in active:
            cbc = int(s.get("currentBetChips") or 0)
            if cbc == sb_chips and sb_seat is None:
                sb_seat = s.get("seatNumber")
            elif cbc == bb_chips and bb_seat is None:
                bb_seat = s.get("seatNumber")
        if sb_seat and sb_seat in active_nums:
            idx_sb = active_nums.index(sb_seat)
            btn_num = active_nums[(idx_sb - 1) % n]

    if btn_num and btn_num in active_nums:
        idx_btn = active_nums.index(btn_num)
        idx_self = active_nums.index(self_num)
        rel = (idx_self - idx_btn) % n
        if rel == 0:
            return "BTN"
        elif rel == 1:
            return "SB"
        elif rel == 2:
            return "BB"
        elif (n - rel) == 1:
            return "CO"
        elif (n - rel) == 2:
            return "HJ"
        else:
            return "UTG"

    idx_self = active_nums.index(self_num)
    rel_pos = idx_self / max(n - 1, 1)
    if rel_pos >= 0.8:
        return "BTN"
    elif rel_pos >= 0.6:
        return "CO"
    elif rel_pos >= 0.4:
        return "HJ"
    elif rel_pos <= 0.2:
        return "BB"
    return "IP" if rel_pos > 0.5 else "OOP"


def _is_ip(position: str) -> bool:
    return position in ("BTN", "CO", "IP", "HJ")


def _is_ep(position: str) -> bool:
    return position in ("UTG", "EP", "OOP")


# ─── Board texture ────────────────────────────────────────────────────────────

def _board_texture(board: list[str]) -> dict:
    if not board:
        return {"type": "dry", "wet": False, "monotone": False, "paired": False,
                "has_ace": False, "has_king": False}

    rank_order = "23456789TJQKA"
    suits = [c[-1].lower() for c in board if c]
    ranks = [c[0].upper() for c in board if c and c[0].upper() in rank_order]

    suit_counts: dict[str, int] = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1

    rank_counts: dict[str, int] = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    monotone = len(set(suits)) == 1 and len(suits) >= 2
    flush_possible = max(suit_counts.values(), default=0) >= 2
    paired = max(rank_counts.values(), default=0) >= 2
    has_ace = "A" in ranks
    has_king = "K" in ranks

    rank_idxs = sorted(rank_order.find(r) for r in ranks if rank_order.find(r) >= 0)
    connected = False
    if len(rank_idxs) >= 2:
        span = rank_idxs[-1] - rank_idxs[0]
        connected = span <= 4

    wet = flush_possible and connected
    if monotone:
        btype = "monotone"
    elif paired:
        btype = "paired"
    elif wet:
        btype = "wet"
    elif connected or flush_possible:
        btype = "semi_wet"
    else:
        btype = "dry"

    return {
        "type": btype, "wet": wet or monotone,
        "monotone": monotone, "paired": paired,
        "has_ace": has_ace, "has_king": has_king,
        "flush_possible": flush_possible, "connected": connected,
    }


# ─── Draw detection ───────────────────────────────────────────────────────────

def _has_draw(hole: list[str], board: list[str]) -> dict:
    if len(hole) != 2 or len(board) < 3:
        return {"flush_draw": False, "oesd": False, "gutshot": False,
                "backdoor_flush": False, "draw_equity": 0.0}

    all_cards = hole + board
    rank_order = "23456789TJQKA"

    hero_suits = {c[-1].lower() for c in hole if c}
    suit_counts: dict[str, int] = {}
    for c in all_cards:
        s = c[-1].lower()
        suit_counts[s] = suit_counts.get(s, 0) + 1

    flush_draw = any(cnt >= 4 and s in hero_suits for s, cnt in suit_counts.items())
    backdoor_flush = (not flush_draw and
                      any(cnt == 3 and s in hero_suits
                          for s, cnt in suit_counts.items()) and len(board) == 3)

    def ridx(c: str) -> int:
        return rank_order.find(c[0].upper())

    hero_ridxs = {ridx(c) for c in hole if ridx(c) >= 0}
    board_ridxs = {ridx(c) for c in board if ridx(c) >= 0}
    all_ridxs = hero_ridxs | board_ridxs

    oesd = False
    gutshot = False
    for start in range(9):
        straight = set(range(start, start + 5))
        have = straight & all_ridxs
        if len(have) == 4 and (straight & hero_ridxs):
            missing = sorted(straight - all_ridxs)
            if missing:
                if missing[0] == start or missing[0] == start + 4:
                    oesd = True
                else:
                    gutshot = True

    draw_equity = (
        0.54 if flush_draw and oesd else
        0.35 if flush_draw else
        0.32 if oesd else
        0.17 if gutshot else
        0.08 if backdoor_flush else 0.0
    )

    return {
        "flush_draw": flush_draw, "oesd": oesd,
        "gutshot": gutshot, "backdoor_flush": backdoor_flush,
        "draw_equity": draw_equity,
    }


# ─── Equity estimation ────────────────────────────────────────────────────────

def estimate_equity(hole: list[str], board: list[str],
                    sims: int = 250, deadline_s: float = 10.0) -> float:
    cls = _hand_class(hole)
    if not _HAS_TREYS or deadline_s < 2.0 or not board:
        return _PREFLOP_EQUITY.get(cls, 0.45)
    try:
        ev = TreysEvaluator()
        hero = [TreysCard.new(_to_treys(c)) for c in hole]
        board_t = [TreysCard.new(_to_treys(c)) for c in board]
        used = set(hero) | set(board_t)
        rng = random.Random(2026)
        wins = ties = 0
        for _ in range(sims):
            deck = TreysDeck()
            deck.cards = [c for c in deck.cards if c not in used]
            rng.shuffle(deck.cards)
            opp = [deck.cards.pop(), deck.cards.pop()]
            runout = []
            for _ in range(5 - len(board_t)):
                runout.append(deck.cards.pop())
            full_board = board_t + runout
            hero_rank = ev.evaluate(full_board, hero)
            opp_rank = ev.evaluate(full_board, opp)
            if hero_rank < opp_rank:
                wins += 1
            elif hero_rank == opp_rank:
                ties += 1
        return (wins + 0.5 * ties) / max(sims, 1)
    except Exception:
        return _PREFLOP_EQUITY.get(cls, 0.45)


def _to_treys(card_str: str) -> str:
    if not card_str:
        return "2c"
    r = card_str[0].upper()
    if card_str.startswith("10"):
        r = "T"
        s = card_str[2].lower() if len(card_str) > 2 else "x"
    else:
        s = card_str[-1].lower()
    return r + s


# ─── Sizing helpers ───────────────────────────────────────────────────────────

def _open_size(bb_chips: int, allowed: dict) -> int:
    """BR79: always open 3x BB."""
    target = int(bb_chips * 3)
    rr = allowed.get("raiseRange") or {}
    lo = int(rr.get("min") or target)
    hi = int(rr.get("max") or target * 4)
    return max(lo, min(target, hi))


def _3bet_size(call_chips: int, position: str, allowed: dict) -> int:
    """BR79: 3x the open if IP, 4x if OOP."""
    mult = 3.0 if _is_ip(position) else 4.0
    target = int(call_chips * mult)
    rr = allowed.get("raiseRange") or {}
    lo = int(rr.get("min") or target)
    hi = int(rr.get("max") or target * 2)
    return max(lo, min(target, hi))


def _cbet_size(pot: int, allowed: dict, big: bool = False) -> int:
    """BR79: 1/2 to 2/3 pot. Use 2/3 on wet boards, 1/2 on dry."""
    ratio = 0.66 if big else 0.50
    target = int(pot * ratio)
    br = allowed.get("betRange") or {}
    lo = int(br.get("min") or max(target // 2, 1))
    hi = int(br.get("max") or target * 2)
    return max(lo, min(target, hi))


def _raise_size(call_chips: int, pot: int, allowed: dict) -> int:
    """Raise to ~3x the bet + pot-sized pressure."""
    target = int(call_chips * 3 + pot * 0.5)
    rr = allowed.get("raiseRange") or {}
    lo = int(rr.get("min") or call_chips * 2)
    hi = int(rr.get("max") or lo * 4)
    return max(lo, min(target, hi))


# ─── Auto Research hook ───────────────────────────────────────────────────────

def retrieve_solver_context(table: dict) -> dict:
    """Return preflop chart context for the LLM."""
    if (table.get("street") or "Preflop") != "Preflop":
        return {}
    self_num = table.get("selfSeatNumber")
    seats = table.get("seats") or []
    self_seat = next((s for s in seats if s.get("seatNumber") == self_num), {})
    hole = list(self_seat.get("holeCards") or [])
    hc = _hand_class(hole)
    if not hc:
        return {}
    position = _detect_position(table)
    opp_type = _detect_opponent_type(table)
    in_range = hc in _BR79_OPEN_RANGE
    return {
        "br79_chart": {
            "hand": hc,
            "position": position,
            "in_open_range": in_range,
            "is_3bet_hand": hc in _BR79_3BET_HANDS,
            "is_flat_hand": hc in _BR79_FLAT_HANDS,
            "opponent_type": opp_type,
        }
    }


# ─── Preflop decision ─────────────────────────────────────────────────────────

def _decide_preflop(
    hole: list[str], allowed: dict, available: set, table: dict,
    call_chips: int, bb_chips: int, position: str, pot: int,
) -> dict:
    hc = _hand_class(hole)
    opp_type = _detect_opponent_type(table)

    # callToAmount tells us the total preflop raise size in chips
    call_to = int(allowed.get("callToAmount") or allowed.get("callAmount") or call_chips or 0)
    raise_in_bb = call_to / max(bb_chips, 1)

    # ── Facing a 4-bet (roughly > 10 BB) ──────────────────────────────────
    if raise_in_bb > 10:
        if hc in _BR79_4BET_CONTINUE:
            # Jam or call
            if "raise" in available or "all-in" in available:
                return _jam_or_call(allowed, available, table,
                                    eq=0.90, msg=f"5-bet jam, {hc} vs 4-bet")
            return _build("call", None, table, allowed, eq=0.88, po=0.0,
                          msg=f"calling 4-bet with {hc}")
        return _build("fold", None, table, allowed, eq=0.0, po=1.0,
                      msg=f"folding {hc} to 4-bet — BR79: only continue with nuts")

    # ── Facing a 3-bet (roughly 4-10 BB) ──────────────────────────────────
    if raise_in_bb > 4:
        if hc in {"AA", "KK"}:
            return _jam_or_call(allowed, available, table,
                                eq=0.90, msg=f"4-bet jam {hc}")
        if hc in _BR79_3BET_HANDS and "raise" in available:
            rr = allowed.get("raiseRange") or {}
            lo = int(rr.get("min") or call_chips * 2)
            hi = int(rr.get("max") or lo * 3)
            amount = max(lo, min(int(call_chips * 2.5), hi))
            return _build("raise", amount, table, allowed, eq=0.85, po=0.0,
                          msg=f"4-bet {hc} for value")
        if hc in {"QQ", "JJ", "TT", "AKo", "AKs", "AQs"} and "call" in available:
            return _build("call", None, table, allowed, eq=0.78,
                          po=call_chips / max(pot + call_chips, 1),
                          msg=f"calling 3-bet with {hc}")
        return _build("fold", None, table, allowed, eq=0.0, po=1.0,
                      msg=f"folding {hc} to 3-bet — BR79: fold non-premiums")

    # ── Facing a single open raise (roughly 2-4 BB) ──────────────────────
    if raise_in_bb > 1.5:
        # BR79 exception: vs EP raise, flat call even premiums (they likely have a big hand)
        raiser_is_ep = _raiser_is_ep(table)

        # Value 3-bet (JJ+/AK/AQs) — UNLESS vs EP raise
        if hc in _BR79_3BET_HANDS and not raiser_is_ep and "raise" in available:
            amount = _3bet_size(call_to, position, allowed)
            return _build("raise", amount, table, allowed, eq=0.82, po=0.0,
                          msg=f"3-bet {hc} — BR79: {int(3 if _is_ip(position) else 4)}x {'IP' if _is_ip(position) else 'OOP'}")

        # Flat call range: small pairs, connectors, offsuit broadways
        if hc in _BR79_FLAT_HANDS and "call" in available:
            # BR79: minimize OOP flat calls
            if not _is_ip(position) and hc in {"22", "33", "44", "55", "66", "ATo", "KJo", "QJo"}:
                return _build("fold", None, table, allowed, eq=0.0, po=1.0,
                              msg=f"folding {hc} OOP — BR79: avoid OOP flat calls with weak speculative hands")
            return _build("call", None, table, allowed,
                          eq=_PREFLOP_EQUITY.get(hc, 0.50),
                          po=call_chips / max(pot + call_chips, 1),
                          msg=f"flat calling {hc} — set mine / speculate")

        # Premium that we'd 3-bet but raiser is EP — flat call instead
        if hc in _BR79_3BET_HANDS and raiser_is_ep and "call" in available:
            return _build("call", None, table, allowed, eq=0.82,
                          po=call_chips / max(pot + call_chips, 1),
                          msg=f"flat {hc} vs EP raise — BR79: don't 3-bet EP openers")

        # Everything else in open range but not flat/3bet range — fold
        return _build("fold", None, table, allowed, eq=0.0, po=1.0,
                      msg=f"folding {hc} — not in BR79 flat/3bet range vs raise")

    # ── No aggression — open or check/fold ───────────────────────────────
    if "check" in available:
        return _build("check", None, table, allowed, eq=0.5, po=0.0,
                      msg="checking BB option — free play")

    if call_chips == 0:
        # First to act — open raise if in range
        if hc in _BR79_OPEN_RANGE and "raise" in available:
            size = _open_size(bb_chips, allowed)
            return _build("raise", size, table, allowed,
                          eq=_PREFLOP_EQUITY.get(hc, 0.55), po=0.0,
                          msg=f"open {hc} 3x BB from {position} — BR79 TAG")
        return _build("fold", None, table, allowed, eq=0.0, po=1.0,
                      msg=f"folding {hc} — not in BR79 top 20% range")

    # Facing a limp (call_chips ≈ bb_chips) — isolate with good hands, fold junk
    if hc in _BR79_OPEN_RANGE and "raise" in available:
        size = _open_size(bb_chips, allowed)
        return _build("raise", size, table, allowed,
                      eq=_PREFLOP_EQUITY.get(hc, 0.55), po=0.0,
                      msg=f"isolating limp with {hc} — BR79: always raise, never limp")
    if hc in _BR79_FLAT_HANDS and "call" in available and _is_ip(position):
        return _build("call", None, table, allowed,
                      eq=_PREFLOP_EQUITY.get(hc, 0.50),
                      po=call_chips / max(pot + call_chips, 1),
                      msg=f"completing with {hc} IP")
    return _build("fold", None, table, allowed, eq=0.0, po=1.0,
                  msg=f"folding {hc} preflop — not playable")


def _raiser_is_ep(table: dict) -> bool:
    """Try to detect if the raiser is in early position (UTG/EP).
    BR79: don't 3-bet vs EP raises, they usually have a strong hand."""
    seats = table.get("seats") or []
    self_num = table.get("selfSeatNumber")
    active = [s for s in seats if s.get("status") not in ("folded", "out", "bust")]
    active_nums = sorted(s.get("seatNumber", 0) for s in active)
    n = len(active_nums)
    if n < 3:
        return False

    # Look for the seat with a large currentBetChips that isn't SB/BB
    bb_chips = int(table.get("bigBlindChips") or 20)
    sb_chips = int(table.get("smallBlindChips") or 10)
    for s in active:
        if s.get("seatNumber") == self_num:
            continue
        cbc = int(s.get("currentBetChips") or 0)
        if cbc > bb_chips:  # they raised
            seat_num = s.get("seatNumber")
            if seat_num not in active_nums:
                continue
            # Find their position
            btn_num = table.get("buttonSeatNumber") or table.get("dealerSeatNumber")
            if btn_num and btn_num in active_nums:
                idx_btn = active_nums.index(btn_num)
                idx_raiser = active_nums.index(seat_num)
                rel = (idx_raiser - idx_btn) % n
                # EP = UTG area (rel 3 in 6-max means 3 seats after BTN = UTG)
                if rel >= 3:
                    return True
    return False


# ─── Postflop decision ────────────────────────────────────────────────────────

def _decide_postflop(
    hole: list[str], board: list[str], allowed: dict, available: set,
    table: dict, equity: float, pot_odds: float, texture: dict, draws: dict,
    pot: int, call_chips: int, position: str, street: str,
) -> dict:
    opp_type = _detect_opponent_type(table)
    ip = _is_ip(position)
    wet_board = texture.get("wet") or texture.get("type") in ("wet", "monotone")
    dry_board = texture.get("type") in ("dry", "paired")
    draw_eq = draws.get("draw_equity", 0.0)
    has_draw = draw_eq > 0.15

    # Good draw + made hand = combined equity
    effective_eq = max(equity, draw_eq)

    # ── Facing a bet ──────────────────────────────────────────────────────
    if call_chips > 0:
        # BR79: FAST-PLAY strong hands — raise with TPTK+ when facing a bet
        if equity >= 0.72 and "raise" in available:
            amount = _raise_size(call_chips, pot, allowed)
            return _build("raise", amount, table, allowed, eq=equity, po=pot_odds,
                          msg="fast-play strong hand — BR79: never slow play at micros")

        # Strong draw (flush+OESD) — semi-bluff raise in position
        if draw_eq >= 0.54 and ip and "raise" in available:
            amount = _raise_size(call_chips, pot, allowed)
            return _build("raise", amount, table, allowed, eq=draw_eq, po=pot_odds,
                          msg="semi-bluff raise: flush draw + OESD combo")

        # BR79: call with top pair weak kicker, middle pair, draws
        if effective_eq >= pot_odds + 0.05 and "call" in available:
            draw_note = ""
            if draws.get("flush_draw"):
                draw_note = " (flush draw)"
            elif draws.get("oesd"):
                draw_note = " (OESD)"
            return _build("call", None, table, allowed, eq=effective_eq, po=pot_odds,
                          msg=f"calling — eq {int(effective_eq*100)}% covers price{draw_note}")

        # BR79: with nothing, just fold — avoid FPS
        if "check" in available:
            return _build("check", None, table, allowed, eq=equity, po=pot_odds,
                          msg="checking — not enough to continue vs bet")
        return _build("fold", None, table, allowed, eq=equity, po=pot_odds,
                      msg="folding — BR79: if you have nothing, just fold")

    # ── No bet to face — check or bet ────────────────────────────────────
    if call_chips == 0:
        # ── FLOP C-BET LOGIC ──────────────────────────────────────────────
        if street == "Flop":
            # BR79: c-bet ~2/3 of time vs 1 opponent with any pair, any draw, Ax/Kx high IP
            should_cbet = False
            cbet_reason = ""

            if equity >= 0.55:  # have a real hand
                should_cbet = True
                cbet_reason = f"value cbet eq {int(equity*100)}%"
            elif has_draw:  # have a draw
                should_cbet = True
                cbet_reason = f"semi-bluff cbet with draw"
            elif ip and (equity >= 0.45):  # ace/king/queen high IP
                should_cbet = True
                cbet_reason = "thin cbet IP with overcards"
            # BR79: DON'T cbet OOP with nothing on wet board vs calling station
            elif not ip and wet_board and opp_type == "caller":
                should_cbet = False
                cbet_reason = "checking — OOP nothing on wet board vs caller"

            if should_cbet and "bet" in available:
                # Use bigger size on wet boards (2/3 pot), smaller on dry (1/2 pot)
                amount = _cbet_size(pot, allowed, big=wet_board)
                return _build("bet", amount, table, allowed, eq=effective_eq, po=0.0,
                              msg=f"c-bet — {cbet_reason}")

            if "check" in available:
                return _build("check", None, table, allowed, eq=equity, po=0.0,
                              msg="checking flop — BR79: no c-bet conditions met")

        # ── TURN DOUBLE-BARREL LOGIC ──────────────────────────────────────
        elif street == "Turn":
            # BR79: double barrel ~50% — only with made hand or good draw
            should_barrel = False
            barrel_reason = ""

            if equity >= 0.60:  # solid made hand
                should_barrel = True
                barrel_reason = f"double barrel with made hand eq {int(equity*100)}%"
            elif has_draw and draw_eq >= 0.30:  # strong draw
                should_barrel = True
                barrel_reason = "double barrel with strong draw"
            # BR79 scare card exception: dry flop + A or K on turn + vs folder/tight player
            elif dry_board and (texture.get("has_ace") or texture.get("has_king")):
                if opp_type == "folder" or (not ip and equity >= 0.40):
                    should_barrel = True
                    barrel_reason = "bluff barrel — scare card on dry board vs tight player"
            # BR79: DON'T barrel vs calling station with nothing
            elif opp_type == "caller":
                should_barrel = False

            if should_barrel and "bet" in available:
                amount = _cbet_size(pot, allowed, big=wet_board)
                return _build("bet", amount, table, allowed, eq=effective_eq, po=0.0,
                              msg=f"turn barrel — {barrel_reason}")

            if "check" in available:
                return _build("check", None, table, allowed, eq=equity, po=0.0,
                              msg="checking turn — BR79: no barrel conditions met")

        # ── RIVER LOGIC ───────────────────────────────────────────────────
        elif street == "River":
            # BR79: triple barrel only with TPTK+ (equity > 0.70)
            # Also value bet thin (top pair) when pot was checked on a prior street
            if equity >= 0.72 and "bet" in available:
                # Size: 2/3 pot for value
                br = allowed.get("betRange") or {}
                target = int(pot * 0.66)
                lo = int(br.get("min") or max(target // 2, 1))
                hi = int(br.get("max") or target * 2)
                amount = max(lo, min(target, hi))
                return _build("bet", amount, table, allowed, eq=equity, po=0.0,
                              msg="triple barrel / river value bet — BR79: TPTK+ bets river")

            # Value bet thin vs calling station (they call with worse)
            if equity >= 0.55 and opp_type == "caller" and "bet" in available:
                br = allowed.get("betRange") or {}
                target = int(pot * 0.50)  # smaller size for thin value
                lo = int(br.get("min") or max(target // 2, 1))
                hi = int(br.get("max") or target * 2)
                amount = max(lo, min(target, hi))
                return _build("bet", amount, table, allowed, eq=equity, po=0.0,
                              msg="thin value river — BR79: calling stations pay off")

            if "check" in available:
                return _build("check", None, table, allowed, eq=equity, po=0.0,
                              msg="checking river — no value bet / not TPTK+")

    # Fallback
    if "check" in available:
        return _build("check", None, table, allowed, eq=equity, po=pot_odds,
                      msg="free option")
    return _build("fold", None, table, allowed, eq=0.0, po=1.0,
                  msg="folding — no good action available")


def _jam_or_call(allowed: dict, available: set, table: dict,
                 eq: float, msg: str) -> dict:
    if "all-in" in available:
        hi = int((allowed.get("raiseRange") or {}).get("max") or
                 allowed.get("allInToAmount") or 9999)
        return _build("all-in", hi, table, allowed, eq=eq, po=0.0, msg=msg)
    if "raise" in available:
        hi = int((allowed.get("raiseRange") or {}).get("max") or 9999)
        return _build("raise", hi, table, allowed, eq=eq, po=0.0, msg=msg)
    if "call" in available:
        return _build("call", None, table, allowed, eq=eq, po=0.0, msg=msg)
    return _build("fold", None, table, allowed, eq=0.0, po=1.0, msg="no jam option")


# ─── Main decide() ────────────────────────────────────────────────────────────

def decide(table: dict, deadline_s: float = 10.0,
           research_context: Optional[dict] = None) -> dict:
    """BlackRain79 TAG decide() — tight preflop, aggressive postflop,
    value-heavy vs calling stations, fold with nothing."""
    allowed = table.get("allowedActions") or {}
    available = set(allowed.get("availableActions") or [])

    if deadline_s < 2.0:
        if "check" in available:
            return _build("check", None, table, allowed, eq=0.5, po=0.0,
                          msg="deadline tight, checking")
        return _build("fold", None, table, allowed, eq=0.0, po=1.0,
                      msg="deadline tight, folding")

    street = (table.get("street") or "Preflop")
    self_num = table.get("selfSeatNumber")
    seats = table.get("seats") or []
    self_seat = next((s for s in seats if s.get("seatNumber") == self_num), {})
    hole = list(self_seat.get("holeCards") or [])
    board = list(table.get("boardCards") or [])

    pot = max(int(table.get("potChips") or 0), 1)
    call_chips = int(allowed.get("callChips") or 0)
    bb_chips = max(int(table.get("bigBlindChips") or 20), 1)
    position = _detect_position(table)

    if street == "Preflop":
        return _decide_preflop(
            hole, allowed, available, table,
            call_chips, bb_chips, position, pot,
        )

    equity = estimate_equity(hole, board, sims=250, deadline_s=deadline_s)
    pot_odds = call_chips / max(pot + call_chips, 1) if call_chips else 0.0
    texture = _board_texture(board)
    draws = _has_draw(hole, board)

    return _decide_postflop(
        hole, board, allowed, available, table,
        equity, pot_odds, texture, draws,
        pot, call_chips, position, street,
    )


# ─── Build helpers ────────────────────────────────────────────────────────────

_FALLBACK_REASONING = '{vr: "std", ke: "legal", pp: "pot control"}'


def _build(action: str, amount: Optional[int], table: dict, allowed: dict,
           eq: float, po: float, msg: str) -> dict:
    reasoning = _build_reasoning(action, eq, po, table, allowed)
    payload: dict[str, Any] = {
        "action": action,
        "message": msg[:500],
        "reasoning": reasoning,
    }
    if amount is not None and action in ("bet", "raise", "all-in"):
        payload["amount"] = int(amount)
    return payload


def _build_reasoning(action: str, equity: float, pot_odds: float,
                     table: dict, allowed: dict) -> str:
    board = table.get("boardCards") or []
    street = (table.get("street") or "Preflop")
    pos_label = _detect_position(table)
    plan_map = {"Preflop": "see flop", "Flop": "barrel T",
                "Turn": "ck R", "River": "showdown"}
    pp = f"{pos_label} {plan_map.get(street, 'pot ctrl')}"[:30]

    if not board:
        bf = "[]"
    else:
        suits = [c[-1].lower() for c in board if c]
        feats: list[str] = []
        for s in set(suits):
            if suits.count(s) >= 2:
                feats.append(f"FD-{s}")
        ranks = [c[0].upper() for c in board if c]
        if len(set(ranks)) < len(ranks):
            feats.append("paired")
        bf = "[" + ",".join(feats[:3]) + "]" if feats else "[dry]"

    ke = f"{int(round(equity * 100))}% eq"[:30]
    sr = ""
    if action in ("bet", "raise", "all-in"):
        sr = f"po {int(round(pot_odds * 100))}% sized for FE"[:30]
    elif action == "call":
        sr = f"po {int(round(pot_odds * 100))}% covered"[:30]

    parts = [f'vr: "ln:unknown"', f'ke: "{ke}"', f'bf: {bf}', f'pp: "{pp}"']
    if sr:
        parts.append(f'sr: "{sr}"')
    yaml = "{" + ", ".join(parts) + "}"
    if len(yaml) <= 150:
        return yaml
    for drop_i in (4, 2):
        if drop_i < len(parts):
            trimmed = parts[:drop_i] + parts[drop_i + 1:]
            candidate = "{" + ", ".join(trimmed) + "}"
            if len(candidate) <= 150:
                return candidate
    return _FALLBACK_REASONING


# ─── Live loop ────────────────────────────────────────────────────────────────

def _safe_research_context(table: dict, retrieve_fn: Any) -> dict:
    if retrieve_fn is None:
        return {}
    try:
        ctx = retrieve_fn(table)
        return ctx if isinstance(ctx, dict) else {}
    except Exception as e:
        print(f"[arena-pokerkit] research hook failed: {e}", file=sys.stderr)
        return {}


def _validate_pending_tables(pending: Any) -> list[dict]:
    if not isinstance(pending, dict):
        return []
    raw = pending.get("tables")
    if not isinstance(raw, list):
        return []
    return [r for r in raw
            if isinstance(r, dict) and isinstance(r.get("tableId"), str) and r["tableId"]]


def _emit_heartbeat(phase: Any, completed: Any, target: Any, score: Any,
                    pending_count: int, label: str = "", eta_str: str = "") -> None:
    print(f"[arena-pokerkit{label}] phase={phase} | "
          f"completedHands={completed}/{target} | "
          f"adjustedBbPer100={score} | pending={pending_count}{eta_str}")


def _compute_eta(start_time: float, hands_done: Any, target: Any) -> str:
    try:
        hd, tgt = int(hands_done or 0), int(target or 0)
    except (TypeError, ValueError):
        return ""
    if hd <= 0 or tgt <= hd:
        return ""
    elapsed = time.monotonic() - start_time
    if elapsed <= 0:
        return ""
    eta_s = int((tgt - hd) * elapsed / hd)
    return f" | ETA {eta_s // 60}m{eta_s % 60:02d}s"


_ACTION_ALIASES = {"all_in": "all-in", "allin": "all-in"}


def _normalize_action_name(action: dict) -> dict:
    if not isinstance(action, dict):
        return action
    name = action.get("action")
    if isinstance(name, str) and name in _ACTION_ALIASES:
        out = dict(action)
        out["action"] = _ACTION_ALIASES[name]
        return out
    return action


def _attempt_credential_repair(client: ArenaClient, args: argparse.Namespace) -> bool:
    try:
        from arena_client import _move_creds_aside, _restore_creds_backup
        _move_creds_aside()
        client.api_key = None
        try:
            creds = load_or_register(client, args.handle, args.name, args.quote)
        except Exception:
            _restore_creds_backup()
            raise
        return bool(creds.get("apiKey") or client.api_key)
    except Exception as e:
        print(f"[arena-pokerkit] credential repair failed: {e}", file=sys.stderr)
        return False


def _run_benchmark_loop(
    client: ArenaClient, args: argparse.Namespace, competition_id: str,
    decide_fn: Any, retrieve_fn: Any, terminal_phases: set,
    terminal_statuses: set, label: str = "",
) -> int:
    state = load_state()
    rng = random.Random()
    last_completed_hands = 0
    saw_status_refresh = False
    last_status_at = 0.0
    last_heartbeat_at = 0.0
    credential_repair_used = False
    loop_start_monotonic = time.monotonic()

    _emit_heartbeat(phase="(starting)", completed=0, target="?", score=None,
                    pending_count=0, label=label)
    last_heartbeat_at = time.time()

    while True:
        tables: list[dict] = []
        try:
            pending = client.get(f"/texas/pending-actions?competitionId={competition_id}")
            tables = sorted(_validate_pending_tables(pending),
                            key=lambda t: (t.get("actionDeadlineAt") or 0))
        except ArenaError as e:
            print(f"[arena-pokerkit] pending-actions error: {e}", file=sys.stderr)
            if e.status in (401, 403):
                if not credential_repair_used and _attempt_credential_repair(client, args):
                    credential_repair_used = True
                    continue
                return 4
            if e.status == 404:
                raise

        if tables:
            table = tables[0]
            deadline_ms = table.get("actionDeadlineAt") or 0
            deadline_s = (max(0.0, (deadline_ms / 1000.0) - time.time())
                          if deadline_ms else 10.0)
            research_context = _safe_research_context(table, retrieve_fn)
            try:
                action = decide_fn(table, deadline_s=deadline_s,
                                   research_context=research_context)
            except TypeError:
                action = decide_fn(table, deadline_s=deadline_s)
            action = _normalize_action_name(action)
            payload = {"tableId": table["tableId"], **action}
            try:
                client.post("/texas/action", payload)
                state["hands_played"] = state.get("hands_played", 0) + 1
                state["last_action"] = {"action": action["action"],
                                        "amount": action.get("amount"),
                                        "at": int(time.time())}
                save_state(state)
            except ArenaError as e:
                if e.status == 409:
                    state["stale_count"] = state.get("stale_count", 0) + 1
                    save_state(state)
                    continue
                if e.status in (401, 403):
                    if not credential_repair_used and _attempt_credential_repair(client, args):
                        credential_repair_used = True
                        continue
                    return 4
                if e.status == 400:
                    state["rejection_count"] = state.get("rejection_count", 0) + 1
                    save_state(state)
                    try:
                        client.post("/texas/action", {
                            "tableId": table["tableId"], "action": "fold",
                            "message": "fallback after illegal action",
                            "reasoning": _FALLBACK_REASONING,
                        })
                    except ArenaError:
                        pass
                    continue
                raise

            if (args.max_hands and saw_status_refresh
                    and last_completed_hands >= args.max_hands):
                print(f"[arena-pokerkit] hit --max-hands={args.max_hands}, stopping")
                return 0

        now = time.time()
        if (not tables) or (now - last_status_at >= STATUS_REFRESH_S):
            status = None
            try:
                status = client.get(f"/texas/benchmark/status?competitionId={competition_id}")
            except ArenaError as e:
                print(f"[arena-pokerkit] status error: {e}", file=sys.stderr)
                if e.status in (401, 403):
                    if not credential_repair_used and _attempt_credential_repair(client, args):
                        credential_repair_used = True
                        continue
                    return 4
            last_status_at = now
            if isinstance(status, dict):
                match = status.get("match") or {}
                saw_status_refresh = True
                try:
                    last_completed_hands = int(match.get("completedHands") or 0)
                except (TypeError, ValueError):
                    last_completed_hands = 0

                if args.max_hands and last_completed_hands >= args.max_hands:
                    print(f"[arena-pokerkit{label}] hit --max-hands={args.max_hands}, stopping")
                    return 0

                if now - last_heartbeat_at >= 5.0:
                    _emit_heartbeat(phase=match.get("phase"),
                                    completed=match.get("completedHands"),
                                    target=match.get("targetHands"),
                                    score=match.get("adjustedBbPer100"),
                                    pending_count=len(tables), label=label,
                                    eta_str=_compute_eta(loop_start_monotonic,
                                                         match.get("completedHands"),
                                                         match.get("targetHands")))
                    last_heartbeat_at = now

                phase = match.get("phase")
                msstatus = match.get("status")
                if phase in terminal_phases or msstatus in terminal_statuses:
                    print(f"[arena-pokerkit{label}] match terminal ({phase}/{msstatus}) | "
                          f"hands={match.get('completedHands')} | "
                          f"adjustedBbPer100={match.get('adjustedBbPer100')}")
                    score = match.get("adjustedBbPer100")
                    if score is not None:
                        try:
                            s = float(score)
                            tag = ("🏆 crushing it" if s > 8 else
                                   "✓ positive — solid BR79 play" if s > 0 else
                                   "↺ slightly negative — iterate" if s > -10 else
                                   "⚠ bleed — check for leaks")
                            print(f"[arena-pokerkit{label}] verdict: {tag} (score: {s:+.1f} bb/100)")
                        except (TypeError, ValueError):
                            pass
                    state["bankroll"] = int(match.get("rawChipDelta") or 0)
                    save_state(state)
                    try:
                        decide_version = os.environ.get("ARENA_DECIDE_VERSION", "BR79-TAG")
                        append_iteration({
                            "bb_per_100": float(match.get("adjustedBbPer100") or 0),
                            "hands": int(match.get("completedHands") or 0),
                            "decide_version": decide_version,
                            "phase": match.get("phase"),
                            "status": match.get("status"),
                        })
                    except Exception:
                        pass
                    return 0

        if not tables:
            time.sleep(POLL_INTERVAL + rng.uniform(-POLL_JITTER, POLL_JITTER))


def run_live_benchmark(args: argparse.Namespace,
                       decide_fn: Optional[Any] = None) -> int:
    load_dotenv()
    api_key = os.environ.get("ARENA_API_KEY") or None
    base = os.environ.get("ARENA_API_BASE", DEFAULT_BASE)
    competition_id = args.competition_id or os.environ.get("ARENA_COMPETITION_ID")
    if not competition_id:
        print("ERROR: set ARENA_COMPETITION_ID in .env or pass --competition-id",
              file=sys.stderr)
        return 2

    decide_fn = decide_fn or decide
    client = ArenaClient(base, api_key=api_key)
    try:
        creds = load_or_register(client, args.handle, args.name, args.quote)
        agent_id = creds.get("agentId") or creds.get("id") or "?"
        print(f"[arena-pokerkit] registered agent={agent_id} base={base}")
        schema = fetch_introspection(client)
        assert_endpoints(schema)
        terminal_phases, terminal_statuses = resolve_terminal_phases(schema)
        print(f"[arena-pokerkit] introspection OK")
        try:
            start_resp = client.post("/texas/benchmark/start",
                                     {"competitionId": competition_id})
        except ArenaError as e:
            if e.status == 402:
                print("[arena-pokerkit] entry fee required", file=sys.stderr)
                return 3
            if e.status == 409:
                # Match already exists — resume it by fetching current status
                print("[arena-pokerkit] match already exists — resuming existing benchmark")
                try:
                    status_resp = client.get(
                        f"/texas/benchmark/status?competitionId={competition_id}")
                    start_resp = status_resp if isinstance(status_resp, dict) else {}
                except ArenaError:
                    start_resp = {}
            else:
                raise
        if not isinstance(start_resp, dict):
            raise ArenaError(0, str(start_resp)[:200], "benchmark/start malformed")
        match = start_resp.get("match") or {}
        if match.get("phase") in terminal_phases or match.get("status") in terminal_statuses:
            print(f"[arena-pokerkit] already terminal: {json.dumps(match, sort_keys=True)}")
            return 0
        print(f"[arena-pokerkit] benchmark started/resumed: phase={match.get('phase')} "
              f"completed={match.get('completedHands')} target={match.get('targetHands')}")
        return _run_benchmark_loop(
            client=client, args=args, competition_id=competition_id,
            decide_fn=decide_fn, retrieve_fn=retrieve_solver_context,
            terminal_phases=terminal_phases, terminal_statuses=terminal_statuses,
        )
    finally:
        client.close()


def load_external_decide(path: str) -> Any:
    import importlib.util
    from pathlib import Path as _Path
    p = _Path(path)
    if not p.exists():
        raise SystemExit(f"[arena-pokerkit] --agent path not found: {path}")
    spec = importlib.util.spec_from_file_location(f"_ext_{p.stem}", str(p))
    if spec is None or spec.loader is None:
        raise SystemExit(f"[arena-pokerkit] could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    fn = getattr(mod, "decide", None)
    if not callable(fn):
        raise SystemExit(f"[arena-pokerkit] {path} has no decide() function")
    return fn


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="BlackRain79 TAG Competition Agent")
    parser.add_argument("--competition-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-scenario", choices=("instant", "queued", "stale"),
                        default="instant")
    parser.add_argument("--max-hands", type=int, default=0)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--handle", default="adah-br79-agent")
    parser.add_argument("--name", default="Adah BR79 Agent")
    parser.add_argument("--quote", default="tight is right, aggression wins")
    args = parser.parse_args(argv)

    decide_fn = decide
    if args.agent:
        decide_fn = load_external_decide(args.agent)

    if args.dry_run:
        from mock import run_mock_benchmark
        return run_mock_benchmark(args, decide_fn=decide_fn,
                                  retrieve_solver_context=retrieve_solver_context)
    return run_live_benchmark(args, decide_fn=decide_fn)


if __name__ == "__main__":
    sys.exit(main())
