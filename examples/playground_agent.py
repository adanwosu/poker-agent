"""Arena PokerKit — Playground Agent (PvP Live Tables).

Competition: [Poker] Playground S1 (cmpy2qy65002ud9ej6b7jjq0l)
Mode: Texas Hold'em live tables vs other agents.

Flow:
  join → poll pending-actions → submit action → hand ends → rejoin
  Repeat until bankroll is exhausted.

Uses the same BlackRain79 TAG decide() from agent.py.
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
    assert_endpoints,
    fetch_introspection,
    load_or_register,
    load_state,
    resolve_terminal_phases,
    save_state,
)
from agent import decide, retrieve_solver_context, _FALLBACK_REASONING

PLAYGROUND_COMPETITION_ID = "cmpy2qy65002ud9ej6b7jjq0l"
POLL_INTERVAL = 2.0
POLL_JITTER = 0.5
_ACTION_ALIASES = {"all_in": "all-in", "allin": "all-in"}


def _normalize(action: dict) -> dict:
    if not isinstance(action, dict):
        return action
    name = action.get("action")
    if isinstance(name, str) and name in _ACTION_ALIASES:
        out = dict(action)
        out["action"] = _ACTION_ALIASES[name]
        return out
    return action


def _emit(msg: str) -> None:
    print(f"[playground] {msg}", flush=True)


def run_playground(args: argparse.Namespace) -> int:
    load_dotenv()
    api_key = os.environ.get("ARENA_API_KEY") or None
    base = os.environ.get("ARENA_API_BASE", DEFAULT_BASE)
    # Always use the playground competition — ignore ARENA_COMPETITION_ID (that points to Eval)
    competition_id = args.competition_id or PLAYGROUND_COMPETITION_ID

    client = ArenaClient(base, api_key=api_key)
    rng = random.Random()

    try:
        # Register / verify creds
        creds = load_or_register(client, args.handle, args.name, args.quote)
        agent_id = creds.get("agentId") or creds.get("id") or "?"
        _emit(f"registered agent={agent_id} base={base}")

        # Introspect
        schema = fetch_introspection(client)
        assert_endpoints(schema)
        _emit("introspection OK")

        state = load_state()
        hands_played = state.get("hands_played", 0)
        total_joins = 0

        while True:
            # Join the playground queue
            _emit(f"joining playground (competition={competition_id}) ...")
            join_resp = None
            try:
                join_resp = client.post("/texas/join", {"competitionId": competition_id})
            except ArenaError as e:
                if e.status == 409:
                    # Already in matchmaking queue — just start polling
                    _emit("already in matchmaking queue — polling for table seat...")
                    join_resp = {}
                elif e.status == 402:
                    _emit("entry fee required — playground is not free right now")
                    return 3
                elif e.status == 403:
                    _emit("agent must be X-verified and claimed to enter this competition")
                    return 4
                if e.status == 400:
                    body = str(e.body)
                    if "bankroll" in body.lower() or "insufficient" in body.lower():
                        _emit("bankroll exhausted — season over")
                        return 0
                    _emit(f"join rejected: {body[:200]}")
                    return 5
                raise

            total_joins += 1
            if isinstance(join_resp, dict):
                table = join_resp.get("table") or {}
                bankroll = join_resp.get("bankroll") or join_resp.get("bankrollChips")
                buy_in = table.get("buyInChips") or "?"
                blinds = f"{table.get('smallBlindChips','?')}/{table.get('bigBlindChips','?')}"
                stack = None
                seats = table.get("seats") or []
                self_num = table.get("selfSeatNumber")
                for s in seats:
                    if s.get("seatNumber") == self_num:
                        stack = s.get("stackChips")
                _emit(f"seated at table — buy-in={buy_in} blinds={blinds} stack={stack} bankroll={bankroll}")

            # Play loop for this table session
            last_status_log = 0.0
            while True:
                tables: list[dict] = []
                try:
                    pending = client.get(f"/texas/pending-actions?competitionId={competition_id}")
                    if isinstance(pending, dict):
                        raw = pending.get("tables") or []
                        tables = [t for t in raw
                                  if isinstance(t, dict) and isinstance(t.get("tableId"), str)]
                        tables = sorted(tables, key=lambda t: (t.get("actionDeadlineAt") or 0))
                except ArenaError as e:
                    _emit(f"pending-actions error: {e}")
                    time.sleep(2)
                    continue

                if tables:
                    table = tables[0]
                    deadline_ms = table.get("actionDeadlineAt") or 0
                    deadline_s = (max(0.0, (deadline_ms / 1000.0) - time.time())
                                  if deadline_ms else 10.0)

                    # Get research context
                    try:
                        rc = retrieve_solver_context(table)
                    except Exception:
                        rc = {}

                    # Decide
                    try:
                        action = decide(table, deadline_s=deadline_s, research_context=rc)
                    except Exception as ex:
                        _emit(f"decide() error: {ex} — folding")
                        action = {"action": "fold", "message": "error fallback",
                                  "reasoning": _FALLBACK_REASONING}

                    action = _normalize(action)
                    payload = {"tableId": table["tableId"], **action}

                    try:
                        resp = client.post("/texas/action", payload)
                        hands_played += 1
                        state["hands_played"] = hands_played

                        # Check if hand ended / busted from response
                        if isinstance(resp, dict):
                            resp_table = resp.get("table") or {}
                            seats = resp_table.get("seats") or []
                            self_num = resp_table.get("selfSeatNumber")
                            for s in seats:
                                if s.get("seatNumber") == self_num:
                                    stack = s.get("stackChips", 0)
                                    if stack == 0:
                                        _emit("stack = 0, will rejoin")
                            bankroll = resp.get("bankroll") or resp.get("bankrollChips")
                            if bankroll is not None:
                                state["bankroll"] = bankroll

                        save_state(state)
                        _emit(f"action={action['action']} hands={hands_played} "
                              f"bankroll={state.get('bankroll', '?')}")

                    except ArenaError as e:
                        if e.status == 409:
                            continue  # stale, re-poll
                        if e.status == 400:
                            # fallback fold
                            try:
                                client.post("/texas/action", {
                                    "tableId": table["tableId"],
                                    "action": "fold",
                                    "message": "illegal action fallback",
                                    "reasoning": _FALLBACK_REASONING,
                                })
                            except ArenaError:
                                pass
                            continue
                        _emit(f"action error: {e}")
                        if e.status in (401, 403):
                            return 4

                else:
                    # No pending tables — check if we need to rejoin
                    now = time.time()
                    if now - last_status_log > 10.0:
                        _emit(f"waiting for table... hands={hands_played} joins={total_joins}")
                        last_status_log = now

                    # Check if we're still seated by trying to see if there's
                    # anything pending. If nothing for 30s, try rejoining.
                    time.sleep(POLL_INTERVAL + rng.uniform(-POLL_JITTER, POLL_JITTER))

                    # After a gap with no tables, try rejoining
                    # (hand ended and we need a new seat)
                    if now - last_status_log > 25:
                        _emit("no activity for 30s — rejoining")
                        break  # break inner loop to rejoin

    except KeyboardInterrupt:
        _emit("stopped by user")
        return 0
    finally:
        client.close()

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Playground PvP agent — BlackRain79 TAG")
    parser.add_argument("--competition-id", default=None,
                        help=f"Competition ID (default: {PLAYGROUND_COMPETITION_ID})")
    parser.add_argument("--handle", default="adah_rain")
    parser.add_argument("--name", default="Adah Rain")
    parser.add_argument("--quote", default="tight is right, aggression wins")
    args = parser.parse_args(argv)
    return run_playground(args)


if __name__ == "__main__":
    sys.exit(main())
