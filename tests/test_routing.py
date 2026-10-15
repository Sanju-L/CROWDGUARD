# ──────────────────────────────────────────────
# CrowdGuard -- Route Engine Unit Tests
# ──────────────────────────────────────────────

"""
Tests for dynamic risk-aware routing.

Scenarios tested:
    1. Normal route (all SAFE)
    2. Congested exit (prefers alternate)
    3. Blocked zone (routes around)
    4. Alternate route when risk changes
    5. All exits blocked
    6. Explanation content

Usage:
    cd CrowdGuard
    .venv\\Scripts\\python.exe -m tests.test_routing
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing.route_engine import RouteEngine, RouteResult

# ── Helpers ──────────────────────────────────

PASS = 0
FAIL = 0


def check(condition: bool, label: str) -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}")


def divider(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def make_engine() -> RouteEngine:
    """Create and load a fresh engine."""
    engine = RouteEngine()
    engine.load_graph()
    return engine


# ── Test 1: Normal route (all SAFE) ──────────

def test_normal_route() -> None:
    divider("1. Normal route (all zones SAFE)")

    engine = make_engine()
    # All zones default to SAFE

    # From ZONE_A: closest exit via base_cost
    #   ZONE_A -> ZONE_C -> EXIT_A: 1.0 + 0.8 = 1.8
    #   ZONE_A -> ZONE_B -> ZONE_D -> EXIT_B: 1.5 + 1.0 + 0.8 = 3.3
    #   (or ZONE_A -> ZONE_D -> EXIT_B: 2.0 + 0.8 = 2.8)
    result = engine.find_safest_route("ZONE_A")

    check(result.recommended_exit == "EXIT_A", f"best exit = EXIT_A (got {result.recommended_exit})")
    check("ZONE_A" in result.route_nodes, "route includes ZONE_A")
    check("EXIT_A" in result.route_nodes, "route includes EXIT_A")
    check(result.route_nodes[0] == "ZONE_A", "starts at ZONE_A")
    check(result.route_nodes[-1] == "EXIT_A", "ends at EXIT_A")
    check(result.total_cost < result.all_exits_evaluated.get("EXIT_B", 999),
          f"EXIT_A cheaper than EXIT_B")
    print(f"    Route: {' -> '.join(result.route_nodes)}, cost={result.total_cost}")


# ── Test 2: Congested exit ───────────────────

def test_congested_exit() -> None:
    divider("2. Congested exit (EXIT_A area is CRITICAL)")

    engine = make_engine()

    # Make ZONE_C and EXIT_A critical -> route should avoid EXIT_A
    engine.update_risk_state({
        "ZONE_C": "CRITICAL",
        "EXIT_A": "CRITICAL",
    })

    result = engine.find_safest_route("ZONE_A")

    # EXIT_A path goes through CRITICAL ZONE_C (cost * 15)
    # EXIT_B via ZONE_A -> ZONE_D -> EXIT_B (all SAFE) should be cheaper
    check(result.recommended_exit == "EXIT_B", f"avoids EXIT_A, picks EXIT_B (got {result.recommended_exit})")
    check("ZONE_C" not in result.route_nodes, "avoids ZONE_C")
    check("EXIT_B" in result.route_nodes, "reaches EXIT_B")
    print(f"    Route: {' -> '.join(result.route_nodes)}, cost={result.total_cost}")
    print(f"    EXIT_A cost: {result.all_exits_evaluated.get('EXIT_A')}")
    print(f"    EXIT_B cost: {result.all_exits_evaluated.get('EXIT_B')}")


# ── Test 3: Blocked zone ────────────────────

def test_blocked_zone() -> None:
    divider("3. Blocked zone (ZONE_D blocked)")

    engine = make_engine()

    engine.update_risk_state({"ZONE_D": "BLOCKED"})

    # From ZONE_B: ZONE_D is blocked so can't go ZONE_B -> ZONE_D -> EXIT_B directly
    # Should route through ZONE_A or ZONE_C instead
    result = engine.find_safest_route("ZONE_B")

    check("ZONE_D" not in result.route_nodes[1:-1],
          f"route avoids ZONE_D (route: {result.route_nodes})")
    check(result.total_cost < float("inf"), "route is reachable")
    print(f"    Route: {' -> '.join(result.route_nodes)}, cost={result.total_cost}")


# ── Test 4: Route changes with risk ─────────

def test_alternate_route() -> None:
    divider("4. Route changes when risk state changes")

    engine = make_engine()

    # Step 1: Normal conditions from ZONE_A
    result1 = engine.find_safest_route("ZONE_A")
    exit1 = result1.recommended_exit
    route1 = result1.route_nodes.copy()

    # Step 2: Make the preferred exit's path dangerous
    if exit1 == "EXIT_A":
        engine.update_risk_state({"ZONE_C": "CRITICAL", "EXIT_A": "CRITICAL"})
    else:
        engine.update_risk_state({"ZONE_D": "CRITICAL", "EXIT_B": "CRITICAL"})

    result2 = engine.find_safest_route("ZONE_A")

    check(result2.recommended_exit != exit1,
          f"exit changed: {exit1} -> {result2.recommended_exit}")
    check(result2.route_nodes != route1,
          f"route changed")
    print(f"    Before: {' -> '.join(route1)} -> {exit1}")
    print(f"    After:  {' -> '.join(result2.route_nodes)} -> {result2.recommended_exit}")


# ── Test 5: All exits blocked ───────────────

def test_all_exits_blocked() -> None:
    divider("5. All exits blocked")

    engine = make_engine()
    engine.update_risk_state({
        "EXIT_A": "BLOCKED",
        "EXIT_B": "BLOCKED",
        "ZONE_C": "BLOCKED",
        "ZONE_D": "BLOCKED",
    })

    result = engine.find_safest_route("ZONE_A")

    check(math.isinf(result.total_cost), f"cost is infinity (got {result.total_cost})")
    check("UNREACHABLE" in result.explanation or "unreachable" in result.explanation.lower(),
          "explanation mentions unreachable")
    print(f"    Cost: {result.total_cost}")


# ── Test 6: WARNING zones increase cost ──────

def test_warning_increases_cost() -> None:
    divider("6. WARNING zones increase cost (but route still works)")

    engine = make_engine()

    # Normal cost from ZONE_A to EXIT_A
    result_safe = engine.find_safest_route("ZONE_A")
    cost_safe = result_safe.all_exits_evaluated.get("EXIT_A", 0)

    # Set ZONE_C to WARNING
    engine.update_risk_state({"ZONE_C": "WARNING"})
    result_warn = engine.find_safest_route("ZONE_A")
    cost_warn = result_warn.all_exits_evaluated.get("EXIT_A", 0)

    check(cost_warn > cost_safe,
          f"WARNING increases cost: {cost_safe} -> {cost_warn}")
    print(f"    SAFE cost to EXIT_A: {cost_safe}")
    print(f"    WARNING cost to EXIT_A: {cost_warn}")


# ── Test 7: Explanation content ──────────────

def test_explanation() -> None:
    divider("7. Explanation content")

    engine = make_engine()
    engine.update_risk_state({"ZONE_C": "HIGH"})

    result = engine.find_safest_route("ZONE_A")

    check("Exit evaluation" in result.explanation, "has exit evaluation section")
    check("Recommended" in result.explanation, "has recommended route")
    check("Total cost" in result.explanation, "has total cost")
    check("->" in result.explanation, "has route arrows")
    print(f"\n{result.explanation}")


# ── Test 8: From ENTRANCE ────────────────────

def test_from_entrance() -> None:
    divider("8. Route from ENTRANCE")

    engine = make_engine()
    result = engine.find_safest_route("ENTRANCE")

    check(result.route_nodes[0] == "ENTRANCE", "starts at ENTRANCE")
    check(result.recommended_exit in ("EXIT_A", "EXIT_B"), f"reaches an exit (got {result.recommended_exit})")
    check(result.total_cost < float("inf"), "reachable")
    print(f"    Route: {' -> '.join(result.route_nodes)}, cost={result.total_cost}")


# ── Test 9: Risk state update / get ──────────

def test_risk_state() -> None:
    divider("9. Risk state management")

    engine = make_engine()

    state = engine.get_risk_state()
    check(all(v == "SAFE" for v in state.values()), "all default to SAFE")

    engine.update_risk_state({"ZONE_A": "HIGH", "ZONE_B": "WARNING"})
    state = engine.get_risk_state()
    check(state["ZONE_A"] == "HIGH", "ZONE_A updated to HIGH")
    check(state["ZONE_B"] == "WARNING", "ZONE_B updated to WARNING")
    check(state["ZONE_C"] == "SAFE", "ZONE_C still SAFE")


# ── Run all ──────────────────────────────────

def main() -> None:
    test_normal_route()
    test_congested_exit()
    test_blocked_zone()
    test_alternate_route()
    test_all_exits_blocked()
    test_warning_increases_cost()
    test_explanation()
    test_from_entrance()
    test_risk_state()

    divider("RESULTS")
    print(f"  Passed : {PASS}")
    print(f"  Failed : {FAIL}")
    if FAIL == 0:
        print(f"  [OK] ALL {PASS} ROUTING TESTS PASSED")
    else:
        print(f"  [!!] {FAIL} TEST(S) FAILED")
        sys.exit(1)
    print()


if __name__ == "__main__":
    main()
