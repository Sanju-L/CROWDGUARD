# ──────────────────────────────────────────────
# CrowdGuard -- Venue Graph Unit Tests
# ──────────────────────────────────────────────

"""
Verifies the venue graph loads correctly and contains
the expected nodes, edges, and attributes.

Usage:
    cd CrowdGuard
    .venv\\Scripts\\python.exe -m tests.test_venue_graph
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing.route_engine import RouteEngine

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


# ── Test: load graph ─────────────────────────

def test_load() -> None:
    divider("Load venue graph")

    engine = RouteEngine()
    check(not engine.is_loaded, "Not loaded initially")

    engine.load_graph()
    check(engine.is_loaded, "Loaded after load_graph()")
    check(engine.graph.number_of_nodes() == 7, f"7 nodes (got {engine.graph.number_of_nodes()})")
    check(engine.graph.number_of_edges() == 10, f"10 edges (got {engine.graph.number_of_edges()})")


# ── Test: node IDs ───────────────────────────

def test_node_ids() -> None:
    divider("Node IDs")

    engine = RouteEngine()
    engine.load_graph()

    expected = {"ENTRANCE", "ZONE_A", "ZONE_B", "ZONE_C", "ZONE_D", "EXIT_A", "EXIT_B"}
    actual = set(engine.graph.nodes())
    check(actual == expected, f"All 7 node IDs present")

    for nid in expected:
        check(nid in engine.graph, f"  {nid} exists")


# ── Test: node attributes ────────────────────

def test_node_attributes() -> None:
    divider("Node attributes")

    engine = RouteEngine()
    engine.load_graph()

    for nid, data in engine.graph.nodes(data=True):
        check("name" in data, f"{nid}: has 'name'")
        check("type" in data, f"{nid}: has 'type'")
        check("x" in data,    f"{nid}: has 'x'")
        check("y" in data,    f"{nid}: has 'y'")
        check(data["type"] in ("entrance", "zone", "exit"), f"{nid}: valid type '{data['type']}'")


# ── Test: edge attributes ────────────────────

def test_edge_attributes() -> None:
    divider("Edge attributes")

    engine = RouteEngine()
    engine.load_graph()

    for u, v, data in engine.graph.edges(data=True):
        check("distance" in data,  f"{u}-{v}: has 'distance'")
        check("base_cost" in data, f"{u}-{v}: has 'base_cost'")
        check("status" in data,    f"{u}-{v}: has 'status'")
        check(data["distance"] > 0, f"{u}-{v}: positive distance")
        check(data["base_cost"] > 0, f"{u}-{v}: positive base_cost")
        check(data["status"] == "open", f"{u}-{v}: status=open")


# ── Test: node type queries ──────────────────

def test_node_types() -> None:
    divider("Node type queries")

    engine = RouteEngine()
    engine.load_graph()

    entrances = engine.get_entrances()
    check(entrances == ["ENTRANCE"], f"1 entrance (got {entrances})")

    exits = engine.get_exits()
    check(set(exits) == {"EXIT_A", "EXIT_B"}, f"2 exits (got {exits})")

    zones = engine.get_nodes_by_type("zone")
    check(set(zones) == {"ZONE_A", "ZONE_B", "ZONE_C", "ZONE_D"}, f"4 zones (got {zones})")


# ── Test: positions ──────────────────────────

def test_positions() -> None:
    divider("Node positions")

    engine = RouteEngine()
    engine.load_graph()

    pos = engine.get_node_positions()
    check(len(pos) == 7, f"7 positions (got {len(pos)})")

    # Verify ENTRANCE is at top (lowest y)
    entrance_y = pos["ENTRANCE"][1]
    for nid, (x, y) in pos.items():
        if nid != "ENTRANCE":
            check(y >= entrance_y, f"{nid} y={y} >= ENTRANCE y={entrance_y}")


# ── Test: connectivity ───────────────────────

def test_connectivity() -> None:
    divider("Graph connectivity")

    engine = RouteEngine()
    engine.load_graph()

    import networkx as nx

    check(nx.is_connected(engine.graph), "Graph is connected")

    # Every node should be reachable from ENTRANCE
    for nid in engine.graph.nodes():
        check(nx.has_path(engine.graph, "ENTRANCE", nid), f"Path ENTRANCE -> {nid}")

    # Every zone should reach at least one exit
    for zone in engine.get_nodes_by_type("zone"):
        exits = engine.get_exits()
        reachable = any(nx.has_path(engine.graph, zone, ex) for ex in exits)
        check(reachable, f"{zone} can reach an exit")


# ── Test: specific edges ────────────────────

def test_specific_edges() -> None:
    divider("Expected edges exist")

    engine = RouteEngine()
    engine.load_graph()

    expected_edges = [
        ("ENTRANCE", "ZONE_A"),
        ("ENTRANCE", "ZONE_B"),
        ("ZONE_A", "ZONE_B"),
        ("ZONE_A", "ZONE_C"),
        ("ZONE_B", "ZONE_D"),
        ("ZONE_C", "ZONE_D"),
        ("ZONE_A", "ZONE_D"),
        ("ZONE_B", "ZONE_C"),
        ("ZONE_C", "EXIT_A"),
        ("ZONE_D", "EXIT_B"),
    ]

    for u, v in expected_edges:
        check(engine.graph.has_edge(u, v), f"Edge {u} -- {v}")


# ── Test: summary ────────────────────────────

def test_summary() -> None:
    divider("Summary")

    engine = RouteEngine()
    engine.load_graph()

    s = engine.summary()
    check(s["nodes"] == 7, f"nodes=7 (got {s['nodes']})")
    check(s["edges"] == 10, f"edges=10 (got {s['edges']})")
    check(len(s["entrances"]) == 1, "1 entrance")
    check(len(s["exits"]) == 2, "2 exits")
    check(len(s["zones"]) == 4, "4 zones")

    print(f"    Summary: {s}")


# ── Test: edge status query ──────────────────

def test_edge_status() -> None:
    divider("Edge status query")

    engine = RouteEngine()
    engine.load_graph()

    check(engine.get_edge_status("ENTRANCE", "ZONE_A") == "open", "ENTRANCE-ZONE_A: open")
    check(engine.get_edge_status("ZONE_C", "EXIT_A") == "open", "ZONE_C-EXIT_A: open")
    check(engine.get_edge_status("EXIT_A", "EXIT_B") == "unknown", "Non-existent edge: unknown")


# ── Run all ──────────────────────────────────

def main() -> None:
    test_load()
    test_node_ids()
    test_node_attributes()
    test_edge_attributes()
    test_node_types()
    test_positions()
    test_connectivity()
    test_specific_edges()
    test_summary()
    test_edge_status()

    divider("RESULTS")
    print(f"  Passed : {PASS}")
    print(f"  Failed : {FAIL}")
    if FAIL == 0:
        print(f"  [OK] ALL {PASS} VENUE GRAPH TESTS PASSED")
    else:
        print(f"  [!!] {FAIL} TEST(S) FAILED")
        sys.exit(1)
    print()


if __name__ == "__main__":
    main()
