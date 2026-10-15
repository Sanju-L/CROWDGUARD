# ──────────────────────────────────────────────
# CrowdGuard -- Venue Graph / Route Engine
# ──────────────────────────────────────────────

"""
Loads a venue graph from JSON into NetworkX and computes
dynamic, risk-aware evacuation routes.

Dynamic routing formula:
    effective_cost(edge) = base_cost * risk_multiplier(zone_risk_level)

    risk_multiplier lookup:
        SAFE     = 1
        WARNING  = 3
        HIGH     = 8
        CRITICAL = 15
        BLOCKED  = infinity

    shortest_path = Dijkstra(source, exit, weight=effective_cost)

The engine evaluates ALL exits and recommends the one with
the lowest total cost.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import networkx as nx

from config import DEFAULT_MAP_FILE, MAP_DIR, ROUTE_COST_MULTIPLIERS


# ── Data Structures ──────────────────────────


@dataclass
class RouteResult:
    """Result of a routing query."""

    source: str
    recommended_exit: str
    route_nodes: List[str]
    total_cost: float
    all_exits_evaluated: Dict[str, float]   # {exit_id: cost}
    explanation: str


# ── Route Engine ─────────────────────────────


class RouteEngine:
    """
    Loads and manages the venue graph.  Computes dynamic
    risk-aware shortest paths using Dijkstra.

    Usage
    -----
    >>> engine = RouteEngine()
    >>> engine.load_graph()
    >>> engine.update_risk_state({"ZONE_A": "HIGH", "ZONE_B": "SAFE", ...})
    >>> result = engine.find_safest_route("ZONE_A")
    >>> print(result.recommended_exit, result.route_nodes)
    """

    def __init__(
        self,
        map_file: str = DEFAULT_MAP_FILE,
        cost_multipliers: Dict[str, float] | None = None,
    ) -> None:
        self.map_path = MAP_DIR / map_file
        self.graph: nx.Graph = nx.Graph()
        self._raw: dict = {}
        self.cost_multipliers = cost_multipliers or ROUTE_COST_MULTIPLIERS

        # Current risk level per node (default SAFE)
        self._risk_state: Dict[str, str] = {}

    # ── Load ─────────────────────────────────

    def load_graph(self, path: Path | str | None = None) -> None:
        """Load the venue graph from a JSON file."""
        p = Path(path) if path else self.map_path
        if not p.exists():
            raise FileNotFoundError(f"Venue map not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._raw = data
        self.graph = nx.Graph()

        for node in data.get("nodes", []):
            self.graph.add_node(
                node["id"],
                name=node.get("name", node["id"]),
                type=node.get("type", "zone"),
                x=node.get("x", 0),
                y=node.get("y", 0),
            )

        for edge in data.get("edges", []):
            self.graph.add_edge(
                edge["source"],
                edge["target"],
                distance=edge.get("distance", 1),
                base_cost=edge.get("base_cost", 1.0),
                status=edge.get("status", "open"),
            )

        # Default all nodes to SAFE
        for n in self.graph.nodes():
            self._risk_state[n] = "SAFE"

    @property
    def is_loaded(self) -> bool:
        return len(self.graph) > 0

    # ── Risk state ───────────────────────────

    def update_risk_state(self, risk_levels: Dict[str, str]) -> None:
        """
        Update the risk level for one or more nodes.

        Parameters
        ----------
        risk_levels : {node_id: risk_level}
            e.g. {"ZONE_A": "HIGH", "EXIT_A": "CRITICAL"}
        """
        for node_id, level in risk_levels.items():
            if node_id in self.graph:
                self._risk_state[node_id] = level

    def get_risk_state(self) -> Dict[str, str]:
        """Return current risk level of every node."""
        return dict(self._risk_state)

    # ── Dynamic edge cost ────────────────────

    def _effective_cost(self, u: str, v: str, edge_data: dict) -> float:
        """
        Compute the effective traversal cost of an edge.

        Formula:
            effective_cost = base_cost * max(multiplier(u), multiplier(v))

        Uses the WORSE risk level of the two endpoints so
        routes avoid entering dangerous zones.
        """
        base = edge_data.get("base_cost", 1.0)
        status = edge_data.get("status", "open")

        # Closed edges are impassable
        if status == "closed":
            return float("inf")

        mult_u = self.cost_multipliers.get(self._risk_state.get(u, "SAFE"), 1)
        mult_v = self.cost_multipliers.get(self._risk_state.get(v, "SAFE"), 1)
        multiplier = max(mult_u, mult_v)

        return base * multiplier

    # ── Routing ──────────────────────────────

    def find_safest_route(self, source: str) -> RouteResult:
        """
        Find the lowest-cost route from *source* to the best exit.

        Evaluates ALL exit nodes via Dijkstra and picks the
        one with the smallest total effective cost.

        Parameters
        ----------
        source : str
            Starting node ID (e.g. "ZONE_A").

        Returns
        -------
        RouteResult
        """
        if not self.is_loaded:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")

        exits = self.get_exits()
        if not exits:
            raise ValueError("No exit nodes defined in the graph.")

        # Evaluate all exits
        exit_costs: Dict[str, float] = {}
        exit_paths: Dict[str, List[str]] = {}

        for ex in exits:
            try:
                path = nx.dijkstra_path(
                    self.graph, source, ex, weight=self._effective_cost,
                )
                cost = nx.dijkstra_path_length(
                    self.graph, source, ex, weight=self._effective_cost,
                )
                exit_costs[ex] = round(cost, 2)
                exit_paths[ex] = path
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                exit_costs[ex] = float("inf")
                exit_paths[ex] = []

        # Pick the cheapest reachable exit
        best_exit = min(exit_costs, key=exit_costs.get)  # type: ignore[arg-type]
        best_cost = exit_costs[best_exit]
        best_path = exit_paths[best_exit]

        explanation = self._build_explanation(
            source, best_exit, best_path, best_cost, exit_costs,
        )

        return RouteResult(
            source=source,
            recommended_exit=best_exit,
            route_nodes=best_path,
            total_cost=best_cost,
            all_exits_evaluated=exit_costs,
            explanation=explanation,
        )

    def _build_explanation(
        self,
        source: str,
        best_exit: str,
        path: List[str],
        cost: float,
        all_costs: Dict[str, float],
    ) -> str:
        lines: List[str] = []

        # Exit evaluation
        lines.append("Exit evaluation:")
        for ex, c in sorted(all_costs.items(), key=lambda x: x[1]):
            risk = self._risk_state.get(ex, "SAFE")
            if math.isinf(c):
                lines.append(f"  {ex} ({risk}): UNREACHABLE")
            else:
                lines.append(f"  {ex} ({risk}): cost {c:.1f}")

        lines.append("")

        if math.isinf(cost):
            lines.append(f"No reachable exit from {source}.")
            return "\n".join(lines)

        # Route
        route_str = " -> ".join(path)
        lines.append(f"Recommended: {route_str}")
        lines.append(f"Total cost: {cost:.1f}")

        # Explain why non-best exits were avoided
        for ex, c in all_costs.items():
            if ex == best_exit:
                continue
            risk = self._risk_state.get(ex, "SAFE")
            if math.isinf(c):
                lines.append(f"  {ex} avoided: unreachable / blocked")
            elif c > cost:
                lines.append(
                    f"  {ex} avoided: cost {c:.1f} vs {cost:.1f} "
                    f"(risk level: {risk})"
                )

        return "\n".join(lines)

    # ── Queries ──────────────────────────────

    def get_nodes_by_type(self, node_type: str) -> List[str]:
        return [
            n for n, d in self.graph.nodes(data=True)
            if d.get("type") == node_type
        ]

    def get_node_positions(self) -> Dict[str, Tuple[float, float]]:
        return {
            n: (d["x"], d["y"])
            for n, d in self.graph.nodes(data=True)
        }

    def get_exits(self) -> List[str]:
        return self.get_nodes_by_type("exit")

    def get_entrances(self) -> List[str]:
        return self.get_nodes_by_type("entrance")

    def get_edge_status(self, u: str, v: str) -> str:
        data = self.graph.get_edge_data(u, v)
        if data is None:
            return "unknown"
        return data.get("status", "unknown")

    def summary(self) -> Dict[str, Any]:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "entrances": self.get_entrances(),
            "exits": self.get_exits(),
            "zones": self.get_nodes_by_type("zone"),
        }
