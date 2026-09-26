"""
05_route_optimization.py
--------------------------
Stage 4 of the strategic roadmap: Prescriptive Modelling — Route Optimization.

Demonstrates the core idea behind last-mile route optimization using a
nearest-neighbour heuristic for the Travelling-Salesman-style routing
problem, and shows the resulting distance saving versus an unoptimized
(input-order) route.

Note: this is a teaching-scale illustration. In production, this stage
would use Google OR-Tools' Vehicle Routing Problem (VRP) solver, which
handles multiple vehicles, capacity constraints, and delivery time windows
far more robustly than the simple heuristic below. The OR-Tools call
pattern is sketched at the bottom of this file as pseudocode.

Run: python 05_route_optimization.py
"""

import numpy as np

RNG = np.random.default_rng(7)


def generate_stops(n_stops=15, area_km=25):
    """Random delivery stop coordinates (km grid) around a depot at (0, 0)."""
    depot = np.array([[0.0, 0.0]])
    stops = RNG.uniform(-area_km / 2, area_km / 2, size=(n_stops, 2))
    return np.vstack([depot, stops])  # index 0 is always the depot


def route_distance(points, order):
    total = 0.0
    for i in range(len(order) - 1):
        total += np.linalg.norm(points[order[i]] - points[order[i + 1]])
    return total


def nearest_neighbour_route(points):
    n = len(points)
    unvisited = set(range(1, n))
    route = [0]
    current = 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: np.linalg.norm(points[current] - points[j]))
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    route.append(0)  # return to depot
    return route


if __name__ == "__main__":
    points = generate_stops(n_stops=15)

    naive_route = list(range(len(points))) + [0]
    optimized_route = nearest_neighbour_route(points)

    naive_dist = route_distance(points, naive_route)
    optimized_dist = route_distance(points, optimized_route)
    improvement_pct = (naive_dist - optimized_dist) / naive_dist * 100

    print(f"Naive (input-order) route distance:      {naive_dist:.1f} km")
    print(f"Nearest-neighbour optimized distance:    {optimized_dist:.1f} km")
    print(f"Distance reduction:                      {improvement_pct:.1f}%")

    print(
        "\nProduction note: replace this heuristic with a proper VRP solver, "
        "e.g. Google OR-Tools, for multi-vehicle capacity- and time-window-"
        "constrained routing:\n"
    )
    print("""
    # --- OR-Tools VRP sketch (pseudocode) ---
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    routing.AddDimension(transit_callback_index, 0, vehicle_capacity, True, "Capacity")
    # add time-window constraints here for delivery-slot commitments ...

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    solution = routing.SolveWithParameters(search_parameters)
    """)
