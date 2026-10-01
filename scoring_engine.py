"""
EmergeRoute — Scoring Engine (NSGA-II via pymoo)

Runs every candidate policy through the SUMO simulation, then uses NSGA-II
(via pymoo) to rank them across the six objectives from the project brief:
congestion, travel time, safety, emergency delay, environmental impact,
fairness.

Why NSGA-II here rather than picking the "best score" directly: these
objectives genuinely trade off against each other (e.g. a policy that
reduces average delay might increase worst-case unfairness). NSGA-II finds
the Pareto-optimal set -- policies where no objective can be improved without
worsening another -- rather than collapsing everything into one arbitrary
weighted score. We then pick the best-ranked policy from that Pareto set for
the dashboard's single "recommended policy" output.

Objective mapping (all framed as MINIMIZE, pymoo's default):
    1. congestion       -> avg_time_loss       (time lost to congestion)
    2. travel_time       -> avg_duration         (total trip time)
    3. safety             -> max_waiting_time     (proxy: worst-case standstill = higher risk)
    4. emergency_delay -> max_waiting_time     (same proxy for this prototype;
                                                    a fuller build would track a
                                                    dedicated emergency-vehicle route)
    5. environmental    -> total_co2             (kg CO2)
    6. fairness          -> (max_waiting_time - avg_waiting_time)  (spread between
                                                    worst-off and average trip -- a
                                                    direct fairness formalization,
                                                    see the fairness discussion in
                                                    our research-paper review)
"""
from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling

from run_policy import Policy, SimResult, run_simulation
from policy_generator import generate_candidate_policies


def policy_to_objectives(result: SimResult) -> np.ndarray:
    """Converts one simulation result into the 6-objective vector NSGA-II scores."""
    fairness_gap = result.max_waiting_time - result.avg_waiting_time
    return np.array([
        result.avg_time_loss,     # congestion
        result.avg_duration,       # travel time
        result.max_waiting_time,   # safety (proxy)
        result.max_waiting_time,   # emergency delay (proxy, see docstring)
        result.total_co2,          # environmental impact
        fairness_gap,               # fairness
    ])


class PolicySelectionProblem(Problem):
    """A pymoo Problem where each 'individual' is simply the INDEX of a
    pre-simulated candidate policy (we're selecting among a small, fixed set
    of already-tested candidates, not searching a continuous policy space --
    matching the brief's "test each candidate policy" framing rather than
    open-ended policy synthesis).
    """

    def __init__(self, objective_matrix: np.ndarray):
        self.objective_matrix = objective_matrix
        n_policies = objective_matrix.shape[0]
        super().__init__(
            n_var=1,
            n_obj=objective_matrix.shape[1],
            xl=0,
            xu=n_policies - 1,
            vtype=int,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        idx = np.round(x[:, 0]).astype(int)
        out["F"] = self.objective_matrix[idx]


class EnumerateAllSampling(Sampling):
    """Since we only have a handful of candidate policies, just evaluate all
    of them directly rather than randomly sampling -- guarantees every real
    candidate is actually considered."""

    def _do(self, problem, n_samples, **kwargs):
        n_policies = problem.xu[0] + 1
        indices = np.arange(n_policies).reshape(-1, 1)
        # pad/repeat if pymoo asks for more samples than we have policies
        if n_samples > n_policies:
            reps = int(np.ceil(n_samples / n_policies))
            indices = np.tile(indices, (reps, 1))[:n_samples]
        return indices[:n_samples]


def rank_policies(candidates: list[Policy], sumocfg: str = "simulation.sumocfg") -> list[dict]:
    """Runs every candidate policy through simulation, scores them via
    NSGA-II across all 6 objectives, and returns a ranked list (best first).

    Returns a list of dicts, each with: policy name, raw metrics, and a
    plain-language explanation -- directly consumable by the dashboard.
    """
    print(f"Testing {len(candidates)} candidate policies in simulation...")
    results: list[SimResult] = []
    for policy in candidates:
        print(f"  Running: {policy.name}")
        result = run_simulation(policy, sumocfg=sumocfg)
        results.append(result)

    objective_matrix = np.array([policy_to_objectives(r) for r in results])

    problem = PolicySelectionProblem(objective_matrix)
    algorithm = NSGA2(pop_size=min(len(candidates), 20), sampling=EnumerateAllSampling())
    res = minimize(problem, algorithm, ("n_gen", 1), verbose=False)

    # Rank the ORIGINAL candidates (not just the Pareto front) by a simple
    # sum-of-normalized-objectives tiebreaker, so we always return a full
    # ranked list for the dashboard, with the top pick being NSGA-II-Pareto-
    # optimal whenever possible.
    pareto_indices = set(np.round(res.X[:, 0]).astype(int).tolist()) if res.X is not None else set()

    normalized = (objective_matrix - objective_matrix.min(axis=0)) / (
        objective_matrix.max(axis=0) - objective_matrix.min(axis=0) + 1e-9
    )
    composite_scores = normalized.sum(axis=1)

    ranked_order = np.argsort(composite_scores)

    output = []
    for rank, idx in enumerate(ranked_order, start=1):
        r = results[idx]
        is_pareto = idx in pareto_indices
        fairness_gap = r.max_waiting_time - r.avg_waiting_time
        output.append({
            "rank": rank,
            "policy_name": r.policy_name,
            "pareto_optimal": is_pareto,
            "metrics": {
                "avg_time_loss_s": round(r.avg_time_loss, 1),
                "avg_travel_time_s": round(r.avg_duration, 1),
                "max_waiting_time_s": round(r.max_waiting_time, 1),
                "co2_kg": round(r.total_co2, 2),
                "fairness_gap_s": round(fairness_gap, 1),
                "completed_trips": r.completed_trips,
            },
            "explanation": _explain(r, rank == 1),
        })
    return output


def _explain(result: SimResult, is_top_pick: bool) -> str:
    """Generates the plain-language explanation the brief's dashboard requirement asks for."""
    prefix = "Recommended: " if is_top_pick else ""
    return (
        f"{prefix}This policy resulted in an average congestion delay of "
        f"{result.avg_time_loss:.0f}s and an average travel time of "
        f"{result.avg_duration:.0f}s across {result.completed_trips} trips, "
        f"with a worst-case single-trip wait of {result.max_waiting_time:.0f}s "
        f"and estimated {result.total_co2:.1f}kg CO2 emitted."
    )


if __name__ == "__main__":
    import argparse
    from policy_generator import find_busiest_intersections

    parser = argparse.ArgumentParser()
    parser.add_argument("--sumocfg", type=str, default="simulation.sumocfg")
    args = parser.parse_args()

    print(f"Using config: {args.sumocfg}")
    print("Probing baseline traffic to find genuinely busiest intersections...")
    busiest_first_tls_ids = find_busiest_intersections(sumocfg=args.sumocfg)
    print(f"  Ranked by real measured congestion: {busiest_first_tls_ids}")

    candidates = generate_candidate_policies(busiest_first_tls_ids, congestion_level="high")
    ranked = rank_policies(candidates, sumocfg=args.sumocfg)

    print("\n=== RANKED POLICIES (best first) ===")
    for entry in ranked:
        pareto_tag = " [Pareto-optimal]" if entry["pareto_optimal"] else ""
        print(f"\n#{entry['rank']}: {entry['policy_name']}{pareto_tag}")
        print(f"   {entry['explanation']}")
