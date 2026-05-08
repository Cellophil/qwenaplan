from typing import Any
import polars as pl


class DCPhysics:
    """
    Encapsulates the DC Power Flow formulation.
    Handles the translation of topology into pyoframe constraints.
    """

    @staticmethod
    def apply_kirchhoff_current_law(bus: "Bus", model: Any):
        """
        KCL: at every bus, injected power equals withdrawn + net outflow.

        Each connected component answers ``injection_at(bus)`` with its
        signed contribution (Generator: ``+var.p_t``; Load:
        ``-p_set`` as ``pf.Param``; storage: ``+(p_out - p_in)``;
        branches: ``+var.p_t`` at ``to_bus``, ``-var.p_t`` at
        ``from_bus``). KCL just sums them and constrains the sum to
        zero — the sign convention lives on the components, not here.

        There is no nodal slack variable: if generation cannot meet load
        with the available transmission, the LP is infeasible. To absorb
        shortfall, add an explicit high-marginal-cost generator at the bus.
        """
        balance_expr = None

        def _accum(expr):
            nonlocal balance_expr
            balance_expr = expr if balance_expr is None else balance_expr + expr

        for c in bus.network.get_connected_power_elements(bus):
            _accum(c.injection_at(bus))
        for branch in bus.network.get_connected_lines(bus):
            _accum(branch.injection_at(bus))

        # If a bus has nothing attached, KCL is trivially 0 == 0; skip the
        # constraint entirely so we don't dump a degenerate row on the solver.
        if balance_expr is None:
            return

        setattr(model, f"kcl_{bus.name}", balance_expr == 0)

    @staticmethod
    def apply_kirchhoff_voltage_law(line: "ACLine", model: Any):
        """
        KVL for DC approximation: P_flow * x_pu = theta_from - theta_to
        """
        lhs = line.var.p_t * line.x_pu
        rhs = line.from_bus.var.theta_t - line.to_bus.var.theta_t

        setattr(model, f"kvl_{line.name}", lhs == rhs)
