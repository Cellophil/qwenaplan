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

        Sign convention (collected on the LHS, balance == 0):
          + Generator output       (var.p_t)
          - Load demand            (Load.get_p_net() returns -p_set)
          + Storage net injection  (p_out - p_in via get_p_net())
          - Outgoing line/link flow (sign depends on from_bus / to_bus)
          + Incoming line/link flow

        There is no nodal slack variable: if generation cannot meet load with
        the available transmission, the LP is infeasible. To absorb shortfall,
        add an explicit high-marginal-cost generator at the bus.
        """
        balance_expr = None

        def _accum(expr):
            nonlocal balance_expr
            balance_expr = expr if balance_expr is None else balance_expr + expr

        # 1. Power elements at this bus (generators, loads, storage units, ...)
        for element in bus.network.get_connected_power_elements(bus):
            # Storage / Load (and composites) implement get_p_net(); plain
            # Generator falls back to its primary variable on var.p_t.
            if hasattr(element, "get_p_net"):
                _accum(element.get_p_net())
            else:
                _accum(element.var.p_t)

        # 2. Line / link flows. from_bus = outflow (subtract), to_bus = inflow (add).
        for line in bus.network.get_connected_lines(bus):
            if line.from_bus == bus:
                _accum(-line.var.p_t)
            else:
                _accum(line.var.p_t)

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
