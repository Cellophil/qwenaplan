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
        KCL: The sum of all power injections must equal the net position
        and the flow leaving/entering via lines and links.
        Equation: P_net + Sum(P_gen) - Sum(P_load) = Sum(P_line_flow)
        """
        # 1. Start with the nodal balance variable (the 'slack' or net position)
        balance_expr = bus.p_net

        # 2. Add all power elements connected to this bus (Generators, Loads, etc.)
        # We assume network.get_connected_power_elements(bus) returns a list of PowerElements
        for element in bus.network.get_connected_power_elements(bus):
            # Use get_p_net() if available (for storage units), otherwise use .p
            if hasattr(element, "get_p_net"):
                balance_expr += element.get_p_net()
            else:
                balance_expr += element.p

        # 3. Subtract flows on AC Lines and Links
        # For lines, we need to know if the bus is 'from' or 'to' to get the sign right
        for line in bus.network.get_connected_lines(bus):
            if line.from_bus == bus:
                balance_expr -= line.p  # Flow leaving
            else:
                balance_expr += line.p  # Flow entering

        # 4. Register the constraint in pyoframe: balance_expr == 0
        setattr(model, f"kcl_{bus.name}", balance_expr == 0)

    @staticmethod
    def apply_kirchhoff_voltage_law(line: "ACLine", model: Any):
        """
        KVL for DC approximation: P_flow * x_pu = theta_from - theta_to
        """
        # This links the line flow variable to the bus angle variables
        lhs = line.p * line.x_pu
        rhs = line.from_bus.theta - line.to_bus.theta

        setattr(model, f"kvl_{line.name}", lhs == rhs)
