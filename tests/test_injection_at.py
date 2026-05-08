"""Tests for ``Component.injection_at`` / ``Component.injection_sign_at``.

These are the small abstraction the views layer is built on (and that KCL
([physics.py:apply_kirchhoff_current_law]) now calls instead of reaching
into ``from_bus`` / ``to_bus`` itself). The behaviour-correctness of KCL
is already covered by the rest of the suite; here we just verify the new
methods return what they say on the tin.

Branches sign their flow by which end ``bus`` is. Power elements ignore
the bus argument. Composites (Battery, PHS) mirror the power-element
default. The "wrong bus" path raises so user code that passes a stale
or unrelated bus into a view fails loudly instead of silently summing
zeros.
"""
import pytest

import qwenaplan as qp


class TestInjectionSignAt:
    """``injection_sign_at(bus)`` is the cheap numeric primitive used by
    the sol-side of the views layer (where ``-pyoframe_expr`` is unhelpful
    — we want a sign to apply on a polars column). Branches return ±1
    by orientation; everything else returns +1."""

    def test_branch_to_bus_is_positive(self, two_bus_network):
        _, bus1, bus2, _, line = two_bus_network
        # Line was added as from_bus=bus1, to_bus=bus2.
        assert line.injection_sign_at(bus2) == +1

    def test_branch_from_bus_is_negative(self, two_bus_network):
        _, bus1, _, _, line = two_bus_network
        assert line.injection_sign_at(bus1) == -1

    def test_branch_unrelated_bus_raises(self, two_bus_network):
        n, _, _, _, line = two_bus_network
        # Make a third bus that the line does not touch.
        bus3 = n.add(qp.Bus, "Bus3")
        with pytest.raises(ValueError, match="does not touch bus"):
            line.injection_sign_at(bus3)

    def test_generator_sign_ignores_bus_arg(self, two_bus_network):
        _, bus1, bus2, gen, _ = two_bus_network
        # Power elements only touch one bus; the bus argument is by
        # contract ignored on the default impl. Both calls return +1.
        assert gen.injection_sign_at(bus1) == +1
        assert gen.injection_sign_at(bus2) == +1


class TestInjectionAt:
    """``injection_at(bus)`` returns the pyoframe expression used by the
    var-side of views and by KCL. We don't assert numerical equivalence
    with ``var.p_t`` directly (pyoframe expressions don't compare by
    structural equality), but we do assert that:

    - it returns *something* (not None) for every component shape, and
    - the branch case returns a *different* object at from_bus vs to_bus
      (the sign is baked in symbolically — one is ``+var.p_t``, one is
      ``-var.p_t`` — so they cannot be the same Python object).
    """

    def test_generator_returns_var_p_t(self, snapshots, two_bus_network):
        n, _, _, gen, _ = two_bus_network
        n.add(qp.Load, "L", bus=n.buses["Bus2"], p_set=10.0)
        n.set_snapshots(snapshots)
        # Generator has no get_p_net → falls back to var.p_t directly.
        assert gen.injection_at(gen.bus) is gen.var.p_t

    def test_branch_signs_differ_between_ends(self, snapshots, two_bus_network):
        n, bus1, bus2, _, line = two_bus_network
        n.add(qp.Load, "L", bus=bus2, p_set=10.0)
        n.set_snapshots(snapshots)
        # Pyoframe needs the variable on a model before arithmetic, so
        # build the model first. ``-var.p_t`` then materialises as an
        # expression distinct from ``var.p_t``.
        n.create_model()
        at_to = line.injection_at(bus2)
        at_from = line.injection_at(bus1)
        # Different objects — one of them is a -1 * var.p_t expression.
        assert at_to is not at_from
