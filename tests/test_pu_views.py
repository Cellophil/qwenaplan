"""Tests for the ``_pu_t`` per-unit views (plan_01).

The plan introduces per-unit (capacity-fraction) views on every component
that has a meaningful nameplate — `gen.p_pu_t`, `storage.soc_pu_t`,
`storage.p_pu_t`, and the analogous composites. They live on **both**
containers:

- ``component.var.<name>_pu_t`` is a pyoframe expression usable in custom
  constraints (no new variables introduced).
- ``component.sol.<name>_pu_t`` is the solved DataFrame with a ``*_pu``
  value column.

These are pure views — divisions by a static nameplate. The tests below
cover three things: numerical correctness on the solution side, that the
``var`` side is a usable pyoframe expression in a custom constraint, and
that undefined views (storage with neither p_nom_in nor p_nom_out) raise
a clear error.
"""
import polars as pl
import pyoptinterface as poi
import pytest

import qwenaplan as qp


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class TestGeneratorPU:
    def test_p_pu_t_is_p_t_over_p_nom(self, snapshots):
        """Solving a 30 MW dispatch on a 100 MW unit must yield p_pu = 0.3."""
        n = qp.Network()
        bus = n.add(qp.Bus, "B")
        gen = n.add(qp.Generator, "G", bus=bus, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Load, "L", bus=bus, p_set=30.0)
        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        p_pu = gen.sol.p_pu_t["p_pu"].to_list()
        assert all(abs(v - 0.3) < 1e-9 for v in p_pu)

    def test_p_pu_t_zero_p_nom_raises(self, network):
        """Dividing by p_nom=0 is undefined; the view must say so loudly."""
        bus = network.add(qp.Bus, "B")
        gen = network.add(qp.Generator, "G", bus=bus, p_nom=0.0)
        # Static lookup on var-side raises immediately (no need to solve).
        with pytest.raises(ZeroDivisionError, match="p_nom=0"):
            _ = gen.var.p_pu_t


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class TestStoragePU:
    def test_soc_pu_t_solved_value_matches_soc_over_e_nom(self, snapshots):
        """A storage at SOC=50 in a 100 MWh unit reports soc_pu = 0.5."""
        n = qp.Network()
        bus = n.add(qp.Bus, "B")
        n.add(qp.Generator, "G", bus=bus, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Load, "L", bus=bus, p_set=10.0)
        s = n.add(qp.StorageUnit, "S", bus=bus, e_nom=100.0,
                  p_nom_in=0.0, p_nom_out=0.0,
                  initial_soc=50.0, soc_min=50.0, soc_max=50.0)
        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        soc_pu = s.sol.soc_pu_t["soc_pu"].to_list()
        assert all(abs(v - 0.5) < 1e-9 for v in soc_pu)

    def test_var_soc_pu_t_usable_as_constraint_expression(self, snapshots):
        """The ``var.soc_pu_t`` expression must compose into a constraint
        and the solver must respect it. We cap soc_pu <= 0.4 on a 100 MWh
        battery starting at 50 MWh; the LP must drain below 40 MWh."""
        n = qp.Network()
        bus = n.add(qp.Bus, "B")
        n.add(qp.Generator, "G", bus=bus, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Load, "L", bus=bus, p_set=10.0)
        s = n.add(qp.StorageUnit, "S", bus=bus, e_nom=100.0,
                  p_nom_in=20.0, p_nom_out=20.0,
                  eff_in=1.0, eff_out=1.0, initial_soc=50.0)
        n.set_snapshots(snapshots)
        n.create_model()
        # Custom constraint via the var-side view. No new variables needed.
        n.model.soc_pu_cap = s.var.soc_pu_t <= 0.4
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        soc = s.sol.soc_t["soc"].to_list()
        for v in soc:
            assert v <= 40.0 + 1e-6

    def test_p_pu_t_undefined_when_no_nameplate(self, network, snapshots):
        """A storage with both p_nom_in and p_nom_out unset has no nameplate
        to normalise net power against. Accessing p_pu_t must raise."""
        bus = network.add(qp.Bus, "B")
        s = network.add(qp.StorageUnit, "S", bus=bus, e_nom=100.0,
                        p_nom_in=None, p_nom_out=None,
                        initial_soc=50.0)
        with pytest.raises(ValueError, match="neither p_nom_in nor p_nom_out"):
            _ = s.var.p_pu_t


# ---------------------------------------------------------------------------
# Battery composite
# ---------------------------------------------------------------------------

class TestBatteryPU:
    def test_battery_p_pu_t_uses_p_nom(self, snapshots):
        """Battery composite normalises p (net) against its single ``p_nom``,
        not the inner storage's max(p_nom_in, p_nom_out) — they're equal for
        Battery, but the composite should drive directly off ``b.p_nom``."""
        n = qp.Network()
        bus = n.add(qp.Bus, "B")
        cheap = pl.Series("time", [1.0, 1.0, 0.0, 0.0])
        n.add(qp.Generator, "Cheap", bus=bus, p_nom=100.0,
              marginal_cost=10.0, p_max_pu=cheap)
        n.add(qp.Generator, "Exp", bus=bus, p_nom=100.0, marginal_cost=100.0)
        n.add(qp.Load, "L", bus=bus, p_set=30.0)
        b = n.add(qp.Battery, "B1", bus=bus,
                  e_nom=100.0, p_nom=20.0,
                  eff_store=1.0, eff_dispatch=1.0, initial_soc=0.0)
        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        # p_pu must match p_t / p_nom on every snapshot.
        p = b.sol.p_t["p"].to_list()
        p_pu = b.sol.p_pu_t["p_pu"].to_list()
        for pi, ppi in zip(p, p_pu):
            assert abs(ppi - pi / 20.0) < 1e-9
