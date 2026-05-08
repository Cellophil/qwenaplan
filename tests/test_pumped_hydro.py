"""Tests for PHS-specific behaviour.

The shared composite shape (defaults, ``soc_min`` mutation, the inner
``storage`` / ``generator`` existence contract) lives in
``test_storage_composite.py`` parametrised over Battery + PHS. This
file holds only PHS-specific things: the asymmetric ``p_nom_turbine``
/ ``p_nom_pump`` knobs, ``gen_efficiency`` and ``influx`` semantics,
the predictable ``_storage`` / ``_generator`` inner-component names,
and the coupling constraint that ties electrical output to water
dispatch.
"""
import polars as pl
import pyoptinterface as poi

import qwenaplan as qp


class TestPumpedHydroInit:
    def test_phs_specific_defaults(self):
        """PHS-only knobs that don't exist on Battery: ``p_nom_turbine``,
        ``p_nom_pump``, and the ``gen_efficiency=0.9`` default."""
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        phs = n.add(qp.PumpedHydroStorage, "PHS", bus=bus,
                    e_nom=1000.0, p_nom_turbine=100.0)
        assert phs.p_nom_turbine == 100.0
        assert phs.p_nom_pump == 0.0
        assert phs.gen_efficiency == 0.9

    def test_phs_specific_custom_parameters(self):
        """PHS-only kwargs accept and round-trip: ``gen_efficiency`` and
        ``influx``. Generic kwargs (e_nom, eff_store/dispatch, soc bounds)
        are covered by test_storage_composite.py."""
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        phs = n.add(qp.PumpedHydroStorage, "PHS", bus=bus,
                    e_nom=2000.0, p_nom_turbine=150.0, p_nom_pump=100.0,
                    gen_efficiency=0.95, influx=5.0)
        assert phs.p_nom_pump == 100.0
        assert phs.gen_efficiency == 0.95
        assert phs.influx == 5.0

    def test_internal_components_named_predictably(self):
        """PHS prefixes its inner components with ``_storage`` /
        ``_generator``. Battery uses a different naming convention (its
        inner storage carries the battery's own name) — see test_battery."""
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        phs = n.add(qp.PumpedHydroStorage, "PHS", bus=bus,
                    e_nom=1000.0, p_nom_turbine=100.0)
        assert phs.storage.name == "PHS_storage"
        assert phs.generator.name == "PHS_generator"


class TestPumpedHydroDelegation:
    def test_p_nom_turbine_mutation_propagates_to_inner(self):
        """PHS-specific: mutating ``p_nom_turbine`` rewires both the inner
        storage's outflow rail and the inner generator's nameplate."""
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        phs = n.add(qp.PumpedHydroStorage, "PHS", bus=bus,
                    e_nom=1000.0, p_nom_turbine=100.0)
        phs.p_nom_turbine = 80.0
        assert phs.storage.p_nom_out == 80.0
        assert phs.generator.p_nom == 80.0


class TestPumpedHydroCoupling:
    """The coupling constraint: ``p_generator = p_dispatch * gen_efficiency``."""

    def test_generator_output_equals_dispatch_times_eff(self, phs_test_network, snapshots):
        n, bus, gen, phs, load = phs_test_network
        phs.eff_dispatch = 1.0
        phs.gen_efficiency = 0.9

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        p_gen = phs.sol.p_t["p"].to_list()
        p_disp = phs.sol.p_dispatch_t["p_dispatch"].to_list()
        for g, d in zip(p_gen, p_disp):
            assert abs(g - d * 0.9) < 1e-6

    def test_phs_prefers_to_serve_load_from_reservoir_when_cheap(self, snapshots):
        """If there's only a costly external generator, PHS should drain its
        initial reservoir to serve load. The objective should be lower than
        without PHS."""
        n_with = qp.Network()
        bus = n_with.add(qp.Bus, "Bus")
        n_with.add(qp.Generator, "Costly", bus=bus, p_nom=100.0,
                   marginal_cost=200.0)
        n_with.add(qp.Load, "L", bus=bus, p_set=20.0)
        n_with.add(qp.PumpedHydroStorage, "PHS", bus=bus,
                   e_nom=1000.0, p_nom_turbine=30.0, p_nom_pump=0.0,
                   eff_dispatch=1.0, gen_efficiency=1.0,
                   initial_soc=500.0)
        n_with.set_snapshots(snapshots)
        n_with.create_model()
        assert n_with.optimize() == poi.TerminationStatusCode.OPTIMAL
        cost_with = n_with.objective_value

        n_without = qp.Network()
        bus2 = n_without.add(qp.Bus, "Bus")
        n_without.add(qp.Generator, "Costly", bus=bus2, p_nom=100.0,
                      marginal_cost=200.0)
        n_without.add(qp.Load, "L", bus=bus2, p_set=20.0)
        n_without.set_snapshots(snapshots)
        n_without.create_model()
        assert n_without.optimize() == poi.TerminationStatusCode.OPTIMAL
        cost_without = n_without.objective_value

        # PHS displaces costly gen → lower total cost.
        assert cost_with < cost_without

    def test_soc_decreases_while_generating(self, snapshots):
        """If the only way to serve a load is to discharge PHS, the
        reservoir SOC must monotonically decrease."""
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        n.add(qp.Generator, "Costly", bus=bus, p_nom=100.0,
              marginal_cost=200.0)
        n.add(qp.Load, "L", bus=bus, p_set=20.0)
        phs = n.add(qp.PumpedHydroStorage, "PHS", bus=bus,
                    e_nom=1000.0, p_nom_turbine=30.0, p_nom_pump=0.0,
                    eff_dispatch=1.0, gen_efficiency=1.0,
                    initial_soc=500.0, influx=0.0)

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL

        soc = phs.sol.soc_t["soc"].to_list()
        # No pumping (p_nom_pump=0), no influx → SOC monotone non-increasing.
        for i in range(1, len(soc)):
            assert soc[i] <= soc[i - 1] + 1e-6
