"""Tests for PumpedHydroStorage (storage + coupled generator).

Init/repr smoke + the PHS-specific story: the inner generator's electrical
output is locked to the inner storage's water dispatch by the coupling
constraint ``p_gen = p_dispatch * gen_efficiency``. Generic SOC dynamics
are covered by test_storage_unit.py.
"""
import polars as pl
import pyoptinterface as poi

import qwenaplan as qp


class TestPumpedHydroInit:
    def test_default_parameters(self):
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        phs = n.add(qp.PumpedHydroStorage, "PHS", bus=bus,
                    e_nom=1000.0, p_nom_turbine=100.0)
        assert phs.e_nom == 1000.0
        assert phs.p_nom_turbine == 100.0
        assert phs.p_nom_pump == 0.0
        assert phs.gen_efficiency == 0.9

    def test_custom_parameters(self):
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        phs = n.add(qp.PumpedHydroStorage, "PHS", bus=bus,
                    e_nom=2000.0, p_nom_turbine=150.0, p_nom_pump=100.0,
                    eff_store=0.85, eff_dispatch=0.9, gen_efficiency=0.95,
                    initial_soc=1000.0, soc_min=200.0, soc_max=1800.0,
                    influx=5.0)
        assert (phs.p_nom_pump, phs.gen_efficiency) == (100.0, 0.95)
        assert (phs.eff_store, phs.eff_dispatch) == (0.85, 0.9)
        assert (phs.initial_soc, phs.soc_min, phs.soc_max) == (1000.0, 200.0, 1800.0)
        assert phs.influx == 5.0

    def test_internal_components_named_predictably(self):
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        phs = n.add(qp.PumpedHydroStorage, "PHS", bus=bus,
                    e_nom=1000.0, p_nom_turbine=100.0)
        assert phs.storage.name == "PHS_storage"
        assert phs.generator.name == "PHS_generator"


class TestPumpedHydroDelegation:
    def test_p_nom_turbine_mutation_propagates_to_inner(self):
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        phs = n.add(qp.PumpedHydroStorage, "PHS", bus=bus,
                    e_nom=1000.0, p_nom_turbine=100.0)
        phs.p_nom_turbine = 80.0
        assert phs.storage.p_nom_out == 80.0
        assert phs.generator.p_nom == 80.0

    def test_soc_min_mutation_propagates(self):
        n = qp.Network()
        bus = n.add(qp.Bus, "Bus")
        phs = n.add(qp.PumpedHydroStorage, "PHS", bus=bus,
                    e_nom=1000.0, p_nom_turbine=100.0)
        phs.soc_min = 250.0
        assert phs.storage.soc_min == 250.0


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
