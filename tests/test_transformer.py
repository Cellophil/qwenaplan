"""Tests for the Transformer component (and the symmetric ACLine v_nom rule).

Transformer is a thin :class:`ACLine` subclass: same KVL row, same flow
variable, same thermal limit; the only difference is the nominal-voltage
rule (must DIFFER between the two buses, where ACLine requires equality).

The PyPSA importer converts ``x`` from PyPSA's "pu on the transformer's
own ``s_nom`` base" to qp's "pu on a 1 MVA system base" — the conversion
test exercises that single line of arithmetic against a duck-typed dict
mock so we don't pull pypsa into the unit-test loop.
"""
import polars as pl
import pyoptinterface as poi
import pytest

import qwenaplan as qp


class TestTransformerInit:
    def test_default_parameters(self, network):
        b1 = network.add(qp.Bus, "B1", v_nom=380.0)
        b2 = network.add(qp.Bus, "B2", v_nom=110.0)
        tx = network.add(qp.Transformer, "T1", from_bus=b1, to_bus=b2)
        assert tx.x_pu == 0.1
        assert tx.s_nom == 0.0
        assert tx.from_bus is b1
        assert tx.to_bus is b2

    def test_custom_parameters(self, network):
        b1 = network.add(qp.Bus, "B1", v_nom=380.0)
        b2 = network.add(qp.Bus, "B2", v_nom=110.0)
        tx = network.add(
            qp.Transformer, "T1", from_bus=b1, to_bus=b2, x_pu=0.001, s_nom=200.0
        )
        assert tx.x_pu == 0.001
        assert tx.s_nom == 200.0

    def test_repr(self, network):
        b1 = network.add(qp.Bus, "B1", v_nom=380.0)
        b2 = network.add(qp.Bus, "B2", v_nom=110.0)
        tx = network.add(
            qp.Transformer, "T1", from_bus=b1, to_bus=b2, x_pu=0.001, s_nom=200.0
        )
        assert repr(tx) == "<Transformer(name=T1, B1->B2, x_pu=0.001, s_nom=200.0)>"

    def test_is_acline_subclass(self):
        assert issubclass(qp.Transformer, qp.ACLine)


class TestTransformerValidation:
    """``from_bus.v_nom`` MUST differ from ``to_bus.v_nom``."""

    def test_rejects_equal_v_nom(self, network):
        b1 = network.add(qp.Bus, "B1", v_nom=380.0)
        b2 = network.add(qp.Bus, "B2", v_nom=380.0)
        with pytest.raises(ValueError, match="equal nominal voltages"):
            network.add(qp.Transformer, "T1", from_bus=b1, to_bus=b2)

    def test_rejects_v_nom_within_tolerance(self, network):
        # Within the relative tolerance of 1e-6 → still flagged as equal.
        b1 = network.add(qp.Bus, "B1", v_nom=380.0)
        b2 = network.add(qp.Bus, "B2", v_nom=380.0 + 1e-9)
        with pytest.raises(ValueError, match="equal nominal voltages"):
            network.add(qp.Transformer, "T1", from_bus=b1, to_bus=b2)

    def test_accepts_unequal_v_nom(self, network):
        b1 = network.add(qp.Bus, "B1", v_nom=380.0)
        b2 = network.add(qp.Bus, "B2", v_nom=110.0)
        # Should not raise.
        network.add(qp.Transformer, "T1", from_bus=b1, to_bus=b2)


class TestACLineValidation:
    """ACLine's symmetric rule: v_nom MUST be equal."""

    def test_rejects_unequal_v_nom(self, network):
        b1 = network.add(qp.Bus, "B1", v_nom=380.0)
        b2 = network.add(qp.Bus, "B2", v_nom=110.0)
        with pytest.raises(ValueError, match="different nominal voltages"):
            network.add(qp.ACLine, "L1", from_bus=b1, to_bus=b2)

    def test_accepts_equal_v_nom(self, network):
        b1 = network.add(qp.Bus, "B1", v_nom=380.0)
        b2 = network.add(qp.Bus, "B2", v_nom=380.0)
        network.add(qp.ACLine, "L1", from_bus=b1, to_bus=b2)


class TestTransformerPhysics:
    """KVL behaviour for transformers is identical to ACLine."""

    def test_carries_full_load_between_voltage_levels(self, snapshots):
        n = qp.Network()
        b1 = n.add(qp.Bus, "B1", v_nom=380.0)
        b2 = n.add(qp.Bus, "B2", v_nom=110.0)
        n.add(qp.Generator, "Gen1", bus=b1, p_nom=100.0, marginal_cost=10.0)
        tx = n.add(
            qp.Transformer, "T1", from_bus=b1, to_bus=b2, x_pu=0.001, s_nom=200.0
        )
        n.add(qp.Load, "L1", bus=b2, p_set=40.0)

        n.set_snapshots(snapshots)
        n.create_model()
        assert n.optimize() == poi.TerminationStatusCode.OPTIMAL
        assert tx.sol.p_t["p"].to_list() == [40.0, 40.0, 40.0, 40.0]

    def test_numerical_equivalence_with_acline(self, snapshots):
        """Identical ``x_pu`` in two networks (one ACLine equal-v_nom,
        one Transformer different-v_nom) must produce identical flows.
        Transformer adds *no* physics on top of ACLine — only the v_nom
        rule changes."""
        # ACLine network: equal v_nom (500/500)
        n_l = qp.Network()
        bL1 = n_l.add(qp.Bus, "BL1", v_nom=500.0)
        bL2 = n_l.add(qp.Bus, "BL2", v_nom=500.0)
        n_l.add(qp.Generator, "Gen", bus=bL1, p_nom=100.0, marginal_cost=10.0)
        line = n_l.add(qp.ACLine, "Br", from_bus=bL1, to_bus=bL2, x_pu=0.05, s_nom=80.0)
        n_l.add(qp.Load, "Ld", bus=bL2, p_set=33.0)
        n_l.set_snapshots(snapshots)
        n_l.create_model()
        n_l.optimize()

        # Transformer network: different v_nom (500/220), same x_pu
        n_t = qp.Network()
        bT1 = n_t.add(qp.Bus, "BT1", v_nom=500.0)
        bT2 = n_t.add(qp.Bus, "BT2", v_nom=220.0)
        n_t.add(qp.Generator, "Gen", bus=bT1, p_nom=100.0, marginal_cost=10.0)
        tx = n_t.add(qp.Transformer, "Br", from_bus=bT1, to_bus=bT2, x_pu=0.05, s_nom=80.0)
        n_t.add(qp.Load, "Ld", bus=bT2, p_set=33.0)
        n_t.set_snapshots(snapshots)
        n_t.create_model()
        n_t.optimize()

        assert line.sol.p_t["p"].to_list() == tx.sol.p_t["p"].to_list()


class TestTransformerImporter:
    """PyPSA's ``x`` is on the transformer's own MVA base; qp's ``x_pu``
    is on the 1 MVA system base. Conversion: ``x_qp = x_pypsa / s_nom``."""

    def _duck_pypsa(self, transformers):
        """Build a stand-in object the importer can call ``getattr`` on.

        ``_get_pypsa_components`` probes ``Bus``, ``bus``, ``Transformer``,
        ``transformer`` (etc.) — singular and capitalized variants. We
        expose attributes under each of those names so the duck-typed
        source resolves the same way a real PyPSA Network would
        (``n.buses``, ``n.transformers``, …) without needing pandas.
        """
        class _Source:
            def __init__(self, transformers):
                buses = {"B1": {"v_nom": 380.0}, "B2": {"v_nom": 110.0}}
                # Singular + capitalized aliases for what the importer
                # probes. Both empty-set components and the populated
                # transformer table are exposed under several names so
                # ``getattr(self.source, name)`` finds something for each.
                self.Bus = self.bus = buses
                self.Transformer = self.transformer = transformers
                empty: dict = {}
                self.Generator = self.generator = empty
                self.Load = self.load = empty
                self.Line = self.line = empty
                self.Link = self.link = empty
                self.StorageUnit = self.storage_unit = empty
                self.Store = self.store = empty
                self.ShuntImpedance = empty
                self.GlobalConstraint = empty
                self.snapshots = list(range(2))
        return _Source(transformers)

    def test_pypsa_x_to_x_pu_conversion(self):
        src = self._duck_pypsa({
            "T1": {"bus0": "B1", "bus1": "B2", "x": 0.10, "s_nom": 100.0},
        })
        importer = qp.PyPSAImporter(src, strict_mode=False)
        target = importer.import_network()
        # 0.10 pu / 100 MVA = 0.001 pu on the system 1 MVA base
        assert target.transformers["T1"].x_pu == pytest.approx(0.001)
        assert target.transformers["T1"].s_nom == 100.0

    def test_pypsa_zero_s_nom_raises(self):
        src = self._duck_pypsa({
            "T1": {"bus0": "B1", "bus1": "B2", "x": 0.10, "s_nom": 0.0},
        })
        importer = qp.PyPSAImporter(src, strict_mode=False)
        importer.import_network()
        # Even in non-strict mode, the error is recorded.
        assert any("s_nom=0" in e for e in importer.errors)


class TestTransformerViews:
    def test_transformers_view_present(self, snapshots):
        n = qp.Network()
        b1 = n.add(qp.Bus, "B1", v_nom=380.0)
        b2 = n.add(qp.Bus, "B2", v_nom=110.0)
        n.add(qp.Generator, "G1", bus=b1, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Transformer, "T1", from_bus=b1, to_bus=b2, x_pu=0.001, s_nom=200.0)
        n.add(qp.Load, "L1", bus=b2, p_set=25.0)
        n.set_snapshots(snapshots)
        n.create_model()
        n.optimize()

        wide = n.views["transformers"].sol.p_t
        # Snapshot col + one column per transformer.
        assert "T1" in wide.columns
        assert wide["T1"].to_list() == [25.0, 25.0, 25.0, 25.0]

    def test_transformer_in_bus_view(self, snapshots):
        """The downstream bus's auto-built view sees the transformer
        with the right sign (+1 at to_bus)."""
        n = qp.Network()
        b1 = n.add(qp.Bus, "B1", v_nom=380.0)
        b2 = n.add(qp.Bus, "B2", v_nom=110.0)
        n.add(qp.Generator, "G1", bus=b1, p_nom=100.0, marginal_cost=10.0)
        n.add(qp.Transformer, "T1", from_bus=b1, to_bus=b2, x_pu=0.001, s_nom=200.0)
        n.add(qp.Load, "L1", bus=b2, p_set=25.0)
        n.set_snapshots(snapshots)
        n.create_model()
        n.optimize()
        # Bus2's view should have 25 from the transformer and -25 from the load
        # → KCL row sums to zero.
        bus2 = n.views["B2"].sol.p_t
        assert "T1" in bus2.columns and "L1" in bus2.columns
        assert bus2["T1"].to_list() == [25.0] * 4
        assert bus2["L1"].to_list() == [-25.0] * 4
