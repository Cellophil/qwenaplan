"""Tests for behaviour shared across all StorageComposite subclasses.

After plan_03, ``Battery`` and ``PumpedHydroStorage`` are thin
``__init__`` wrappers around a single concrete ``StorageComposite``. The
shared shape — both expose ``storage`` / ``generator``, both delegate
``soc_min`` / ``e_nom`` to the inner storage, both build a coupling
row at ``create_model`` time — used to be double-tested in
``test_battery.py`` and ``test_pumped_hydro.py``. Those duplicates now
live here, parametrised over the two factory shapes.

Tests that are genuinely class-specific (Battery's ``p_nom`` symmetry,
PHS's ``gen_efficiency`` coupling numerics, dispatch behaviour) stay in
their respective files.
"""
import pytest

import qwenaplan as qp


# ``factory`` returns ``(network, bus, composite)``. Two factories cover
# the entire concrete-subclass surface.
@pytest.fixture(params=["battery", "phs"])
def composite(request):
    n = qp.Network()
    bus = n.add(qp.Bus, "Bus")
    if request.param == "battery":
        c = n.add(qp.Battery, "X", bus=bus, e_nom=100.0, p_nom=30.0)
    else:
        c = n.add(qp.PumpedHydroStorage, "X", bus=bus,
                  e_nom=100.0, p_nom_turbine=30.0)
    return n, bus, c


class TestStorageCompositeShape:
    """Public surface that every composite must expose."""

    def test_inner_storage_and_generator_present(self, composite):
        _, _, c = composite
        # Both inner components are constructed unconditionally — Battery
        # used to have only a storage, but plan_03 unified the shape.
        assert c.storage is not None
        assert c.generator is not None

    def test_e_nom_delegated_to_inner_storage(self, composite):
        _, _, c = composite
        assert c.e_nom == 100.0
        assert c.e_nom == c.storage.e_nom
        c.e_nom = 80.0
        assert c.storage.e_nom == 80.0

    def test_soc_bounds_default_to_zero_and_e_nom(self, composite):
        _, _, c = composite
        assert c.soc_min == 0.0
        assert c.soc_max == 100.0
        assert c.initial_soc == 0.0

    def test_soc_min_mutation_propagates(self, composite):
        """Both Battery and PHS delegate ``soc_min`` onto the inner storage."""
        _, _, c = composite
        c.soc_min = 25.0
        assert c.storage.soc_min == 25.0

    def test_efficiency_defaults(self, composite):
        """Storage-side efficiencies default to 1.0 on both shapes.

        ``gen_efficiency`` differs between the two (Battery default 1.0,
        PHS default 0.9) and lives in the per-class tests.
        """
        _, _, c = composite
        assert c.eff_store == 1.0
        assert c.eff_dispatch == 1.0
