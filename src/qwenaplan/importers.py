"""PyPSA to qwenaplan Network Importer.

This module provides the PyPSAImporter class that converts networks from the
original PyPSA (linopy/pandas-based) to qwenaplan (pyoframe/polars-based).

Supported components: Bus, Generator, Load, ACLine, Link, StorageUnit
Unsupported components: Transformer, Store, ShuntImpedance, Carrier (standalone),
    GlobalConstraint
"""

import logging
from typing import Any, Dict, List, Optional

import polars as pl

logger = logging.getLogger(__name__)


class PyPSAImporter:
    """Import a PyPSA Network and convert to qwenaplan Network.

    Parameters
    ----------
    pypsa_network : any
        The PyPSA network to import. Can be a path (str/Path) to a .nc/.csv file
        or an already-loaded PyPSA Network object.
    strict_mode : bool, default True
        If True, raise errors on unsupported features.
        If False, skip unsupported features with warnings.

    Attributes
    ----------
    warnings : list[str]
        List of warning messages during import.
    errors : list[str]
        List of error messages during import.
    unsupported_features : dict[str, list[str]]
        Summary of unsupported features found in the source network.
    """

    # Components that are permanently unsupported
    UNSUPPORTED_COMPONENTS = [
        "Transformer",
        "Transformers",
        "Store",
        "Stores",
        "ShuntImpedance",
        "ShuntImpedances",
        "Carrier",
        "Carriers",
        "GlobalConstraint",
        "GlobalConstraints",
    ]

    # PyPSA attribute names that indicate investment/expansion features
    INVESTMENT_ATTRS = [
        "p_nom_extendable",
        "s_nom_extendable",
        "e_nom_extendable",
        "p_nom_min",
        "p_nom_max",
        "s_nom_min",
        "s_nom_max",
        "e_nom_min",
        "e_nom_max",
        "capital_cost",
        "build_year",
        "lifetime",
        "investment_periods",
    ]

    # PyPSA attribute names that indicate unit commitment features
    UNIT_COMMITMENT_ATTRS = [
        "committable",
        "min_up_time",
        "min_down_time",
        "up_time_before",
        "down_time_before",
        "ramp_limit_up",
        "ramp_limit_down",
        "ramp_limit_start_up",
        "ramp_limit_shut_down",
        "linearized_uc",
    ]

    def __init__(self, pypsa_network: Any = None, strict_mode: bool = True):
        self.source = pypsa_network
        self.target = None
        self.strict_mode = strict_mode
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.unsupported_features: Dict[str, List[str]] = {}

    def import_network(self, strict_mode: Optional[bool] = None):
        """Import the PyPSA network and return a qwenaplan Network.

        Parameters
        ----------
        strict_mode : bool, optional
            Override the instance strict_mode setting.

        Returns
        -------
        qwenaplan.network.Network
            The imported qwenaplan network.

        Raises
        ------
        RuntimeError
            If strict_mode is True and unsupported features are found.
        """
        if strict_mode is not None:
            self.strict_mode = strict_mode

        from qwenaplan.network import Network

        self.target = Network()
        self.warnings = []
        self.errors = []
        self.unsupported_features = {}

        # 1. Import snapshots
        self._import_snapshots()

        # 2. Import buses
        self._import_buses()

        # 3. Import generators
        self._import_generators()

        # 4. Import loads (must come before lines so they exist when KCL is built)
        self._import_loads()

        # 5. Import lines
        self._import_lines()

        # 6. Import links
        self._import_links()

        # 7. Import storage units
        self._import_storage_units()

        # 8. Check for unsupported components
        self._check_unsupported()

        if self.errors and self.strict_mode:
            raise RuntimeError(
                f"Import failed with {len(self.errors)} errors:\n"
                + "\n".join(f"  - {e}" for e in self.errors)
            )

        return self.target

    # ------------------------------------------------------------------
    # Snapshot import
    # ------------------------------------------------------------------

    def _import_snapshots(self):
        """Convert PyPSA snapshots to polars Series."""
        snapshots = self._get_pypsa_attr("snapshots")
        if snapshots is None:
            # Default to 24 hourly snapshots if none defined
            logger.warning("No snapshots found in PyPSA network, using default 24 hours")
            snapshots = range(24)

        # Handle different snapshot formats
        if hasattr(snapshots, "to_series"):
            # pandas Index -> polars Series
            name = getattr(snapshots, "name", "time") or "time"
            values = [str(s) for s in snapshots]
            self.snapshots = pl.Series(name=name, values=values)
        elif hasattr(snapshots, "tolist"):
            # numpy array or similar
            self.snapshots = pl.Series(
                name="time", values=[str(s) for s in snapshots.tolist()]
            )
        else:
            # Assume iterable
            self.snapshots = pl.Series(
                name="time", values=[str(s) for s in snapshots]
            )

        self.target.snapshots = self.snapshots

    # ------------------------------------------------------------------
    # Bus import
    # ------------------------------------------------------------------

    def _import_buses(self):
        """Import buses with supported attributes only."""
        buses = self._get_pypsa_components("Bus")
        if not buses:
            buses = self._get_pypsa_components("bus")

        for name, attrs in buses.items():
            try:
                self.target.add(
                    "Bus",
                    name=name,
                    v_nom=self._get_attr(attrs, "v_nom", 1.0),
                    carrier=self._get_attr(attrs, "carrier", "AC"),
                    x=self._get_attr(attrs, "x", 0.0),
                    y=self._get_attr(attrs, "y", 0.0),
                )
            except Exception as e:
                self.errors.append(f"Failed to import bus '{name}': {e}")

    # ------------------------------------------------------------------
    # Generator import
    # ------------------------------------------------------------------

    def _import_generators(self):
        """Import generators, flagging unsupported attributes."""
        generators = self._get_pypsa_components("Generator")
        if not generators:
            generators = self._get_pypsa_components("generator")

        for name, attrs in generators.items():
            # Check for unsupported attributes
            unsupported = self._check_component_attrs(attrs, self.INVESTMENT_ATTRS + self.UNIT_COMMITMENT_ATTRS)
            if unsupported:
                self.warnings.append(
                    f"Generator '{name}' has unsupported attributes: {unsupported}"
                )
                if self.strict_mode:
                    self.errors.append(
                        f"Generator '{name}' has unsupported features in strict mode: {unsupported}"
                    )
                    continue

            try:
                bus = self._resolve_bus(attrs, "bus")
                if bus is None:
                    self.errors.append(f"Generator '{name}' references unknown bus")
                    continue

                p_max_pu = self._get_attr(attrs, "p_max_pu", None)
                # Handle p_max_pu as series/profile - not fully supported
                if hasattr(p_max_pu, "to_series"):
                    p_max_pu = p_max_pu.iloc[0] if len(p_max_pu) > 0 else None
                    self.warnings.append(
                        f"Generator '{name}' has time-varying p_max_pu, using first value"
                    )

                self.target.add(
                    "Generator",
                    name=name,
                    bus=bus,
                    p_nom=self._get_attr(attrs, "p_nom", 0.0),
                    marginal_cost=self._get_attr(attrs, "marginal_cost", 0.0),
                    carrier=self._get_attr(attrs, "carrier", ""),
                    p_max_pu=p_max_pu,
                )
            except Exception as e:
                self.errors.append(f"Failed to import generator '{name}': {e}")

    # ------------------------------------------------------------------
    # Load import
    # ------------------------------------------------------------------

    def _import_loads(self):
        """Import loads. PyPSA Loads have a (possibly time-varying) ``p_set``.

        Time-varying ``p_set`` is supported here because Load is a parameter,
        not a variable — it costs nothing extra in the LP. We coerce the
        PyPSA series (pandas) to a Polars Series aligned with our snapshots.
        """
        loads = self._get_pypsa_components("Load")
        if not loads:
            loads = self._get_pypsa_components("load")

        for name, attrs in loads.items():
            try:
                bus = self._resolve_bus(attrs, "bus")
                if bus is None:
                    self.errors.append(f"Load '{name}' references unknown bus")
                    continue

                p_set = self._get_attr(attrs, "p_set", 0.0)
                # Coerce pandas/numpy time-series to a polars Series.
                if hasattr(p_set, "to_list") and not isinstance(p_set, (int, float)):
                    p_set = pl.Series(name=self.target.snapshots.name, values=list(p_set))
                elif hasattr(p_set, "tolist"):
                    p_set = pl.Series(name=self.target.snapshots.name, values=p_set.tolist())

                self.target.add(
                    "Load",
                    name=name,
                    bus=bus,
                    p_set=p_set,
                    carrier=self._get_attr(attrs, "carrier", ""),
                )
            except Exception as e:
                self.errors.append(f"Failed to import load '{name}': {e}")

    # ------------------------------------------------------------------
    # Line import
    # ------------------------------------------------------------------

    def _import_lines(self):
        """Import lines, flagging unsupported attributes."""
        # Try both "Line" and "Line" naming conventions
        lines = self._get_pypsa_components("Line")
        if not lines:
            lines = self._get_pypsa_components("line")

        for name, attrs in lines.items():
            # Check for investment attributes
            unsupported = self._check_component_attrs(attrs, self.INVESTMENT_ATTRS)
            if unsupported:
                self.warnings.append(
                    f"Line '{name}' has unsupported attributes: {unsupported}"
                )
                if self.strict_mode:
                    self.errors.append(
                        f"Line '{name}' has unsupported features in strict mode: {unsupported}"
                    )
                    continue

            try:
                from_bus = self._resolve_bus(attrs, "bus0")
                to_bus = self._resolve_bus(attrs, "bus1")
                if from_bus is None or to_bus is None:
                    self.errors.append(f"Line '{name}' references unknown bus")
                    continue

                # Get reactance - try both x_pu and x (per-unit reactance)
                x_pu = self._get_attr(attrs, "x_pu", None)
                if x_pu is None:
                    x_pu = self._get_attr(attrs, "x", None)

                if x_pu is None:
                    # Try to get from type - not supported, use default
                    self.warnings.append(
                        f"Line '{name}' has no reactance (x_pu), using default 0.1"
                    )
                    x_pu = 0.1

                s_nom = self._get_attr(attrs, "s_nom", 0.0)

                self.target.add(
                    "ACLine",
                    name=name,
                    from_bus=from_bus,
                    to_bus=to_bus,
                    x_pu=x_pu,
                    s_nom=s_nom,
                )
            except Exception as e:
                self.errors.append(f"Failed to import line '{name}': {e}")

    # ------------------------------------------------------------------
    # Link import
    # ------------------------------------------------------------------

    def _import_links(self):
        """Import links, flagging unsupported attributes."""
        links = self._get_pypsa_components("Link")
        if not links:
            links = self._get_pypsa_components("link")

        for name, attrs in links.items():
            # Check for unsupported attributes
            unsupported = self._check_component_attrs(
                attrs, self.INVESTMENT_ATTRS + ["p_set", "p_min_pu", "p_max_pu"]
            )
            if unsupported:
                self.warnings.append(
                    f"Link '{name}' has unsupported attributes: {unsupported}"
                )
                if self.strict_mode:
                    self.errors.append(
                        f"Link '{name}' has unsupported features in strict mode: {unsupported}"
                    )
                    continue

            try:
                from_bus = self._resolve_bus(attrs, "bus0")
                to_bus = self._resolve_bus(attrs, "bus1")
                if from_bus is None or to_bus is None:
                    self.errors.append(f"Link '{name}' references unknown bus")
                    continue

                self.target.add(
                    "Link",
                    name=name,
                    from_bus=from_bus,
                    to_bus=to_bus,
                    p_nom=self._get_attr(attrs, "p_nom", 0.0),
                    carrier=self._get_attr(attrs, "carrier", ""),
                    efficiency=self._get_attr(attrs, "efficiency", 1.0),
                )
            except Exception as e:
                self.errors.append(f"Failed to import link '{name}': {e}")

    # ------------------------------------------------------------------
    # Storage unit import
    # ------------------------------------------------------------------

    def _import_storage_units(self):
        """Import storage units, flagging unsupported attributes."""
        storage_units = self._get_pypsa_components("StorageUnit")
        if not storage_units:
            storage_units = self._get_pypsa_components("storage_unit")

        for name, attrs in storage_units.items():
            # Check for unsupported attributes
            unsupported = self._check_component_attrs(
                attrs,
                self.INVESTMENT_ATTRS
                + [
                    "cyclic_state_of_charge",
                    "cyclic_state_of_charge_per_period",
                    "state_of_charge_initial_per_period",
                    "cyclic",
                    "inflow_weight",
                    "standing_loss",
                    "p_set",
                ],
            )
            if unsupported:
                self.warnings.append(
                    f"StorageUnit '{name}' has unsupported attributes: {unsupported}"
                )
                if self.strict_mode:
                    self.errors.append(
                        f"StorageUnit '{name}' has unsupported features in strict mode: {unsupported}"
                    )
                    continue

            try:
                bus = self._resolve_bus(attrs, "bus")
                if bus is None:
                    self.errors.append(f"StorageUnit '{name}' references unknown bus")
                    continue

                # Map PyPSA attributes to qwenaplan parameters
                p_nom_in = self._get_attr(attrs, "p_nom_in", 0.0)
                p_nom_out = self._get_attr(attrs, "p_nom_out", 0.0)

                # PyPSA uses p_nom for total, p_nom_in/p_nom_out may not exist
                if p_nom_in == 0 and p_nom_out == 0:
                    p_nom_total = self._get_attr(attrs, "p_nom", 0.0)
                    if p_nom_total > 0:
                        p_nom_in = p_nom_total
                        p_nom_out = p_nom_total

                eff_store = self._get_attr(attrs, "eff_store", 1.0)
                eff_dispatch = self._get_attr(attrs, "eff_dispatch", 1.0)
                initial_soc = self._get_attr(attrs, "state_of_charge_initial", 0.0)
                soc_max = self._get_attr(attrs, "state_of_charge_max", None)
                soc_min = self._get_attr(attrs, "state_of_charge_min", None)
                inflow = self._get_attr(attrs, "inflow", 0.0)
                carrier = self._get_attr(attrs, "carrier", "")

                self.target.add(
                    "StorageUnit",
                    name=name,
                    bus=bus,
                    e_nom=self._get_attr(attrs, "e_nom", 0.0),
                    p_nom_in=p_nom_in,
                    p_nom_out=p_nom_out,
                    eff_in=eff_store,
                    eff_out=eff_dispatch,
                    initial_soc=initial_soc,
                    soc_min=soc_min,
                    soc_max=soc_max,
                    influx=inflow,
                    carrier=carrier,
                )
            except Exception as e:
                self.errors.append(f"Failed to import storage unit '{name}': {e}")

    # ------------------------------------------------------------------
    # Unsupported feature detection
    # ------------------------------------------------------------------

    def _check_unsupported(self):
        """Check for and report unsupported components/features."""
        # Check for transformers
        transformers = self._get_pypsa_components("Transformer")
        if not transformers:
            transformers = self._get_pypsa_components("transformer")
        if transformers:
            self.unsupported_features["Transformers"] = list(transformers.keys())
            self.warnings.append(
                f"Skipping {len(transformers)} transformer(s): {list(transformers.keys())}. "
                "Transformers are not supported in qwenaplan."
            )
            self.errors.append(
                f"Network contains {len(transformers)} transformer(s) which are not supported."
            )

        # Check for stores
        stores = self._get_pypsa_components("Store")
        if not stores:
            stores = self._get_pypsa_components("store")
        if stores:
            self.unsupported_features["Stores"] = list(stores.keys())
            self.warnings.append(
                f"Skipping {len(stores)} store(s): {list(stores.keys())}. "
                "Stores are not supported in qwenaplan."
            )
            self.errors.append(
                f"Network contains {len(stores)} store(s) which are not supported."
            )

        # Check for shunt impedances
        shunts = self._get_pypsa_components("ShuntImpedance")
        if not shunts:
            shunts = self._get_pypsa_components("shunt_impedances")
        if shunts:
            self.unsupported_features["ShuntImpedances"] = list(shunts.keys())
            self.warnings.append(
                f"Skipping {len(shunts)} shunt impedance(s): {list(shunts.keys())}. "
                "Shunt impedances are AC power flow features not supported in DC OPF."
            )

        # Check for global constraints
        global_constraints = self._get_pypsa_components("GlobalConstraint")
        if not global_constraints:
            global_constraints = self._get_pypsa_components("global_constraints")
        if global_constraints:
            self.unsupported_features["GlobalConstraints"] = list(global_constraints.keys())
            self.warnings.append(
                f"Skipping {len(global_constraints)} global constraint(s): {list(global_constraints.keys())}. "
                "Global constraints (e.g., CO2 limits) are not supported in qwenaplan."
            )

        # Check for investment period definitions
        investment_periods = self._get_pypsa_attr("investment_periods")
        if investment_periods is not None:
            self.unsupported_features["InvestmentPeriods"] = ["investment_periods defined"]
            self.warnings.append(
                "Investment periods defined in network but multi-horizon optimization "
                "is not supported in qwenaplan."
            )

        # Check all components for investment/unit commitment attributes
        for component_name in ["Generator", "generator", "Line", "line", "Link", "link",
                               "StorageUnit", "storage_unit", "Store", "store"]:
            components = self._get_pypsa_components(component_name)
            for name, attrs in components.items():
                investment_attrs = self._check_component_attrs(attrs, self.INVESTMENT_ATTRS)
                if investment_attrs:
                    key = f"{component_name}_investment"
                    if key not in self.unsupported_features:
                        self.unsupported_features[key] = []
                    self.unsupported_features[key].append(f"{name}: {investment_attrs}")

                uc_attrs = self._check_component_attrs(attrs, self.UNIT_COMMITMENT_ATTRS)
                if uc_attrs:
                    key = f"{component_name}_unit_commitment"
                    if key not in self.unsupported_features:
                        self.unsupported_features[key] = []
                    self.unsupported_features[key].append(f"{name}: {uc_attrs}")

    # ------------------------------------------------------------------
    # Helper methods for PyPSA network interaction
    # ------------------------------------------------------------------

    def _get_pypsa_attr(self, attr_name: str, default=None):
        """Get an attribute from the PyPSA network."""
        if self.source is None:
            return default
        try:
            return getattr(self.source, attr_name, default)
        except Exception:
            return default

    def _get_pypsa_components(self, component_name: str):
        """Get all components of a given type from the PyPSA network.

        Handles different PyPSA naming conventions and data formats.
        """
        if self.source is None:
            return {}

        try:
            # Try different naming conventions
            for name in [component_name, component_name.lower(),
                         component_name.capitalize()]:
                try:
                    components = getattr(self.source, name)
                    if components is None:
                        continue

                    # Handle different data formats
                    if hasattr(components, "to_pandas"):
                        components = components.to_pandas()
                    if hasattr(components, "df"):
                        components = components.df

                    if hasattr(components, "to_dict"):
                        # pandas DataFrame -> dict of records
                        return components.to_dict(orient="index")
                    elif isinstance(components, dict):
                        return components
                    elif hasattr(components, "items"):
                        return dict(components.items())
                except AttributeError:
                    continue

            # Try via components API (newer PyPSA versions)
            try:
                comps = self.source.components
                for name in [component_name, component_name.lower()]:
                    comp = comps.get(name)
                    if comp:
                        df = getattr(comp, "pandas", None) or getattr(comp, "data", None)
                        if df is not None:
                            if hasattr(df, "to_pandas"):
                                df = df.to_pandas()
                            if hasattr(df, "to_dict"):
                                return df.to_dict(orient="index")
            except (AttributeError, KeyError):
                pass

        except Exception as e:
            self.warnings.append(f"Error reading components '{component_name}': {e}")

        return {}

    def _get_attr(self, attrs: dict, attr_name: str, default=None):
        """Get an attribute value from a component's attribute dict."""
        if attrs is None:
            return default
        return attrs.get(attr_name, default)

    def _resolve_bus(self, attrs: dict, bus_attr: str = "bus"):
        """Resolve a bus reference to an actual Bus object in the target network."""
        bus_name = self._get_attr(attrs, bus_attr)
        if bus_name is None:
            return None
        return self.target.buses.get(bus_name)

    def _check_component_attrs(self, attrs: dict, attr_names: list):
        """Check if a component has any of the specified unsupported attributes.

        Returns
        -------
        list[str]
            List of unsupported attribute names found.
        """
        if attrs is None:
            return []
        return [attr for attr in attr_names if attr in attrs and attrs[attr]]

    def get_warnings(self) -> List[str]:
        """Return list of warnings during import."""
        return self.warnings

    def get_unsupported_summary(self) -> Dict[str, List[str]]:
        """Return summary of unsupported features found."""
        return self.unsupported_features

    def __repr__(self) -> str:
        return (
            f"<PyPSAImporter(strict_mode={self.strict_mode}, "
            f"warnings={len(self.warnings)}, errors={len(self.errors)})>"
        )
