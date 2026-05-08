from .base import (
    Component,
    PowerElement,
    BranchElement,
    _solution_as,
    _VarContainer,
    _SolContainer,
)
import polars as pl
import pyoframe as pf
from .physics import DCPhysics  # Import the logic


# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------

class Bus(Component):
    """Bus (node) in the power grid.

    Parameters
    ----------
    name : str
        Unique identifier for the bus
    network : Network
        Reference to the parent network
    v_nom : float, default 1.0
        Nominal voltage in kV (used for impedance base conversion)
    carrier : str, default "AC"
        Energy carrier / bus type (organizational only)
    x : float, default 0.0
        X coordinate for geographic visualization
    y : float, default 0.0
        Y coordinate for geographic visualization
    """

    def __init__(self, name: str, network: "Network", v_nom: float = 1.0,
                 carrier: str = "AC", x: float = 0.0, y: float = 0.0):
        super().__init__(name, network)
        self.v_nom = v_nom
        self.carrier = carrier
        self.x = x
        self.y = y

    def setup_variables(self):
        # Phase angle (rad) - indexed by snapshots. Required by KVL on AC lines.
        # No nodal injection variable: KCL is closed (Σ injections = Σ flows)
        # with Loads providing demand and Generators providing supply.
        df = self.network.snapshots.to_frame()
        self.var.theta_t = pf.Variable(df)

    def setup_variables_for_model(self, model):
        # Add variables to the model so they can be used in expressions
        setattr(model, f"theta_{self.name}", self.var.theta_t)

    def setup_constraints(self, model):
        # Trigger the physics engine to build KCL for this bus
        DCPhysics.apply_kirchhoff_current_law(self, model)

    def setup_objective(self, network):
        # Bus has no direct contribution to the objective function
        pass

    def __repr__(self) -> str:
        return f"<Bus(name={self.name}, v_nom={self.v_nom})>"


# ---------------------------------------------------------------------------
# ACLine
# ---------------------------------------------------------------------------

class ACLine(BranchElement):
    def __init__(
        self,
        name: str,
        network: "Network",
        from_bus: "Bus",
        to_bus: "Bus",
        x_pu: float = 0.1,
        s_nom: float = 0.0,
    ):
        super().__init__(name, network, from_bus, to_bus)
        self.x_pu = x_pu
        self.s_nom = s_nom

    def setup_variables(self):
        # Explicit flow variable for numerical stability - indexed by snapshots
        df = self.network.snapshots.to_frame()
        self.var.p_t = pf.Variable(df)

    def setup_variables_for_model(self, model):
        setattr(model, f"p_{self.name}", self.var.p_t)

    def setup_constraints(self, model):
        # 1. Apply KVL to link this line's flow to bus angles
        DCPhysics.apply_kirchhoff_voltage_law(self, model)

        # 2. Add thermal limits (if defined). Two separate constraints —
        # Python's chained comparison ``a <= x <= b`` silently discards one
        # half when the operands are pyoframe objects, so we split them.
        if self.s_nom > 0:
            setattr(model, f"line_limit_upper_{self.name}", self.var.p_t <= self.s_nom)
            setattr(model, f"line_limit_lower_{self.name}", self.var.p_t >= -self.s_nom)

    def setup_objective(self, network):
        # ACLine has no direct contribution to the objective function
        pass

    def __repr__(self) -> str:
        return f"<ACLine(name={self.name}, {self.from_bus.name}->{self.to_bus.name}, x_pu={self.x_pu}, s_nom={self.s_nom})>"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class _GeneratorVar(_VarContainer):
    """Generator variable container with the ``p_pu_t`` view."""

    @property
    def p_pu_t(self):
        """Capacity factor expression: ``p_t / p_nom`` (pyoframe expression).

        Useful for writing constraints like ``gen.var.p_pu_t <= 0.8`` against
        scaled limits, or for slicing the LP by carrier capacity factor.
        """
        gen = self._owner
        if gen.p_nom == 0:
            raise ZeroDivisionError(
                f"Generator '{gen.name}' has p_nom=0; p_pu_t is undefined."
            )
        return self.p_t / gen.p_nom


class _GeneratorSol(_SolContainer):
    """Generator solution container with the ``p_pu_t`` view."""

    @property
    def p_pu_t(self) -> pl.DataFrame:
        """Solved capacity factor: ``p / p_nom`` per snapshot."""
        gen = self._owner
        if gen.p_nom == 0:
            raise ZeroDivisionError(
                f"Generator '{gen.name}' has p_nom=0; p_pu_t is undefined."
            )
        df = self.p_t  # cols: <snapshot>, p
        return df.with_columns((pl.col("p") / gen.p_nom).alias("p_pu")).select(
            [self._owner.network.snapshots.name, "p_pu"]
        )


class Generator(PowerElement):
    """Power generator component.

    Parameters
    ----------
    name : str
        Unique identifier for the generator
    network : Network
        Reference to the parent network
    bus : Bus
        Bus where the generator is connected
    p_nom : float, default 0.0
        Nominal power limit (MW)
    marginal_cost : float, default 0.0
        Marginal cost per MWh
    carrier : str, default ""
        Energy carrier (e.g., "AC", "solar", "wind") - organizational metadata
    p_max_pu : float | pl.Series | None, default None
        Maximum power as per-unit of p_nom (None = use p_nom directly).
        Can be a static value (0-1) or a time-series profile.
    p_min_pu : float | pl.Series | None, default None
        Minimum power as per-unit of p_nom (None = use 0).
        Can be a static value (0-1) or a time-series profile.
        Must be <= p_max_pu wherever both are defined.
    ramp_limit_up : float, default None
        Maximum ramp-up rate as per-unit of p_nom per snapshot (None = no limit).
        0.2 means the generator can increase by at most 20% of p_nom between snapshots.
    ramp_limit_down : float, default None
        Maximum ramp-down rate as per-unit of p_nom per snapshot (None = no limit).
        0.2 means the generator can decrease by at most 20% of p_nom between snapshots.
    """

    def __init__(
        self,
        name: str,
        network: "Network",
        bus: "Bus",
        p_nom: float = 0.0,
        marginal_cost: float = 0.0,
        carrier: str = "",
        p_max_pu: float | pl.Series = None,
        p_min_pu: float | pl.Series = None,
        ramp_limit_up: float = None,
        ramp_limit_down: float = None,
    ):
        super().__init__(name, network, bus)
        self.p_nom = p_nom
        self.marginal_cost = marginal_cost
        self.carrier = carrier

        # Validate p_min_pu <= p_max_pu for static values
        if (isinstance(p_max_pu, (int, float)) and isinstance(p_min_pu, (int, float)) and
            p_max_pu is not None and p_min_pu is not None):
            if p_min_pu > p_max_pu:
                raise ValueError(
                    f"Generator '{self.name}': p_min_pu ({p_min_pu}) must be <= p_max_pu ({p_max_pu})"
                )

        self._p_max_pu_profile = p_max_pu if isinstance(p_max_pu, pl.Series) else None
        self._p_max_pu = p_max_pu if not isinstance(p_max_pu, pl.Series) else None
        self._p_min_pu_profile = p_min_pu if isinstance(p_min_pu, pl.Series) else None
        self._p_min_pu = p_min_pu if not isinstance(p_min_pu, pl.Series) else None

        self.ramp_limit_up = ramp_limit_up
        self.ramp_limit_down = ramp_limit_down

    def _var_container_cls(self):
        return _GeneratorVar

    def _sol_container_cls(self):
        return _GeneratorSol

    def setup_variables(self):
        df = self.network.snapshots.to_frame()
        self.var.p_t = pf.Variable(df)

    def setup_variables_for_model(self, model):
        setattr(model, f"p_{self.name}", self.var.p_t)

    def setup_constraints(self, model):
        snapshots = self.network.snapshots
        dim_name = snapshots.name

        # 1. Build max limit (as Param for pyoframe compatibility)
        if self._p_max_pu_profile is not None:
            # Profile-based max: validate against profile min
            profile_min = self._p_max_pu_profile.min()
            if self._p_min_pu is not None and not isinstance(self._p_min_pu, pl.Series):
                if self._p_min_pu > profile_min:
                    raise ValueError(
                        f"Generator '{self.name}': p_min_pu ({self._p_min_pu}) exceeds "
                        f"minimum p_max_pu profile value ({profile_min})"
                    )
            # Create DataFrame with snapshots index and scaled profile values
            max_df = snapshots.to_frame().with_columns(
                (self._p_max_pu_profile * self.p_nom).alias("max")
            )
            max_param = pf.Param(max_df)
        elif self._p_max_pu is not None:
            max_param = pf.Param(snapshots.to_frame().with_columns(
                pl.lit(self.p_nom * self._p_max_pu).alias("max")
            ))
        else:
            max_param = pf.Param(snapshots.to_frame().with_columns(
                pl.lit(self.p_nom).alias("max")
            ))

        # 2. Build min limit (as Param for pyoframe compatibility)
        if self._p_min_pu_profile is not None:
            # Profile-based min: validate against static max
            profile_max = self._p_min_pu_profile.max()
            if self._p_max_pu is not None and not isinstance(self._p_max_pu, pl.Series):
                if self._p_max_pu < profile_max:
                    raise ValueError(
                        f"Generator '{self.name}': p_max_pu ({self._p_max_pu}) is below "
                        f"maximum p_min_pu profile value ({profile_max})"
                    )
            min_df = snapshots.to_frame().with_columns(
                (self._p_min_pu_profile * self.p_nom).alias("min")
            )
            min_param = pf.Param(min_df)
        elif self._p_min_pu is not None:
            min_param = pf.Param(snapshots.to_frame().with_columns(
                pl.lit(self.p_nom * self._p_min_pu).alias("min")
            ))
        else:
            min_param = pf.Param(snapshots.to_frame().with_columns(
                pl.lit(0.0).alias("min")
            ))

        # Apply min/max bounds
        setattr(model, f"gen_limit_{self.name}", self.var.p_t <= max_param)
        setattr(model, f"gen_lower_{self.name}", self.var.p_t >= min_param)

        # 3. Ramping constraints (if defined)
        if self.ramp_limit_up is not None or self.ramp_limit_down is not None:
            if self.ramp_limit_up is not None:
                ramp_up_limit = self.p_nom * self.ramp_limit_up
                p_current = self.var.p_t.next(dim_name)
                p_previous = self.var.p_t.drop_extras()
                setattr(
                    model,
                    f"gen_ramp_up_{self.name}",
                    p_current - p_previous <= ramp_up_limit,
                )

            if self.ramp_limit_down is not None:
                ramp_down_limit = self.p_nom * self.ramp_limit_down
                p_current = self.var.p_t.next(dim_name)
                p_previous = self.var.p_t.drop_extras()
                setattr(
                    model,
                    f"gen_ramp_down_{self.name}",
                    p_previous - p_current <= ramp_down_limit,
                )

    def setup_objective(self, network):
        # Annualised marginal cost contribution per snapshot:
        #   p(t) * marginal_cost * duration(t) * weighting(t)
        if self.marginal_cost != 0:
            cost_weight = network._objective_cost_weight_param()
            network._add_to_objective(self.var.p_t * self.marginal_cost * cost_weight)

    def __repr__(self) -> str:
        return f"<Generator(name={self.name}, bus={self.bus.name}, p_nom={self.p_nom}, marginal_cost={self.marginal_cost})>"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

class _LoadSol(_SolContainer):
    """Load solution container.

    Load has no decision variables — ``p_set`` is parameter data. We expose
    it on ``sol.p_t`` (rather than ``var.p_t``) for symmetry with other
    components: the question "what is the load drawing per snapshot?" is a
    *result-side* question, available pre-solve too.
    """

    @property
    def p_t(self) -> pl.DataFrame:
        ld = self._owner
        snapshots = ld.network.snapshots
        return snapshots.to_frame().with_columns(
            ld._p_set_series.alias("p")
        )


class Load(PowerElement):
    """Demand at a bus.

    A Load is a *parameter*, not a decision variable: ``p_set`` is fixed input
    data that withdraws power from its bus. Loads contribute ``-p_set`` to KCL.

    If you want unmet-demand semantics ("load shedding"), do not use a Load
    with optional satisfaction. Instead, add a high-marginal-cost generator at
    the same bus (e.g. ``marginal_cost=10_000``) and let the optimizer trade
    shedding cost against generation cost. This keeps the model linear and
    makes the cost of shedding explicit.

    Loads have **no** ``var`` attribute (no decision variables); ``sol.p_t``
    returns the parameter data for symmetry with Generator/etc.

    Parameters
    ----------
    name : str
        Unique identifier.
    network : Network
    bus : Bus
        Bus where the load sits (power is withdrawn here).
    p_set : float | pl.Series, default 0.0
        Demand in MW. Scalar applies to every snapshot; a Polars Series must
        be aligned with ``network.snapshots``.
    carrier : str, default ""
        Energy carrier (organisational metadata).
    """

    def __init__(
        self,
        name: str,
        network: "Network",
        bus: "Bus",
        p_set: float | pl.Series = 0.0,
        carrier: str = "",
    ):
        super().__init__(name, network, bus)
        self.carrier = carrier
        self._p_set_profile = p_set if isinstance(p_set, pl.Series) else None
        self._p_set = p_set if not isinstance(p_set, pl.Series) else None
        # Load has no var bag — drop it so attribute-typo errors are loud.
        # (sol stays; it returns the parameter as a DataFrame.)
        del self.var

    def _sol_container_cls(self):
        return _LoadSol

    def setup_variables(self):
        # Load has no decision variable. Pre-build a Polars Series aligned
        # to snapshots for KCL Param construction.
        snapshots = self.network.snapshots
        if self._p_set_profile is not None:
            self._p_set_series = self._p_set_profile
        else:
            self._p_set_series = pl.Series([float(self._p_set)] * len(snapshots))

    def setup_variables_for_model(self, model):
        # Nothing to register: parameters are built lazily during KCL.
        pass

    def setup_constraints(self, model):
        # No own constraints. KCL (in DCPhysics) consumes get_p_net().
        pass

    def setup_objective(self, network):
        # Loads are parameters; they do not contribute to the objective.
        pass

    def get_p_net(self):
        """Net injection from this load = ``-p_set`` (withdrawal).

        Returned as a pyoframe ``Param`` so it can be added directly into the
        KCL expression alongside Variable expressions from generators/storage.
        """
        snapshots = self.network.snapshots
        df = snapshots.to_frame().with_columns(
            (-self._p_set_series).alias("p_set_neg")
        )
        return pf.Param(df)

    # ``sol.p_t`` returns the demand magnitude (positive ``p_set``); the
    # *bus injection* is the negation. Bus views use this so a row of
    # n.views[bus].sol.p_t sums to zero — the load contributes its
    # withdrawal as a negative value, matching the var-side ``-Param``.
    # On the var side ``injection_at`` already pulls ``get_p_net()`` and
    # gets the correct sign for free; the sol side cannot share that
    # path because ``sol.p_t`` is shaped as a polars frame.
    def sol_sign_at(self, bus) -> int:
        return -1

    def __repr__(self) -> str:
        p = self._p_set if self._p_set is not None else "<profile>"
        return f"<Load(name={self.name}, bus={self.bus.name}, p_set={p})>"


# ---------------------------------------------------------------------------
# Link
# ---------------------------------------------------------------------------

class Link(BranchElement):
    """Controllable flow between two buses.

    Parameters
    ----------
    name : str
        Unique identifier for the link
    network : Network
        Reference to the parent network
    from_bus : Bus
        From bus (source)
    to_bus : Bus
        To bus (destination)
    p_nom : float, default 0.0
        Nominal power limit (MW)
    carrier : str, default ""
        Energy carrier (organizational metadata)
    efficiency : float, default 1.0
        Efficiency factor (power loss as fraction, 1.0 = no loss)
    """

    def __init__(
        self,
        name: str,
        network: "Network",
        from_bus: "Bus",
        to_bus: "Bus",
        p_nom: float = 0.0,
        carrier: str = "",
        efficiency: float = 1.0,
    ):
        super().__init__(name, network, from_bus, to_bus)
        self.p_nom = p_nom
        self.carrier = carrier
        self.efficiency = efficiency

    def setup_variables(self):
        df = self.network.snapshots.to_frame()
        self.var.p_t = pf.Variable(df)

    def setup_variables_for_model(self, model):
        setattr(model, f"p_{self.name}", self.var.p_t)

    def setup_constraints(self, model):
        # Split: chained comparison silently drops one half on pyoframe objs.
        setattr(model, f"link_limit_upper_{self.name}", self.var.p_t <= self.p_nom)
        setattr(model, f"link_limit_lower_{self.name}", self.var.p_t >= -self.p_nom)

    def setup_objective(self, network):
        pass

    def __repr__(self) -> str:
        return f"<Link(name={self.name}, {self.from_bus.name}->{self.to_bus.name}, p_nom={self.p_nom})>"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _storage_nameplate(storage) -> float:
    """Pick the larger of ``p_nom_in`` / ``p_nom_out`` as the nameplate
    against which a storage's net power is normalised.

    Returns ``None`` if both are ``None``; callers should raise.
    """
    candidates = [v for v in (storage.p_nom_in, storage.p_nom_out) if v is not None]
    if not candidates:
        return None
    return max(candidates)


class _StorageBaseVar(_VarContainer):
    """``_StorageBase.var`` with ``soc_pu_t`` and ``p_pu_t`` views."""

    @property
    def soc_pu_t(self):
        """SOC fill level expression: ``soc_t / e_nom``."""
        s = self._owner
        if s.e_nom == 0:
            raise ZeroDivisionError(
                f"Storage '{s.name}' has e_nom=0; soc_pu_t is undefined."
            )
        return self.soc_t / s.e_nom

    @property
    def p_pu_t(self):
        """Net power expression: ``(p_out_t - p_in_t) / nameplate``.

        Sign: + = discharging (injection into bus). Nameplate is the larger
        of ``p_nom_in`` and ``p_nom_out``. Raises if both are ``None`` (no
        capacity defined to normalise against).
        """
        s = self._owner
        nameplate = _storage_nameplate(s)
        if nameplate is None:
            raise ValueError(
                f"Storage '{s.name}' has neither p_nom_in nor p_nom_out set; "
                f"p_pu_t is undefined."
            )
        if nameplate == 0:
            raise ZeroDivisionError(
                f"Storage '{s.name}' nameplate is 0; p_pu_t is undefined."
            )
        return (self.p_out_t - self.p_in_t) / nameplate


class _StorageBaseSol(_SolContainer):
    """``_StorageBase.sol`` with ``soc_pu_t`` and ``p_pu_t`` views."""

    @property
    def soc_pu_t(self) -> pl.DataFrame:
        s = self._owner
        if s.e_nom == 0:
            raise ZeroDivisionError(
                f"Storage '{s.name}' has e_nom=0; soc_pu_t is undefined."
            )
        df = self.soc_t  # cols: <snapshot>, soc
        snap = s.network.snapshots.name
        return df.with_columns((pl.col("soc") / s.e_nom).alias("soc_pu")).select(
            [snap, "soc_pu"]
        )

    @property
    def p_pu_t(self) -> pl.DataFrame:
        s = self._owner
        nameplate = _storage_nameplate(s)
        if nameplate is None:
            raise ValueError(
                f"Storage '{s.name}' has neither p_nom_in nor p_nom_out set; "
                f"p_pu_t is undefined."
            )
        if nameplate == 0:
            raise ZeroDivisionError(
                f"Storage '{s.name}' nameplate is 0; p_pu_t is undefined."
            )
        snap = s.network.snapshots.name
        out = self.p_out_t  # cols: snap, p_out
        ins = self.p_in_t   # cols: snap, p_in
        return (
            out.join(ins, on=snap)
            .with_columns(((pl.col("p_out") - pl.col("p_in")) / nameplate).alias("p_pu"))
            .select([snap, "p_pu"])
        )


class _StorageBase(PowerElement):
    """
    Generic base class for storage components.

    This class defines the generic storage interface with p_in (inflow) and p_out (outflow)
    variables. It can be used directly or as a base for composite storage types.

    Follows PyPSA convention:
    soc(t) = soc(t-1) + p_in(t) * eff_in - p_out(t) / eff_out + influx(t)

    Parameters
    ----------
    name : str
        Unique identifier for the storage unit
    network : Network
        Reference to the parent network
    bus : Bus
        Bus where the storage is connected
    e_nom : float
        Nominal energy capacity (MWh)
    p_nom_in : float | None, default None
        Maximum inflow power (MW). ``None`` = no upper bound (unlimited).
        ``0.0`` = literally zero (no charging / no pump available).
    p_nom_out : float | None, default None
        Maximum outflow power (MW). Same convention as ``p_nom_in``.
    eff_in : float, default 1.0
        Inflow efficiency (0 < eff, can be >1 for abstract cases)
    eff_out : float, default 1.0
        Outflow efficiency (0 < eff)
    initial_soc : float, default 0.0
        Initial state of charge (MWh)
    soc_min : float, optional
        Minimum SOC limit (None = 0)
    soc_max : float, optional
        Maximum SOC limit (None = e_nom)
    influx : float, default 0.0
        Constant static influx (MW equivalent, e.g., water inflow)
    """

    def __init__(
        self,
        name: str,
        network: "Network",
        bus: "Bus",
        e_nom: float,
        p_nom_in: float | None = None,
        p_nom_out: float | None = None,
        eff_in: float = 1.0,
        eff_out: float = 1.0,
        initial_soc: float = 0.0,
        soc_min: float = None,
        soc_max: float = None,
        influx: float = 0.0,
        carrier: str = "",
    ):
        super().__init__(name, network, bus)
        self.e_nom = e_nom
        self.carrier = carrier
        self.p_nom_in = p_nom_in
        self.p_nom_out = p_nom_out
        self.eff_in = eff_in
        self.eff_out = eff_out
        self.initial_soc = initial_soc
        self.soc_min = soc_min if soc_min is not None else 0.0
        self.soc_max = soc_max if soc_max is not None else e_nom
        self._influx = influx
        self._influx_profile = None

    def _var_container_cls(self):
        return _StorageBaseVar

    def _sol_container_cls(self):
        return _StorageBaseSol

    def set_influx_profile(self, profile: pl.Series):
        """Set a time-series profile for the static influx."""
        self._influx_profile = profile
        self._influx = None

    def setup_variables(self):
        """Create pyoframe variables indexed by snapshots."""
        df = self.network.snapshots.to_frame()
        self.var.soc_t = pf.Variable(df)
        self.var.p_in_t = pf.Variable(df)
        self.var.p_out_t = pf.Variable(df)

        # Prepare influx as time-series aligned with snapshots
        if self._influx_profile is not None:
            self._influx_series = self._influx_profile
        else:
            self._influx_series = pl.Series([self._influx] * len(df))

    def setup_variables_for_model(self, model):
        """Add variables to the model."""
        setattr(model, f"soc_{self.name}", self.var.soc_t)
        setattr(model, f"p_in_{self.name}", self.var.p_in_t)
        setattr(model, f"p_out_{self.name}", self.var.p_out_t)

    def setup_objective(self, network):
        """Storage base has no direct contribution to the objective function."""
        pass

    def _setup_soc_constraints(self, model):
        """
        Setup SOC balance and bounds constraints.

        Energy-units balance with ``Δt`` from ``network.snapshot_duration``:
        the historical per-snapshot form falls out at ``duration=1.0``.
        """
        snapshots = self.network.snapshots
        dim_name = snapshots.name
        duration_series = self.network.snapshot_duration

        # 1. SOC Balance (PyPSA convention; energy units)
        delta_t_per_in = self.eff_in * duration_series
        delta_t_per_out = duration_series / self.eff_out
        df_in = snapshots.to_frame().with_columns(delta_t_per_in.alias("k"))
        df_out = snapshots.to_frame().with_columns(delta_t_per_out.alias("k"))
        eff_in_dt = pf.Param(df_in)
        eff_out_dt = pf.Param(df_out)
        # influx is in MW; multiply by duration to get MWh per snapshot.
        influx_energy_df = (
            snapshots.to_frame()
            .with_columns(self._influx_series.alias("influx"))
            .with_columns(duration_series.alias("dt"))
            .with_columns((pl.col("influx") * pl.col("dt")).alias("e_influx"))
            .select([snapshots.name, "e_influx"])
        )
        influx_energy_param = pf.Param(influx_energy_df)

        soc = self.var.soc_t
        p_in = self.var.p_in_t
        p_out = self.var.p_out_t

        soc_balance = (
            soc.next(dim_name)
            == (
                soc.drop_extras()
                + p_in.next(dim_name) * eff_in_dt.drop_extras()
                - p_out.next(dim_name) * eff_out_dt.drop_extras()
                + influx_energy_param.drop_extras()
            )
        )
        setattr(model, f"soc_balance_{self.name}", soc_balance)

        # 2. Initial SOC constraint (first snapshot's own Δt).
        first_snapshot = snapshots[0]
        first_filter = pl.col(snapshots.name) == first_snapshot
        initial_soc_constraint = (
            soc.filter(first_filter)
            == self.initial_soc
            + p_in.filter(first_filter) * eff_in_dt.filter(first_filter)
            - p_out.filter(first_filter) * eff_out_dt.filter(first_filter)
            + influx_energy_param.filter(first_filter)
        )
        setattr(model, f"initial_soc_{self.name}", initial_soc_constraint)

        # 3. Inflow power limits.
        setattr(model, f"p_in_floor_{self.name}", p_in >= 0)
        if self.p_nom_in is not None:
            setattr(model, f"p_in_cap_{self.name}", p_in <= self.p_nom_in)

        # 4. Outflow power limits.
        setattr(model, f"p_out_floor_{self.name}", p_out >= 0)
        if self.p_nom_out is not None:
            setattr(model, f"p_out_cap_{self.name}", p_out <= self.p_nom_out)

        # 5. SOC bounds
        setattr(model, f"soc_min_{self.name}", soc >= self.soc_min)
        setattr(model, f"soc_max_{self.name}", soc <= self.soc_max)

    def setup_constraints(self, model):
        """Default storage constraints. Subclasses can override if needed."""
        self._setup_soc_constraints(model)

    def get_p_net(self):
        """
        Return net power injection for KCL.

        Net power = p_out - p_in
        Positive = power injected into bus
        Negative = power withdrawn from bus
        """
        return self.var.p_out_t - self.var.p_in_t


class StorageUnit(_StorageBase):
    """
    Concrete storage unit with charge/discharge semantics.

    This is the standard storage unit where:
    - p_in maps to p_store (charging)
    - p_out maps to p_dispatch (discharging)
    """

    def setup_constraints(self, model):
        """StorageUnit-specific constraints. Same physics as the base class."""
        self._setup_soc_constraints(model)

    def get_p_net(self):
        return self.var.p_out_t - self.var.p_in_t

    # ---- Backward-compatibility aliases (scalar parameter renames only).
    # The plan keeps these because they're user-facing alternative names for
    # the *same* attributes — they are not time-vectorized variables.
    @property
    def p_nom_store(self):
        return self.p_nom_in

    @property
    def p_nom_dispatch(self):
        return self.p_nom_out

    @property
    def eff_store(self):
        return self.eff_in

    @property
    def eff_dispatch(self):
        return self.eff_out

    def __repr__(self) -> str:
        return f"<StorageUnit(name={self.name}, bus={self.bus.name}, e_nom={self.e_nom})>"


# ---------------------------------------------------------------------------
# Storage composites (StorageComposite + Battery + PumpedHydroStorage)
# ---------------------------------------------------------------------------
#
# Single concrete composite. Every composite owns:
#   - one ``_StorageBase``        (the SOC reservoir, p_in / p_out)
#   - one ``Generator``           (the bus-facing electrical variable)
#   - one coupling constraint     (eq. linking inner storage and generator)
#
# Battery and PumpedHydroStorage are now thin subclasses that just rename a
# couple of constructor kwargs (``p_nom`` ↔ ``p_nom_in/out`` ↔
# ``p_nom_turbine`` / ``p_nom_pump``) and set defaults. The shared
# var/sol windows below cover both shapes — there is no longer a
# ``_BatteryVar/Sol`` vs ``_PHSVar/Sol`` split.

class _StorageCompositeVar(_VarContainer):
    """Composite ``var`` window onto its inner storage + generator.

    Every ``_t`` entry is a property that reads through to the inner
    components — there are no own variables on this container. The names
    match the public surface promised by both Battery and PHS in the
    pre-refactor world.
    """

    @property
    def soc_t(self):
        return self._owner._storage.var.soc_t

    @property
    def p_in_t(self):
        return self._owner._storage.var.p_in_t

    @property
    def p_out_t(self):
        return self._owner._storage.var.p_out_t

    @property
    def p_t(self):
        # Net electrical injection. Now a real pyoframe variable
        # (``generator.var.p_t``), tied to the storage by the coupling
        # constraint ``p_t = p_out * gen_eff_out − p_in / gen_eff_in``
        # installed in ``StorageComposite.setup_constraints``.
        return self._owner._generator.var.p_t

    @property
    def p_store_t(self):
        return self._owner._storage.var.p_in_t

    @property
    def p_dispatch_t(self):
        return self._owner._storage.var.p_out_t

    @property
    def soc_pu_t(self):
        return self._owner._storage.var.soc_pu_t

    @property
    def p_pu_t(self):
        c = self._owner
        nameplate = c._p_nom_electrical
        if nameplate == 0:
            raise ZeroDivisionError(
                f"{type(c).__name__} '{c.name}' has p_nom_electrical=0; "
                f"p_pu_t is undefined."
            )
        return c._generator.var.p_t / nameplate


class _StorageCompositeSol(_SolContainer):
    """Composite ``sol`` window onto inner solved values.

    ``p_t`` reads the generator's solved variable directly — no
    hand-subtraction. The shape (snapshot col + value col) matches every
    other component's ``sol.p_t``.
    """

    @property
    def soc_t(self) -> pl.DataFrame:
        return self._owner._storage.sol.soc_t

    @property
    def p_in_t(self) -> pl.DataFrame:
        return self._owner._storage.sol.p_in_t

    @property
    def p_out_t(self) -> pl.DataFrame:
        return self._owner._storage.sol.p_out_t

    @property
    def p_t(self) -> pl.DataFrame:
        return _solution_as(self._owner._generator.var.p_t, "p")

    @property
    def p_store_t(self) -> pl.DataFrame:
        return _solution_as(self._owner._storage.var.p_in_t, "p_store")

    @property
    def p_dispatch_t(self) -> pl.DataFrame:
        return _solution_as(self._owner._storage.var.p_out_t, "p_dispatch")

    @property
    def soc_pu_t(self) -> pl.DataFrame:
        return self._owner._storage.sol.soc_pu_t

    @property
    def p_pu_t(self) -> pl.DataFrame:
        c = self._owner
        nameplate = c._p_nom_electrical
        if nameplate == 0:
            raise ZeroDivisionError(
                f"{type(c).__name__} '{c.name}' has p_nom_electrical=0; "
                f"p_pu_t is undefined."
            )
        snap = c.network.snapshots.name
        df = self.p_t  # cols: snap, p
        return df.with_columns((pl.col("p") / nameplate).alias("p_pu")).select(
            [snap, "p_pu"]
        )


class StorageComposite:
    """Concrete composite of an inner ``_StorageBase`` + ``Generator``.

    The user-facing ``Battery`` and ``PumpedHydroStorage`` classes are
    thin subclasses that fix defaults and rename a couple of kwargs.

    The coupling constraint installed at ``setup_constraints`` time is::

        generator.var.p_t == storage.var.p_out_t * gen_efficiency
                           - storage.var.p_in_t  / gen_efficiency_in

    With ``gen_efficiency = gen_efficiency_in = 1.0`` this reduces to
    ``p_out − p_in`` (today's Battery rule). With
    ``gen_efficiency = 0.9, gen_efficiency_in = 1.0`` it reduces to
    ``p_out × 0.9 − p_in`` (today's PHS rule, where the pump is treated
    as electrical-1:1 and the pump losses are folded into ``eff_store``
    on the storage side). Asymmetric inverter losses (modern batteries)
    are expressed by setting both knobs.

    Parameters
    ----------
    name : str
        Unique identifier for the composite (used as the prefix for the
        inner components' names by default).
    network : Network
    bus : Bus
        Bus where the composite injects / withdraws.
    e_nom : float
        Reservoir capacity (MWh).
    p_nom_in, p_nom_out : float | None
        Storage-side nameplates (charging / discharging respectively).
        See ``_StorageBase`` for the ``None`` semantics.
    p_nom_electrical : float
        Generator nameplate (MW). Bounds ``generator.var.p_t``.
    eff_in, eff_out : float, default 1.0
        Storage-side efficiencies (DC chemistry / hydraulic).
    gen_efficiency : float, default 1.0
        Discharge-side electrical conversion (mechanical→electrical).
    gen_efficiency_in : float, default 1.0
        Charge-side electrical conversion (electrical→mechanical).
    initial_soc, soc_min, soc_max : float, optional
        Forwarded to ``_StorageBase``.
    influx : float, default 0.0
        Forwarded to ``_StorageBase``.
    storage_name, generator_name : str, optional
        Inner-component names. Default: ``f"{name}_storage"`` and
        ``f"{name}_generator"``. Battery overrides ``storage_name=name``
        to preserve its pre-refactor inner naming.
    carrier : str, default ""
        Forwarded to the inner ``_StorageBase`` for organisational
        metadata.
    """

    def __init__(
        self,
        name: str,
        network: "Network",
        bus: "Bus",
        *,
        e_nom: float,
        p_nom_in: float | None,
        p_nom_out: float | None,
        p_nom_electrical: float,
        eff_in: float = 1.0,
        eff_out: float = 1.0,
        gen_efficiency: float = 1.0,
        gen_efficiency_in: float = 1.0,
        initial_soc: float = 0.0,
        soc_min: float = None,
        soc_max: float = None,
        influx: float = 0.0,
        storage_name: str | None = None,
        generator_name: str | None = None,
        carrier: str = "",
    ):
        self.name = name
        self.network = network
        self.bus = bus

        # Coupling parameters live on the composite (mutable post-construction
        # at the user's discretion; the constraint is built at create_model
        # time so changes before then take effect).
        self.gen_efficiency = gen_efficiency
        self.gen_efficiency_in = gen_efficiency_in
        # Stored separately from the generator's p_nom because subclass
        # property setters may rewrite both — we want a single source of
        # truth for the per-unit normaliser.
        self._p_nom_electrical = p_nom_electrical

        self._storage = _StorageBase(
            name=storage_name if storage_name is not None else f"{name}_storage",
            network=network,
            bus=bus,
            e_nom=e_nom,
            p_nom_in=p_nom_in,
            p_nom_out=p_nom_out,
            eff_in=eff_in,
            eff_out=eff_out,
            initial_soc=initial_soc,
            soc_min=soc_min,
            soc_max=soc_max,
            influx=influx,
            carrier=carrier,
        )
        self._generator = Generator(
            name=generator_name if generator_name is not None else f"{name}_generator",
            network=network,
            bus=bus,
            p_nom=p_nom_electrical,
            carrier=carrier,
        )
        self.var = _StorageCompositeVar(owner=self)
        self.sol = _StorageCompositeSol(owner=self)

    @property
    def storage(self) -> "_StorageBase":
        """Access the internal storage component."""
        return self._storage

    @property
    def generator(self) -> "Generator":
        """Access the internal generator component."""
        return self._generator

    # ---- shared property delegates --------------------------------------
    # The kwargs that map 1:1 to the inner storage live here so subclasses
    # don't repeat themselves. Subclasses add their own renamed kwargs
    # (``p_nom`` for Battery, ``p_nom_turbine`` / ``p_nom_pump`` for PHS).

    @property
    def e_nom(self): return self._storage.e_nom
    @e_nom.setter
    def e_nom(self, v): self._storage.e_nom = v

    @property
    def eff_store(self): return self._storage.eff_in
    @eff_store.setter
    def eff_store(self, v): self._storage.eff_in = v

    @property
    def eff_dispatch(self): return self._storage.eff_out
    @eff_dispatch.setter
    def eff_dispatch(self, v): self._storage.eff_out = v

    @property
    def initial_soc(self): return self._storage.initial_soc
    @initial_soc.setter
    def initial_soc(self, v): self._storage.initial_soc = v

    @property
    def soc_min(self): return self._storage.soc_min
    @soc_min.setter
    def soc_min(self, v): self._storage.soc_min = v

    @property
    def soc_max(self): return self._storage.soc_max
    @soc_max.setter
    def soc_max(self, v): self._storage.soc_max = v

    @property
    def influx(self): return self._storage._influx
    @influx.setter
    def influx(self, v): self._storage._influx = v

    # ---- lifecycle hooks (mirror Component's contract) ------------------

    def setup_variables(self):
        self._storage.setup_variables()
        self._generator.setup_variables()

    def setup_variables_for_model(self, model):
        self._storage.setup_variables_for_model(model)
        self._generator.setup_variables_for_model(model)

    def setup_constraints(self, model):
        """Install storage SOC and the coupling row.

        We deliberately do **not** call ``self._generator.setup_constraints``:
        :class:`Generator` would install ``0 <= p_t <= p_nom``, but for a
        composite storage the electrical variable must be free to swing
        negative on charging. The feasible band is fully determined by
        the storage's ``p_in`` / ``p_out`` bounds plus the coupling
        equality below — adding the generator's own bounds would clip
        the charge half on batteries, and on PHS would clip whenever
        ``p_nom_pump > p_nom_turbine``.
        """
        self._storage._setup_soc_constraints(model)

        # Coupling: electrical p_t = p_out × η_out − p_in / η_in.
        # ``η_out = η_in = 1`` collapses to ``p_out − p_in`` (Battery
        # rule). ``η_out = 0.9, η_in = 1`` collapses to
        # ``p_out × 0.9 − p_in`` (today's PHS rule).
        coupling = (
            self._generator.var.p_t
            == self._storage.var.p_out_t * self.gen_efficiency
            - self._storage.var.p_in_t / self.gen_efficiency_in
        )
        setattr(model, f"{self.name}_coupling", coupling)

    def setup_objective(self, network):
        """Delegate to inner components (Generator may carry a marginal cost)."""
        self._storage.setup_objective(network)
        self._generator.setup_objective(network)

    def get_p_net(self):
        """KCL injection: the generator's electrical variable.

        The coupling constraint guarantees this equals the bus-side
        electrical net (charge minus discharge, both signed by their
        inverter / generator efficiencies).
        """
        return self._generator.var.p_t

    # ---- bus-injection abstraction --------------------------------------
    # Mirrors :meth:`Component.injection_at` / :meth:`Component.injection_sign_at`
    # so KCL and the views layer can call them uniformly across power
    # elements, branches, and composites. Composite isn't a Component
    # subclass (it owns two of them), so we duplicate the small default.

    def injection_sign_at(self, bus) -> int:
        return +1

    def sol_sign_at(self, bus) -> int:
        # ``sol.p_t`` is the generator's solved value — already in
        # injection convention (+ = bus injection).
        return +1

    def injection_at(self, bus):
        sign = self.injection_sign_at(bus)
        base = self.get_p_net()
        return base if sign == 1 else sign * base

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__}(name={self.name}, bus={self.bus.name}, "
            f"e_nom={self.e_nom}, p_nom_electrical={self._p_nom_electrical})>"
        )


class Battery(StorageComposite):
    """Battery — composite storage with symmetric AC charge/discharge.

    A battery has a single inverter doing both AC→DC (charging) and
    DC→AC (discharging). The default ``gen_efficiency = gen_efficiency_in
    = 1.0`` reproduces the pre-refactor rule ``p_t = p_out − p_in``;
    users with measured asymmetric inverter losses can set both knobs.

    Parameters
    ----------
    name, network, bus
        See :class:`StorageComposite`.
    e_nom : float
        Energy capacity (MWh).
    p_nom : float
        Maximum power (MW), applied to charge, discharge, and the inverter.
    eff_store, eff_dispatch : float, default 1.0
        DC-side charging / discharging efficiencies.
    gen_efficiency, gen_efficiency_in : float, default 1.0
        Inverter electrical efficiencies (discharge / charge sides).
    initial_soc, soc_min, soc_max : float, optional
        SOC parameters.
    influx : float, default 0.0
        External energy inflow (rarely used for a battery; default 0).
    """

    def __init__(
        self,
        name: str,
        network: "Network",
        bus: "Bus",
        *,
        e_nom: float,
        p_nom: float,
        eff_store: float = 1.0,
        eff_dispatch: float = 1.0,
        gen_efficiency: float = 1.0,
        gen_efficiency_in: float = 1.0,
        initial_soc: float = 0.0,
        soc_min: float = None,
        soc_max: float = None,
        influx: float = 0.0,
    ):
        super().__init__(
            name, network, bus,
            e_nom=e_nom,
            p_nom_in=p_nom, p_nom_out=p_nom,
            p_nom_electrical=p_nom,
            eff_in=eff_store, eff_out=eff_dispatch,
            gen_efficiency=gen_efficiency,
            gen_efficiency_in=gen_efficiency_in,
            initial_soc=initial_soc, soc_min=soc_min, soc_max=soc_max,
            influx=influx,
            # Preserve the pre-refactor inner-storage name (just ``name``)
            # so any external code reaching for ``model.soc_<name>`` keeps
            # working.
            storage_name=name,
        )

    # Single ``p_nom`` knob spans both inner storage rails and the inverter.
    @property
    def p_nom(self): return self._storage.p_nom_in   # in == out for battery
    @p_nom.setter
    def p_nom(self, v):
        self._storage.p_nom_in = v
        self._storage.p_nom_out = v
        self._generator.p_nom = v
        self._p_nom_electrical = v


class PumpedHydroStorage(StorageComposite):
    """Pumped-hydro plant — reservoir + turbine generator + (optional) pump.

    The default ``gen_efficiency = 0.9, gen_efficiency_in = 1.0``
    reproduces the pre-refactor PHS rule ``p_t = p_out × 0.9 − p_in``,
    where the pump is treated as electrically 1:1 and pump losses are
    folded into ``eff_store`` on the storage side.

    Variable mapping
    ----------------
    - ``storage.p_in``  → pumping (water back into the reservoir)
    - ``storage.p_out`` → water flow through the turbine
    - ``generator.p``   → electrical bus injection

    Parameters
    ----------
    name, network, bus
        See :class:`StorageComposite`.
    e_nom : float
        Reservoir capacity (MWh equivalent).
    p_nom_turbine : float
        Maximum turbine power (MW). Bounds both the water dispatch and
        the generator's electrical output.
    p_nom_pump : float, default 0.0
        Maximum pumping power (MW); 0.0 means no pumping rail.
    eff_store, eff_dispatch : float, default 1.0
        Pump and hydraulic efficiencies.
    gen_efficiency : float, default 0.9
        Mechanical→electrical (generator) efficiency.
    gen_efficiency_in : float, default 1.0
        Electrical→mechanical (pump motor) efficiency. The historical
        PHS rule ignored this term, hence the default of 1.0; set to
        less than 1 to model pump-motor electrical loss explicitly.
    initial_soc, soc_min, soc_max : float, optional
        SOC parameters.
    influx : float, default 0.0
        Constant water inflow to the reservoir (MW equivalent).
    """

    def __init__(
        self,
        name: str,
        network: "Network",
        bus: "Bus",
        *,
        e_nom: float,
        p_nom_turbine: float,
        p_nom_pump: float = 0.0,
        eff_store: float = 1.0,
        eff_dispatch: float = 1.0,
        gen_efficiency: float = 0.9,
        gen_efficiency_in: float = 1.0,
        initial_soc: float = 0.0,
        soc_min: float = None,
        soc_max: float = None,
        influx: float = 0.0,
    ):
        super().__init__(
            name, network, bus,
            e_nom=e_nom,
            p_nom_in=p_nom_pump, p_nom_out=p_nom_turbine,
            p_nom_electrical=p_nom_turbine,
            eff_in=eff_store, eff_out=eff_dispatch,
            gen_efficiency=gen_efficiency,
            gen_efficiency_in=gen_efficiency_in,
            initial_soc=initial_soc, soc_min=soc_min, soc_max=soc_max,
            influx=influx,
        )

    @property
    def p_nom_pump(self): return self._storage.p_nom_in
    @p_nom_pump.setter
    def p_nom_pump(self, v): self._storage.p_nom_in = v

    @property
    def p_nom_turbine(self): return self._storage.p_nom_out
    @p_nom_turbine.setter
    def p_nom_turbine(self, v):
        self._storage.p_nom_out = v
        self._generator.p_nom = v
        self._p_nom_electrical = v
