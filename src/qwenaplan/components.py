from abc import ABC, abstractmethod
from .base import Component, PowerElement, BranchElement, _solution_as
import polars as pl
import pyoframe as pf
from .physics import DCPhysics  # Import the logic


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
        # with Loads providing demand and Generators providing supply. If the
        # user wants a slack, they add a high-marginal-cost generator
        # explicitly.
        df = self.network.snapshots.to_frame()
        self.theta = pf.Variable(df)

    def setup_variables_for_model(self, model):
        # Add variables to the model so they can be used in expressions
        setattr(model, f"theta_{self.name}", self.theta)

    def setup_constraints(self, model):
        # Trigger the physics engine to build KCL for this bus
        DCPhysics.apply_kirchhoff_current_law(self, model)

    def setup_objective(self, network):
        # Bus has no direct contribution to the objective function
        pass

    @property
    def theta_t(self) -> pl.DataFrame:
        """Solved phase angle (rad) per snapshot."""
        return _solution_as(self.theta, "theta")

    def __repr__(self) -> str:
        return f"<Bus(name={self.name}, v_nom={self.v_nom})>"


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
        # Explicit flow variable for numerical stability (as discussed) - indexed by snapshots
        df = self.network.snapshots.to_frame()
        self.p = pf.Variable(df)

    def setup_variables_for_model(self, model):
        # Add variables to the model so they can be used in expressions
        setattr(model, f"p_{self.name}", self.p)

    def setup_constraints(self, model):
        # 1. Apply KVL to link this line's flow to bus angles
        DCPhysics.apply_kirchhoff_voltage_law(self, model)

        # 2. Add thermal limits (if defined). Two separate constraints —
        # Python's chained comparison ``a <= x <= b`` silently discards one
        # half when the operands are pyoframe objects, so we split them.
        if self.s_nom > 0:
            setattr(model, f"line_limit_upper_{self.name}", self.p <= self.s_nom)
            setattr(model, f"line_limit_lower_{self.name}", self.p >= -self.s_nom)

    def setup_objective(self, network):
        # ACLine has no direct contribution to the objective function
        pass

    @property
    def p_t(self) -> pl.DataFrame:
        """Solved line flow (MW) per snapshot. Positive = from_bus → to_bus."""
        return _solution_as(self.p, "p")

    def __repr__(self) -> str:
        return f"<ACLine(name={self.name}, {self.from_bus.name}->{self.to_bus.name}, x_pu={self.x_pu}, s_nom={self.s_nom})>"


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

    def setup_variables(self):
        # Create P_t variable - indexed by snapshots
        df = self.network.snapshots.to_frame()
        self.p = pf.Variable(df)

    def setup_variables_for_model(self, model):
        # Add variables to the model so they can be used in expressions
        setattr(model, f"p_{self.name}", self.p)

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
        
        # Apply min/max bounds: min_param <= p <= max_param
        setattr(model, f"gen_limit_{self.name}", self.p <= max_param)
        setattr(model, f"gen_lower_{self.name}", self.p >= min_param)
        
        # 3. Ramping constraints (if defined)
        if self.ramp_limit_up is not None or self.ramp_limit_down is not None:
            if self.ramp_limit_up is not None:
                ramp_up_limit = self.p_nom * self.ramp_limit_up
                # p(t) - p(t-1) <= ramp_up_limit
                # Use .next() to shift: p.next() is p(t), p is p(t-1) (with first element dropped)
                p_current = self.p.next(dim_name)
                p_previous = self.p.drop_extras()
                setattr(
                    model,
                    f"gen_ramp_up_{self.name}",
                    p_current - p_previous <= ramp_up_limit,
                )
            
            if self.ramp_limit_down is not None:
                ramp_down_limit = self.p_nom * self.ramp_limit_down
                # p(t-1) - p(t) <= ramp_down_limit
                p_current = self.p.next(dim_name)
                p_previous = self.p.drop_extras()
                setattr(
                    model,
                    f"gen_ramp_down_{self.name}",
                    p_previous - p_current <= ramp_down_limit,
                )

    def setup_objective(self, network):
        # Annualised marginal cost contribution per snapshot:
        #   p(t) * marginal_cost * duration(t) * weighting(t)
        # Defaults of duration=1, weighting=1 reproduce the previous
        # per-snapshot cost convention exactly.
        if self.marginal_cost != 0:
            cost_weight = network._objective_cost_weight_param()
            network._add_to_objective(self.p * self.marginal_cost * cost_weight)

    @property
    def p_t(self) -> pl.DataFrame:
        """Solved generator output (MW) per snapshot."""
        return _solution_as(self.p, "p")

    def __repr__(self) -> str:
        return f"<Generator(name={self.name}, bus={self.bus.name}, p_nom={self.p_nom}, marginal_cost={self.marginal_cost})>"


class Load(PowerElement):
    """Demand at a bus.

    A Load is a *parameter*, not a decision variable: ``p_set`` is fixed input
    data that withdraws power from its bus. Loads contribute ``-p_set`` to KCL.

    If you want unmet-demand semantics ("load shedding"), do not use a Load
    with optional satisfaction. Instead, add a high-marginal-cost generator at
    the same bus (e.g. ``marginal_cost=10_000``) and let the optimizer trade
    shedding cost against generation cost. This keeps the model linear and
    makes the cost of shedding explicit.

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

    def setup_variables(self):
        # Load has no decision variable. We pre-build a Polars Series aligned
        # to snapshots so setup_constraints / KCL can construct a Param.
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

    @property
    def p_t(self) -> pl.DataFrame:
        """Demand (MW) per snapshot. Symmetric with Generator.p_t for tooling.

        This is parameter data, available even before the model is solved.
        """
        snapshots = self.network.snapshots
        return snapshots.to_frame().with_columns(
            self._p_set_series.alias("p")
        )

    def __repr__(self) -> str:
        p = self._p_set if self._p_set is not None else "<profile>"
        return f"<Load(name={self.name}, bus={self.bus.name}, p_set={p})>"


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
        self.p = pf.Variable(df)

    def setup_variables_for_model(self, model):
        # Add variables to the model so they can be used in expressions
        setattr(model, f"p_{self.name}", self.p)

    def setup_constraints(self, model):
        # Split: chained comparison silently drops one half on pyoframe objs.
        setattr(model, f"link_limit_upper_{self.name}", self.p <= self.p_nom)
        setattr(model, f"link_limit_lower_{self.name}", self.p >= -self.p_nom)

    def setup_objective(self, network):
        # Link has no direct contribution to the objective function
        pass

    @property
    def p_t(self) -> pl.DataFrame:
        """Solved link flow (MW) per snapshot. Positive = from_bus → to_bus."""
        return _solution_as(self.p, "p")

    def __repr__(self) -> str:
        return f"<Link(name={self.name}, {self.from_bus.name}->{self.to_bus.name}, p_nom={self.p_nom})>"


class _StorageBase(PowerElement):
    """
    Generic base class for storage components.
    
    This class defines the generic storage interface with p_in (inflow) and p_out (outflow)
    variables. It can be used directly or as a base for composite storage types.
    
    The SOC dynamics are defined here and reused by all storage types.
    
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

    def set_influx_profile(self, profile: pl.Series):
        """Set a time-series profile for the static influx."""
        self._influx_profile = profile
        self._influx = None

    def setup_variables(self):
        """Create pyoframe variables indexed by snapshots."""
        df = self.network.snapshots.to_frame()
        self.soc = pf.Variable(df)
        self.p_in = pf.Variable(df)
        self.p_out = pf.Variable(df)

        # Prepare influx as time-series aligned with snapshots
        if self._influx_profile is not None:
            self._influx_series = self._influx_profile
        else:
            self._influx_series = pl.Series([self._influx] * len(df))

    def setup_variables_for_model(self, model):
        """Add variables to the model."""
        setattr(model, f"soc_{self.name}", self.soc)
        setattr(model, f"p_in_{self.name}", self.p_in)
        setattr(model, f"p_out_{self.name}", self.p_out)

    def setup_objective(self, network):
        """Storage base has no direct contribution to the objective function."""
        pass

    def _setup_soc_constraints(self, model):
        """
        Setup SOC balance and bounds constraints.

        This is the core storage logic that all storage types share. The
        balance is in **energy units** (MWh), so each power term is multiplied
        by the snapshot duration (hours) of the *destination* snapshot. With
        the default duration of 1.0 the equation matches the historical
        per-snapshot form.

        ``self._influx_series`` (set in ``setup_variables``) is the source of
        truth for influx; subclasses configure it via ``influx`` /
        ``set_influx_profile`` rather than passing a separate Param.
        """
        snapshots = self.network.snapshots
        dim_name = snapshots.name
        # Pre-multiply influx by duration in polars so we work with a single
        # *energy-per-snapshot* Param. This avoids needing to multiply
        # Param×Param at pyoframe level, which is awkward across .next()/.drop_extras().
        duration_series = self.network.snapshot_duration

        # 1. SOC Balance Equation (PyPSA convention; energy units)
        #    soc(t+1) = soc(t) + p_in(t+1)*eff_in*Δt(t+1)
        #             - p_out(t+1)/eff_out*Δt(t+1)
        #             + influx(t+1)*Δt(t+1)
        # Variables: .next() shifts to t+1 values.
        # Params: pyoframe aligns by index when added; .drop_extras() projects
        # onto the constraint dimension. We construct each per-step Param
        # already pre-multiplied by Δt to keep the algebra linear in pyoframe.
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

        soc_balance = (
            self.soc.next(dim_name)
            == (
                self.soc.drop_extras()
                + self.p_in.next(dim_name) * eff_in_dt.drop_extras()
                - self.p_out.next(dim_name) * eff_out_dt.drop_extras()
                + influx_energy_param.drop_extras()
            )
        )
        setattr(model, f"soc_balance_{self.name}", soc_balance)

        # 2. Initial SOC constraint (first snapshot's own Δt).
        first_snapshot = snapshots[0]
        first_filter = pl.col(snapshots.name) == first_snapshot
        initial_soc_constraint = (
            self.soc.filter(first_filter)
            == self.initial_soc
            + self.p_in.filter(first_filter) * eff_in_dt.filter(first_filter)
            - self.p_out.filter(first_filter) * eff_out_dt.filter(first_filter)
            + influx_energy_param.filter(first_filter)
        )
        setattr(model, f"initial_soc_{self.name}", initial_soc_constraint)

        # 3. Inflow power limits. Two separate constraints — Python's chained
        # ``0 <= x <= n`` silently keeps only one half on pyoframe objects.
        # Convention: ``p_nom_in = 0`` means literally 0 MW (no charging /
        # no pump available). ``p_nom_in = None`` means unbounded above.
        setattr(model, f"p_in_floor_{self.name}", self.p_in >= 0)
        if self.p_nom_in is not None:
            setattr(model, f"p_in_cap_{self.name}", self.p_in <= self.p_nom_in)

        # 4. Outflow power limits, same convention.
        setattr(model, f"p_out_floor_{self.name}", self.p_out >= 0)
        if self.p_nom_out is not None:
            setattr(model, f"p_out_cap_{self.name}", self.p_out <= self.p_nom_out)

        # 5. SOC bounds
        setattr(model, f"soc_min_{self.name}", self.soc >= self.soc_min)
        setattr(model, f"soc_max_{self.name}", self.soc <= self.soc_max)

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
        return self.p_out - self.p_in

    @property
    def soc_t(self) -> pl.DataFrame:
        """Solved state of charge (MWh) per snapshot."""
        return _solution_as(self.soc, "soc")

    @property
    def p_in_t(self) -> pl.DataFrame:
        """Solved charging power (MW) per snapshot."""
        return _solution_as(self.p_in, "p_in")

    @property
    def p_out_t(self) -> pl.DataFrame:
        """Solved discharging power (MW) per snapshot."""
        return _solution_as(self.p_out, "p_out")


class StorageUnit(_StorageBase):
    """
    Concrete storage unit with charge/discharge semantics.
    
    This is the standard storage unit where:
    - p_in maps to p_store (charging)
    - p_out maps to p_dispatch (discharging)
    
    For backward compatibility and direct use cases.
    """

    def setup_constraints(self, model):
        """StorageUnit-specific constraints. Same physics as the base class."""
        self._setup_soc_constraints(model)

    def get_p_net(self):
        """
        Return net power injection for KCL.

        Net power = dispatch (out) - store (in)
        Positive = power injected into bus
        Negative = power withdrawn from bus
        """
        return self.p_out - self.p_in

    # Backward compatibility properties
    @property
    def p_store(self):
        """Alias for p_in (charging)."""
        return self.p_in

    @property
    def p_dispatch(self):
        """Alias for p_out (discharging)."""
        return self.p_out

    @property
    def p_nom_store(self):
        """Alias for p_nom_in."""
        return self.p_nom_in

    @property
    def p_nom_dispatch(self):
        """Alias for p_nom_out."""
        return self.p_nom_out

    @property
    def eff_store(self):
        """Alias for eff_in."""
        return self.eff_in

    @property
    def eff_dispatch(self):
        """Alias for eff_out."""
        return self.eff_out

    def __repr__(self) -> str:
        return f"<StorageUnit(name={self.name}, bus={self.bus.name}, e_nom={self.e_nom})>"


class StorageComposite(ABC):
    """
    Abstract base class for composite storage components.
    
    A composite storage holds a _StorageBase instance and optionally a Generator,
    and maps the storage's p_in/p_out to domain-specific variables.
    
    Subclasses define:
    - How p_in/p_out map to their specific variables (e.g., charge/discharge, pump/turbine)
    - Whether a generator is needed (e.g., pumped hydro has a generator, battery doesn't)
    - Any coupling constraints between storage and generator
    """

    def __init__(self, name: str, network: "Network", bus: "Bus"):
        self.name = name
        self.network = network
        self.bus = bus
        self._storage: _StorageBase = None
        self._generator: Generator = None

    @property
    def storage(self) -> _StorageBase:
        """Access the internal storage component."""
        return self._storage

    @property
    def generator(self) -> Generator:
        """Access the internal generator component (if any)."""
        return self._generator

    def setup_variables(self):
        """Setup variables for all internal components."""
        self._storage.setup_variables()
        if self._generator:
            self._generator.setup_variables()

    def setup_variables_for_model(self, model):
        """Add variables to the model."""
        self._storage.setup_variables_for_model(model)
        if self._generator:
            self._generator.setup_variables_for_model(model)

    @abstractmethod
    def setup_constraints(self, model):
        """Setup constraints - must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_p_net(self):
        """Return net power injection for KCL."""
        pass

    def setup_objective(self, network):
        """Delegate objective contribution to internal components."""
        self._storage.setup_objective(network)
        if self._generator:
            self._generator.setup_objective(network)

    # Common property delegates
    @property
    def soc(self):
        """State of charge (from storage)."""
        return self._storage.soc

    @property
    def soc_t(self) -> pl.DataFrame:
        """Solved state of charge (MWh) per snapshot."""
        return self._storage.soc_t


class PumpedHydroStorage(StorageComposite):
    """
    Composite class for pumped hydro storage.

    Combines a _StorageBase (reservoir/SOC) with a Generator (turbine output).
    The storage p_out represents water flow through the turbine,
    and the generator output is the electrical power produced.

    Variable mapping:
    - storage.p_in -> pump (pumping water back to reservoir)
    - storage.p_out -> dispatch (water flowing through turbine)
    - generator.p -> electrical output (dispatch * gen_efficiency)

    Parameters
    ----------
    name : str
        Unique identifier for the pumped hydro storage
    network : Network
        Reference to the parent network
    bus : Bus
        Bus where the storage is connected
    e_nom : float
        Nominal energy capacity of reservoir (MWh equivalent)
    p_nom_turbine : float
        Maximum turbine power (MW) - limits both dispatch and generator
    p_nom_pump : float, default 0
        Maximum pumping power (MW), 0 = unlimited
    eff_store : float, default 1.0
        Pumping efficiency
    eff_dispatch : float, default 1.0
        Hydraulic efficiency (water to mechanical)
    gen_efficiency : float, default 0.9
        Generator efficiency (mechanical to electrical)
    initial_soc : float, default 0.0
        Initial state of charge
    soc_min : float, optional
        Minimum SOC limit
    soc_max : float, optional
        Maximum SOC limit (defaults to e_nom)
    influx : float, default 0.0
        Constant static influx (water inflow)
    """

    def __init__(
        self,
        name: str,
        network: "Network",
        bus: "Bus",
        e_nom: float,
        p_nom_turbine: float,
        p_nom_pump: float = 0.0,
        eff_store: float = 1.0,
        eff_dispatch: float = 1.0,
        gen_efficiency: float = 0.9,
        initial_soc: float = 0.0,
        soc_min: float = None,
        soc_max: float = None,
        influx: float = 0.0,
    ):
        super().__init__(name, network, bus)

        # gen_efficiency couples storage outflow to electrical output and is
        # owned by this composite (not by the inner storage), so it stays a
        # plain attribute. Everything else is delegated to the inner storage.
        self.gen_efficiency = gen_efficiency

        # Build the inner storage as the single source of truth for shared
        # parameters. User mutations on this composite (e.g. ``phs.soc_min = 5``)
        # propagate via property setters below.
        self._storage = _StorageBase(
            name=f"{name}_storage",
            network=network,
            bus=bus,
            e_nom=e_nom,
            p_nom_in=p_nom_pump,        # p_in = pump
            p_nom_out=p_nom_turbine,    # p_out = water dispatch
            eff_in=eff_store,
            eff_out=eff_dispatch,
            initial_soc=initial_soc,
            soc_min=soc_min,
            soc_max=soc_max,
            influx=influx,
        )
        self._generator = Generator(
            name=f"{name}_generator",
            network=network,
            bus=bus,
            p_nom=p_nom_turbine,
        )

    # Property delegates: composite reads/writes flow into the inner storage
    # / generator. Mutations remain visible at create_model() time.
    @property
    def e_nom(self): return self._storage.e_nom
    @e_nom.setter
    def e_nom(self, v): self._storage.e_nom = v

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

    def setup_constraints(self, model):
        """Setup constraints for storage and generator coupling."""
        # Storage SOC constraints (reuses base class logic; influx already on
        # the inner _StorageBase via __init__).
        self._storage._setup_soc_constraints(model)

        # Setup generator constraints
        self._generator.setup_constraints(model)

        # Coupling constraint: electrical output = water dispatch * gen_efficiency
        setattr(
            model,
            f"{self.name}_coupling",
            self._generator.p == self._storage.p_out * self.gen_efficiency,
        )

    def get_p_net(self):
        """
        Return net power injection for KCL.

        For PHS, this is the generator output (electrical power to bus).
        """
        return self._generator.p

    # Property delegates for transparent access
    @property
    def p(self):
        """Electrical power output (from generator)."""
        return self._generator.p

    @property
    def p_store(self):
        """Pumping power (from storage.p_in)."""
        return self._storage.p_in

    @property
    def p_dispatch(self):
        """Water discharge rate (from storage.p_out)."""
        return self._storage.p_out

    @property
    def p_t(self) -> pl.DataFrame:
        """Solved electrical output (MW) per snapshot."""
        return _solution_as(self._generator.p, "p")

    @property
    def p_store_t(self) -> pl.DataFrame:
        """Solved pumping power (MW) per snapshot."""
        return _solution_as(self._storage.p_in, "p_store")

    @property
    def p_dispatch_t(self) -> pl.DataFrame:
        """Solved water discharge (MW equiv.) per snapshot."""
        return _solution_as(self._storage.p_out, "p_dispatch")

    def setup_objective(self, network):
        """Delegate objective to internal generator (which may have marginal costs)."""
        if self._generator:
            self._generator.setup_objective(network)

    def __repr__(self) -> str:
        return f"<PumpedHydroStorage(name={self.name}, bus={self.bus.name}, e_nom={self.e_nom}, p_nom_turbine={self.p_nom_turbine})>"


class Battery(StorageComposite):
    """
    Composite class for battery storage.

    A battery is a storage unit with direct electrical coupling (no separate generator).
    Net power = dispatch - store (positive = discharging to bus).

    Variable mapping:
    - storage.p_in -> charge (charging the battery)
    - storage.p_out -> discharge (discharging from battery)
    - p (net) -> discharge - charge (electrical power to/from bus)

    Parameters
    ----------
    name : str
        Unique identifier for the battery
    network : Network
        Reference to the parent network
    bus : Bus
        Bus where the battery is connected
    e_nom : float
        Nominal energy capacity (MWh)
    p_nom : float
        Maximum power (MW) for both charge and discharge
    eff_store : float, default 1.0
        Charging efficiency
    eff_dispatch : float, default 1.0
        Discharging efficiency
    initial_soc : float, default 0.0
        Initial state of charge
    soc_min : float, optional
        Minimum SOC limit
    soc_max : float, optional
        Maximum SOC limit (defaults to e_nom)
    """

    def __init__(
        self,
        name: str,
        network: "Network",
        bus: "Bus",
        e_nom: float,
        p_nom: float,
        eff_store: float = 1.0,
        eff_dispatch: float = 1.0,
        initial_soc: float = 0.0,
        soc_min: float = None,
        soc_max: float = None,
    ):
        super().__init__(name, network, bus)

        # Create the internal storage component eagerly with the supplied
        # parameters. From here on, all of (e_nom, p_nom, eff_*, initial_soc,
        # soc_min, soc_max) are owned by self._storage and surfaced via
        # properties on this composite. This means user mutations like
        # ``battery.soc_min = 20`` after construction are visible to the SOC
        # constraint at create_model() time, where they actually matter.
        self._storage = _StorageBase(
            name=name,
            network=network,
            bus=bus,
            e_nom=e_nom,
            p_nom_in=p_nom,   # p_in = charge
            p_nom_out=p_nom,  # p_out = discharge
            eff_in=eff_store,
            eff_out=eff_dispatch,
            initial_soc=initial_soc,
            soc_min=soc_min,
            soc_max=soc_max,
            influx=0.0,  # No influx for battery
        )

    # Property delegates so attribute mutations (and reads) on the composite
    # always go through the inner storage — no risk of stale duplicate state.
    @property
    def e_nom(self): return self._storage.e_nom
    @e_nom.setter
    def e_nom(self, v): self._storage.e_nom = v

    @property
    def p_nom(self): return self._storage.p_nom_in  # in == out for battery
    @p_nom.setter
    def p_nom(self, v):
        self._storage.p_nom_in = v
        self._storage.p_nom_out = v

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

    def setup_constraints(self, model):
        """Setup constraints for storage. Battery has no influx (set at __init__)."""
        self._storage._setup_soc_constraints(model)

    def setup_objective(self, network):
        """Delegate objective to internal storage (battery has no direct objective contribution)."""
        self._storage.setup_objective(network)

    def get_p_net(self):
        """
        Return net power injection for KCL.

        Net power = dispatch - store
        Positive = power injected into bus (discharging)
        Negative = power withdrawn from bus (charging)
        """
        return self._storage.p_out - self._storage.p_in

    # Property delegates for transparent access
    @property
    def p(self):
        """Net electrical power (dispatch - store)."""
        return self._storage.p_out - self._storage.p_in

    @property
    def p_store(self):
        """Charging power (from storage.p_in)."""
        return self._storage.p_in

    @property
    def p_dispatch(self):
        """Discharging power (from storage.p_out)."""
        return self._storage.p_out

    @property
    def p_t(self) -> pl.DataFrame:
        """Solved net power (MW): discharge − charge per snapshot."""
        out = self._storage.p_out_t  # cols: time, p_out
        ins = self._storage.p_in_t   # cols: time, p_in
        return out.join(ins, on=self.network.snapshots.name).with_columns(
            (pl.col("p_out") - pl.col("p_in")).alias("p")
        ).select([self.network.snapshots.name, "p"])

    @property
    def p_store_t(self) -> pl.DataFrame:
        """Solved charging power (MW) per snapshot."""
        return _solution_as(self._storage.p_in, "p_store")

    @property
    def p_dispatch_t(self) -> pl.DataFrame:
        """Solved discharging power (MW) per snapshot."""
        return _solution_as(self._storage.p_out, "p_dispatch")

    def __repr__(self) -> str:
        return f"<Battery(name={self.name}, bus={self.bus.name}, e_nom={self.e_nom}, p_nom={self.p_nom})>"
