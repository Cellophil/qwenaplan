# Showcase: build a small 3-bus system, solve a 24 h dispatch, plot the result.
#
# Run as a script (`./conda/bin/python showcase.py`) or step through the
# `#%%` cells interactively in VS Code.

#%% Setup

import polars as pl
import plotly.graph_objects as go
import pyoptinterface as poi

import qwenaplan as qp


#%% Topology
#
#         Bus1 ── ACLine_main ──▶ Bus2 ── ACLine_skinny ──▶ Bus3
#          │      (s_nom 80)      │       (s_nom 30) ◀── this binds
# [Coal cheap]                    │
#                            [Solar, Battery, Load_urban]
#          │                                                 │
#          └──────────── Link_dc ────────────────────────────┘
#                        (p_nom 25)
#                                                          [Peaker, Load_rural]
#
# Bus1 has a cheap, always-on coal plant. Bus2 is the urban hub (variable
# load, solar, battery storage). Bus3 is a small remote node fed by the
# skinny B2–B3 line plus a DC link straight from B1; both inter-zonal
# corridors are deliberately undersized so the LP has to ration cheap
# energy and prices diverge across zones.

n = qp.Network()

bus1 = n.add(qp.Bus, "Bus1", v_nom=380.0, x=0.0, y=0.0)
bus2 = n.add(qp.Bus, "Bus2", v_nom=380.0, x=1.0, y=0.0)
bus3 = n.add(qp.Bus, "Bus3", v_nom=380.0, x=2.0, y=-0.5)


#%% Time axis and profiles

snapshots = pl.Series("hour", list(range(24)))

# Solar follows a clipped sinusoid peaking at noon.
solar_profile = pl.Series(
    "hour",
    [max(0.0, -((h - 12) / 6) ** 2 + 1.0) for h in range(24)],
)

# Urban load: morning + evening peaks, midday plateau.
urban_load = pl.Series(
    "hour",
    [25, 22, 20, 20, 22, 28, 38, 48, 52, 50, 48, 47,
     46, 45, 44, 45, 50, 60, 65, 60, 52, 45, 38, 30],
)

# Rural load: small, flatter, slight evening peak.
rural_load = pl.Series(
    "hour",
    [12, 11, 10, 10, 11, 13, 16, 18, 18, 17, 16, 15,
     15, 15, 15, 16, 18, 22, 24, 23, 20, 17, 14, 13],
)


#%% Components

n.add(qp.Generator, "Coal", bus=bus1,
      p_nom=120.0, marginal_cost=25.0, carrier="coal",
      ramp_limit_up=0.2, ramp_limit_down=0.2)

n.add(qp.Generator, "Solar", bus=bus2,
      p_nom=70.0, marginal_cost=0.0, carrier="solar",
      p_max_pu=solar_profile)

n.add(qp.Generator, "Peaker", bus=bus3,
      p_nom=40.0, marginal_cost=180.0, carrier="gas")

battery = n.add(qp.Battery, "Battery", bus=bus2,
                e_nom=120.0, p_nom=30.0,
                eff_store=0.93, eff_dispatch=0.93, initial_soc=30.0)

n.add(qp.Load, "Load_urban", bus=bus2, p_set=urban_load)
n.add(qp.Load, "Load_rural", bus=bus3, p_set=rural_load)

line_main = n.add(qp.ACLine, "ACLine_main", from_bus=bus1, to_bus=bus2,
                  x_pu=0.05, s_nom=80.0)
line_skinny = n.add(qp.ACLine, "ACLine_skinny", from_bus=bus2, to_bus=bus3,
                    x_pu=0.10, s_nom=30.0)
link_dc = n.add(qp.Link, "Link_dc", from_bus=bus1, to_bus=bus3, p_nom=25.0)


#%% Solve

n.set_snapshots(snapshots, duration=1.0, weighting=1.0)
n.create_model()

status = n.optimize()
assert status == poi.TerminationStatusCode.OPTIMAL
print(f"objective = {n.objective_value:,.0f} €  ({status})")


#%% Pull results into one tidy frame
#
# Every component exposes `.sol.<name>_t` as a polars DataFrame keyed by
# the snapshot dim. Joining on "hour" gives one long-form frame ready
# for plotly.

H = snapshots.name  # "hour"

dispatch = (
    n.generators["Coal"].sol.p_t.rename({"p": "Coal"})
    .join(n.generators["Solar"].sol.p_t.rename({"p": "Solar"}), on=H)
    .join(n.generators["Peaker"].sol.p_t.rename({"p": "Peaker"}), on=H)
    .join(battery.sol.p_t.rename({"p": "Battery_net"}), on=H)
)

total_load = (
    n.loads["Load_urban"].sol.p_t.rename({"p": "urban"})
    .join(n.loads["Load_rural"].sol.p_t.rename({"p": "rural"}), on=H)
    .with_columns((pl.col("urban") + pl.col("rural")).alias("total"))
)

print(dispatch.head())


#%% Plot 1 — dispatch stack vs total load

# Stacked area for supply, line on top for demand. Battery splits into
# charge (negative) and discharge (positive) so the stack is honest.
batt_charge = dispatch["Battery_net"].clip(upper_bound=0.0)
batt_discharge = dispatch["Battery_net"].clip(lower_bound=0.0)

fig1 = go.Figure()
for name, color in [("Coal", "#5a5a5a"), ("Solar", "#f4c430"),
                    ("Peaker", "#b03030")]:
    fig1.add_trace(go.Scatter(
        x=dispatch[H], y=dispatch[name],
        name=name, stackgroup="supply", line=dict(width=0),
        fillcolor=color,
    ))
fig1.add_trace(go.Scatter(
    x=dispatch[H], y=batt_discharge,
    name="Battery discharge", stackgroup="supply", line=dict(width=0),
    fillcolor="#3aa0c0",
))
fig1.add_trace(go.Scatter(
    x=dispatch[H], y=batt_charge,
    name="Battery charge", stackgroup="charge", line=dict(width=0),
    fillcolor="rgba(58,160,192,0.4)",
))
fig1.add_trace(go.Scatter(
    x=total_load[H], y=total_load["total"],
    name="Total load", line=dict(color="black", width=2, dash="dash"),
))
fig1.update_layout(
    title="Dispatch by carrier vs. total demand",
    xaxis_title="hour", yaxis_title="MW",
    template="plotly_white",
)
fig1.show()


#%% Plot 2 — battery state of charge and per-unit fill level

soc = battery.sol.soc_t
soc_pu = battery.sol.soc_pu_t  # = soc / e_nom

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=soc[H], y=soc["soc"], name="SOC (MWh)", line=dict(color="#3aa0c0"),
))
fig2.add_trace(go.Scatter(
    x=soc_pu[H], y=soc_pu["soc_pu"], name="SOC fill (p.u.)",
    yaxis="y2", line=dict(color="#3aa0c0", dash="dot"),
))
fig2.update_layout(
    title="Battery — SOC trajectory",
    xaxis_title="hour",
    yaxis=dict(title="MWh"),
    yaxis2=dict(title="fill (0–1)", overlaying="y", side="right",
                range=[0, 1]),
    template="plotly_white",
)
fig2.show()


#%% Plot 3 — corridor flows and the line that binds
#
# Both inter-zonal paths (skinny AC line, DC link) are undersized. We
# plot their flows alongside ±s_nom / ±p_nom envelopes; the dual on the
# upper / lower limit constraint quantifies the *congestion rent* per MW.

flow_skinny = line_skinny.sol.p_t["p"]
flow_link = link_dc.sol.p_t["p"]

# pyoframe exposes constraint duals as polars frames; we registered the
# line's thermal limits as `line_limit_upper_<name>` / `..._lower_<name>`.
dual_upper = n.model.line_limit_upper_ACLine_skinny.dual["dual"]
dual_lower = n.model.line_limit_lower_ACLine_skinny.dual["dual"]
# Congestion rent = the higher-magnitude side per snapshot. Sign convention
# of the solver puts the binding side as a (small) negative value.
rent = pl.Series([max(abs(u), abs(l)) for u, l in zip(dual_upper, dual_lower)])

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=snapshots, y=flow_skinny,
                          name="ACLine skinny (B2→B3)",
                          line=dict(color="#b03030")))
fig3.add_trace(go.Scatter(x=snapshots, y=flow_link,
                          name="Link DC (B1→B3)",
                          line=dict(color="#3070b0")))
for s_nom, color in [(line_skinny.s_nom, "#b03030"),
                     (link_dc.p_nom, "#3070b0")]:
    fig3.add_hline(y=s_nom, line_dash="dot", line_color=color, opacity=0.5)
    fig3.add_hline(y=-s_nom, line_dash="dot", line_color=color, opacity=0.5)
fig3.add_trace(go.Bar(x=snapshots, y=rent, name="Congestion rent (€/MWh)",
                      yaxis="y2", marker_color="#aaaaaa", opacity=0.5))
fig3.update_layout(
    title="Corridor flows and congestion rent on the skinny AC line",
    xaxis_title="hour",
    yaxis=dict(title="flow (MW)"),
    yaxis2=dict(title="rent (€/MWh)", overlaying="y", side="right"),
    template="plotly_white",
)
fig3.show()


#%% Plot 4 — nodal prices (KCL duals = LMPs)

lmps = pl.DataFrame({
    H: snapshots,
    "Bus1": n.model.kcl_Bus1.dual["dual"],
    "Bus2": n.model.kcl_Bus2.dual["dual"],
    "Bus3": n.model.kcl_Bus3.dual["dual"],
})

fig4 = go.Figure()
for bus, color in [("Bus1", "#5a5a5a"), ("Bus2", "#f4c430"),
                   ("Bus3", "#b03030")]:
    fig4.add_trace(go.Scatter(x=lmps[H], y=lmps[bus], name=bus,
                              line=dict(color=color)))
fig4.update_layout(
    title="Locational marginal prices (€/MWh)",
    xaxis_title="hour", yaxis_title="LMP (€/MWh)",
    template="plotly_white",
)
fig4.show()


#%% Plot 5 — capacity factors via the `_pu_t` views

cf = (
    n.generators["Coal"].sol.p_pu_t.rename({"p_pu": "Coal"})
    .join(n.generators["Solar"].sol.p_pu_t.rename({"p_pu": "Solar"}), on=H)
    .join(n.generators["Peaker"].sol.p_pu_t.rename({"p_pu": "Peaker"}), on=H)
)

fig5 = go.Figure()
for col, color in [("Coal", "#5a5a5a"), ("Solar", "#f4c430"),
                   ("Peaker", "#b03030")]:
    fig5.add_trace(go.Scatter(x=cf[H], y=cf[col], name=col,
                              line=dict(color=color)))
fig5.update_layout(
    title="Per-unit dispatch (p / p_nom)",
    xaxis_title="hour", yaxis_title="capacity factor",
    yaxis_range=[0, 1.05], template="plotly_white",
)
fig5.show()


#%% Headline numbers

cong_hours = sum(1 for r in rent if r > 1e-6)
print(f"hours where ACLine_skinny is congested: {cong_hours} / 24")
print(f"average LMP spread Bus3 - Bus1: "
      f"{(lmps['Bus3'] - lmps['Bus1']).mean():.1f} €/MWh")
print(f"battery cycled energy: {battery.sol.p_dispatch_t['p_dispatch'].sum():.1f} MWh")
