#!/usr/bin/env python3
"""
Earth Circulation Simulator — physical model and JSON export.

Equations (spec):
  f = coriolis_scale * 2Ω sin(latitude),  Ω = 7.2921e-5 s⁻¹
  T(lat) = cos(latitude)
  ∂P/∂y = -k ∂T/∂y

  Wind:  du/dt = -f v - r_w u,   dv/dt = -∂P/∂y + f u - r_w v
  τx = α u_wind,  τy = α v_wind
  Ocean: du_o/dt = -f v_o + τx - r_o u_o + ν∇²u_o
         dv_o/dt = f u_o + τy - r_o v_o + ν∇²v_o

Grid: 60 lon × 30 lat, lat -90..90, lon 0..360.
"""

import json
import math
import numpy as np
from pathlib import Path

# Grid (spec)
NLON = 60
NLAT = 30
LAT_MIN = -90.0
LAT_MAX = 90.0
LON_MIN = 0.0
LON_MAX = 360.0
OMEGA = 7.2921e-5
DT = 0.1
STEPS = 400

# Physics scaling (match app.js for consistency)
K_PRESSURE = 0.8
R_WIND = 0.15
ALPHA_STRESS = 0.12
R_OCEAN_BASE = 0.08
NU = 0.02

DLON = (LON_MAX - LON_MIN) / NLON
DLAT = (LAT_MAX - LAT_MIN) / NLAT


def lon_index_to_lon(i):
    return LON_MIN + (i + 0.5) * DLON


def lat_index_to_lat(j):
    return LAT_MIN + (j + 0.5) * DLAT


def is_land(i, j):
    lon = lon_index_to_lon(i)
    lat = lat_index_to_lat(j)
    if lat <= -60:
        return True
    if 230 <= lon <= 330 and -55 <= lat <= 75:
        return True
    if lon <= 150 and -35 <= lat <= 75:
        return True
    if lon >= 340 and -35 <= lat <= 75:
        return True
    return False


def build_land_mask():
    mask = np.zeros((NLAT, NLON), dtype=float)
    for j in range(NLAT):
        for i in range(NLON):
            if is_land(i, j):
                mask[j, i] = 1.0
    return mask


def coriolis_at(j, coriolis_scale):
    lat_rad = math.radians(lat_index_to_lat(j))
    return coriolis_scale * 2 * OMEGA * math.sin(lat_rad)


def pressure_gradient_meridional(j):
    lat_rad = math.radians(lat_index_to_lat(j))
    dT_dlat = -math.sin(lat_rad)
    return -K_PRESSURE * dT_dlat * (math.pi / 180.0)


def laplacian_u(u):
    """Periodic in x (lon), reflective at y (lat) boundaries."""
    u_pad = np.pad(u, ((1, 1), (1, 1)), mode="wrap")
    u_pad[:, 0] = u_pad[:, 1]
    u_pad[:, -1] = u_pad[:, -2]
    u_pad[0, 1:-1] = u[0]
    u_pad[-1, 1:-1] = u[-1]
    lap = u_pad[:-2, 1:-1] + u_pad[2:, 1:-1] + u_pad[1:-1, :-2] + u_pad[1:-1, 2:] - 4 * u
    return lap


def laplacian_v(v):
    return laplacian_u(v)


def step(u_wind, v_wind, u_ocean, v_ocean, land_mask, coriolis_scale, wind_forcing, ocean_friction):
    r_o = R_OCEAN_BASE * ocean_friction

    for j in range(NLAT):
        f = coriolis_at(j, coriolis_scale)
        dPdy = wind_forcing * pressure_gradient_meridional(j)
        for i in range(NLON):
            if land_mask[j, i] > 0.5:
                continue
            u, v = u_wind[j, i], v_wind[j, i]
            du = -f * v - R_WIND * u
            dv = dPdy + f * u - R_WIND * v
            u_wind[j, i] += DT * du
            v_wind[j, i] += DT * dv

    lap_u = laplacian_u(u_ocean)
    lap_v = laplacian_v(v_ocean)

    for j in range(NLAT):
        f = coriolis_at(j, coriolis_scale)
        for i in range(NLON):
            if land_mask[j, i] > 0.5:
                continue
            tau_x = ALPHA_STRESS * u_wind[j, i]
            tau_y = ALPHA_STRESS * v_wind[j, i]
            uo, vo = u_ocean[j, i], v_ocean[j, i]
            du = -f * vo + tau_x - r_o * uo + NU * lap_u[j, i]
            dv = f * uo + tau_y - r_o * vo + NU * lap_v[j, i]
            u_ocean[j, i] += DT * du
            v_ocean[j, i] += DT * dv

    u_wind[land_mask > 0.5] = 0
    v_wind[land_mask > 0.5] = 0
    u_ocean[land_mask > 0.5] = 0
    v_ocean[land_mask > 0.5] = 0

    return u_wind, v_wind, u_ocean, v_ocean


def run_simulation(coriolis_scale=1.0, wind_forcing=1.0, ocean_friction=1.0):
    land_mask = build_land_mask()
    u_wind = np.zeros((NLAT, NLON))
    v_wind = np.zeros((NLAT, NLON))
    u_ocean = np.zeros((NLAT, NLON))
    v_ocean = np.zeros((NLAT, NLON))

    for _ in range(STEPS):
        u_wind, v_wind, u_ocean, v_ocean = step(
            u_wind.copy(), v_wind.copy(), u_ocean.copy(), v_ocean.copy(),
            land_mask, coriolis_scale, wind_forcing, ocean_friction
        )

    lat = [lat_index_to_lat(j) for j in range(NLAT)]
    lon = [lon_index_to_lon(i) for i in range(NLON)]

    return {
        "lat": lat,
        "lon": lon,
        "u_wind": u_wind.tolist(),
        "v_wind": v_wind.tolist(),
        "u_ocean": u_ocean.tolist(),
        "v_ocean": v_ocean.tolist(),
    }


def export_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    data = run_simulation(coriolis_scale=1.0, wind_forcing=1.0, ocean_friction=1.0)
    out_path = Path(__file__).resolve().parent.parent / "data" / "simulation_output.json"
    export_json(data, out_path)
    print(f"Exported to {out_path}")
