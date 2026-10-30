"""
simulator.py
============
A small discrete-event simulation of ONE dark store during the evening rush.

Everything here is SYNTHETIC and built on placeholder assumptions (see StoreParams).
It is a thinking tool, not a model of any real company's operations.

How an order flows through the store
------------------------------------
1. Order arrives (Poisson arrivals, with a surge window, e.g. 8-11 PM).
2. It waits in a FIFO queue for a free picker, then is picked and packed.
3. It then waits for a free rider, who delivers it and rides back.
4. Delivery time = placed -> delivered.  A "breach" is delivery time > SLA (60 min).

Two sources of randomness are NOT visible to the model (pick-time noise and
traffic noise), so the breach label can never be predicted perfectly.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd

# The five features the risk model sees (all known at the moment an order is placed)
FEATURES = ["queue_per_picker", "items", "distance_km", "riders_free", "is_peak"]


@dataclass(frozen=True)
class StoreParams:
    """Every number here is an illustrative placeholder, not a real figure."""

    # --- time window (hours of the day) ---
    open_hour: float = 17.0
    close_hour: float = 24.0
    surge_start: float = 20.0
    surge_end: float = 23.0

    # --- demand ---
    base_orders_per_min: float = 0.48
    surge_multiplier: float = 2.2
    day_volatility: float = 0.20      # day-to-day demand swing (lognormal sigma)
    mean_items: float = 3.2           # items per basket (1 + Poisson)

    # --- store capacity (drawn per day, inclusive ranges) ---
    pickers_low: int = 3
    pickers_high: int = 6
    riders_low: int = 12
    riders_high: int = 17

    # --- service times (minutes) ---
    pick_base_min: float = 1.5
    pick_per_item_min: float = 0.55
    pick_noise_sigma: float = 0.30    # unobserved by the model
    min_per_km: float = 3.6           # ~16.7 km/h door to door
    travel_noise_sigma: float = 0.30  # unobserved by the model (traffic)
    handoff_min: float = 2.0
    return_factor: float = 0.9        # rider ride-back time as a share of outbound

    # --- geography ---
    outer_radius_km: float = 5.0
    inner_radius_km: float = 3.0      # radius used when the geofence is "tight"
    min_distance_km: float = 0.3
    distance_scale_km: float = 2.2    # demand decays with distance

    # --- promise ---
    sla_min: float = 60.0


def generate_day(p: StoreParams, seed: int, day_id: int) -> dict:
    """Pre-draw everything random about one evening, so that a baseline run and a
    policy run can be compared on the *same* orders (paired comparison)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, day_id]))
    day_mult = rng.lognormal(0.0, p.day_volatility)

    minutes = np.arange(int(p.open_hour * 60), int(p.close_hour * 60))
    hours = minutes / 60.0
    in_surge = (hours >= p.surge_start) & (hours < p.surge_end)
    rate = p.base_orders_per_min * np.where(in_surge, p.surge_multiplier, 1.0) * day_mult
    counts = rng.poisson(rate)
    t = np.repeat(minutes, counts) + rng.random(counts.sum())  # minutes since midnight
    n = len(t)

    items = 1 + rng.poisson(p.mean_items - 1.0, size=n)

    # Truncated-exponential distance: most demand is close to the store.
    span = p.outer_radius_km - p.min_distance_km
    u = rng.random(n)
    cdf_max = 1.0 - np.exp(-span / p.distance_scale_km)
    dist = p.min_distance_km - p.distance_scale_km * np.log(1.0 - u * cdf_max)

    return {
        "day": day_id,
        "t": t,
        "items": items,
        "dist": dist,
        "pick_noise": rng.lognormal(0.0, p.pick_noise_sigma, size=n),
        "travel_noise": rng.lognormal(0.0, p.travel_noise_sigma, size=n),
        "pickers": int(rng.integers(p.pickers_low, p.pickers_high + 1)),
        "riders": int(rng.integers(p.riders_low, p.riders_high + 1)),
    }


def run_day(day: dict, p: StoreParams, policy=None) -> pd.DataFrame:
    """Run one evening. `policy` (optional) can reject orders at the door.

    Returns one row per order, with columns:
      day, hour, <FEATURES>, accepted, total_time_min, breach, tight
    Rejected orders have accepted=False and NaN delivery time.
    """
    open_t = p.open_hour * 60.0
    k, r = day["pickers"], day["riders"]
    pick_free = [open_t] * k
    rider_free = [open_t] * r
    waiting_starts: list[float] = []  # pick-start times of accepted orders
    heapq.heapify(pick_free)
    heapq.heapify(rider_free)
    if policy is not None:
        policy.reset()

    rows = []
    for i in range(len(day["t"])):
        t = float(day["t"][i])
        items = int(day["items"][i])
        dist = float(day["dist"][i])

        # Observable state at the moment of ordering
        while waiting_starts and waiting_starts[0] <= t:
            heapq.heappop(waiting_starts)
        queue_len = len(waiting_starts)
        riders_free = sum(1 for f in rider_free if f <= t)
        hour = t / 60.0
        is_peak = int(p.surge_start <= hour < p.surge_end)
        qpp = queue_len / k
        feats = (qpp, items, dist, riders_free, is_peak)

        accept, tight = True, False
        if policy is not None:
            accept, tight = policy.decide(feats)

        if not accept:
            rows.append((day["day"], hour, *feats, False, np.nan, np.nan, tight))
            continue

        start = max(t, heapq.heappop(pick_free))
        pick_time = (p.pick_base_min + p.pick_per_item_min * items) * day["pick_noise"][i]
        end = start + pick_time
        heapq.heappush(pick_free, end)
        heapq.heappush(waiting_starts, start)

        rider_start = max(end, heapq.heappop(rider_free))
        travel = dist * p.min_per_km * day["travel_noise"][i]
        delivered = rider_start + travel + p.handoff_min
        heapq.heappush(rider_free, delivered + travel * p.return_factor)

        total = delivered - t
        rows.append((day["day"], hour, *feats, True, total, float(total > p.sla_min), tight))

    cols = ["day", "hour", *FEATURES, "accepted", "total_time_min", "breach", "tight"]
    return pd.DataFrame(rows, columns=cols)


def simulate_orders(p: StoreParams, n_days: int, seed: int = 42, first_day: int = 0) -> pd.DataFrame:
    """Baseline simulation (no policy): used as training data."""
    frames = [run_day(generate_day(p, seed, d), p) for d in range(first_day, first_day + n_days)]
    return pd.concat(frames, ignore_index=True)
