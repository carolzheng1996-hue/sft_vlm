from __future__ import annotations

import math
from typing import Any

import numpy as np


GENERATOR_VERSION = "v4"
DEFAULT_COMPLEXITY_MIX = {"controlled": 0.20, "compositional": 0.60, "confounded": 0.20}
COMPLEXITY_LEVELS = tuple(DEFAULT_COMPLEXITY_MIX)


def validate_complexity_mix(mix: dict[str, float]) -> dict[str, float]:
    normalized = {str(key): float(value) for key, value in mix.items()}
    if set(normalized) != set(COMPLEXITY_LEVELS):
        raise ValueError(f"complexity_mix must contain exactly {list(COMPLEXITY_LEVELS)}")
    if any(value < 0 for value in normalized.values()) or abs(sum(normalized.values()) - 1.0) > 1e-6:
        raise ValueError("complexity_mix values must be non-negative and sum to 1")
    return normalized


def complexity_schedule(count: int, mix: dict[str, float], seed: int) -> list[str]:
    """Return an exact, deterministic largest-remainder allocation."""
    mix = validate_complexity_mix(mix)
    raw = {name: count * value for name, value in mix.items()}
    quotas = {name: int(value) for name, value in raw.items()}
    remainder = count - sum(quotas.values())
    order = sorted(COMPLEXITY_LEVELS, key=lambda name: (raw[name] - quotas[name], name), reverse=True)
    for name in order[:remainder]:
        quotas[name] += 1
    labels = [name for name in COMPLEXITY_LEVELS for _ in range(quotas[name])]
    rng = np.random.default_rng(seed + 91_337)
    rng.shuffle(labels)
    return labels


def candidate_periods(n: int) -> list[float]:
    base = [5, 7, 12, 18, 24, 30, 48, 72, 96, 120, 168, 240, 365]
    valid = [float(period) for period in base if 4 <= period <= max(4, n / 3)]
    return valid or [max(4.0, float(n) / 4.0)]


def _sample_trend_spec(rng: np.random.Generator, n: int, complexity: str, allow_trend: bool) -> dict[str, Any]:
    if not allow_trend:
        return {"type": "none"}
    choices = {
        "controlled": ["linear"],
        "compositional": ["linear", "quadratic", "saturating", "piecewise"],
        "confounded": ["quadratic", "saturating", "piecewise", "local_reversal", "random_walk"],
    }[complexity]
    kind = str(rng.choice(choices))
    slope = float(rng.uniform(-0.035, 0.055))
    if kind == "linear":
        return {"type": kind, "slope": slope}
    if kind == "quadratic":
        return {"type": kind, "slope": slope, "curvature": float(rng.uniform(-8e-5, 8e-5))}
    if kind == "saturating":
        return {"type": kind, "amplitude": float(rng.uniform(-18, 22)), "midpoint": float(rng.uniform(0.30, 0.70)), "width": float(rng.uniform(0.06, 0.18))}
    if kind == "piecewise":
        return {"type": kind, "first_slope": slope, "second_slope": float(rng.uniform(-0.055, 0.055)), "break_fraction": float(rng.uniform(0.35, 0.70))}
    if kind == "local_reversal":
        return {"type": kind, "slope": slope, "amplitude": float(rng.uniform(-15, 15)), "center": float(rng.uniform(0.35, 0.75)), "width": float(rng.uniform(0.05, 0.14))}
    return {
        "type": kind,
        "step_std": float(rng.uniform(0.03, 0.16)),
        "smooth_window": int(rng.integers(7, max(8, min(49, n // 8)))),
        "innovation_seed": int(rng.integers(0, 2**31 - 1)),
    }


def _sample_seasonal_specs(rng: np.random.Generator, n: int, complexity: str, count: int | None) -> list[dict[str, Any]]:
    if count == 0:
        return []
    if count is None:
        count = 1 if complexity == "controlled" else int(rng.integers(1, 3 if complexity == "compositional" else 4))
    periods = candidate_periods(n)
    replace = count > len(periods)
    chosen = list(rng.choice(periods, size=count, replace=replace))
    specs: list[dict[str, Any]] = []
    for index, raw_period in enumerate(chosen):
        period = float(raw_period)
        if complexity == "confounded" and index > 0:
            period *= float(rng.uniform(0.92, 1.09))
        waveform = "sine" if complexity == "controlled" else str(rng.choice(["sine", "triangle", "soft_square"]))
        spec: dict[str, Any] = {
            "period": round(period, 6),
            "amplitude": float(rng.uniform(3.0, 10.0) / math.sqrt(index + 1)),
            "waveform": waveform,
            "phase": float(rng.uniform(0, 2 * np.pi)),
            "amplitude_modulation": 0.0,
            "period_drift": 0.0,
            "regime_fraction": None,
            "regime_multiplier": 1.0,
        }
        if complexity in {"compositional", "confounded"} and rng.random() < (0.45 if complexity == "compositional" else 0.75):
            spec["amplitude_modulation"] = float(rng.uniform(0.15, 0.55))
            spec["modulation_period"] = float(rng.choice(candidate_periods(max(n, 120))))
        if complexity == "confounded" and rng.random() < 0.65:
            spec["period_drift"] = float(rng.uniform(-0.22, 0.28))
        if complexity in {"compositional", "confounded"} and rng.random() < 0.35:
            spec["regime_fraction"] = float(rng.uniform(0.40, 0.72))
            spec["regime_multiplier"] = float(rng.uniform(0.35, 1.75))
        specs.append(spec)
    return specs


def _sample_noise_spec(rng: np.random.Generator, complexity: str, forced: str | None = None) -> dict[str, Any]:
    if forced:
        kind = forced
    else:
        choices = {
            "controlled": ["gaussian"],
            "compositional": ["gaussian", "student_t", "ar1", "heteroscedastic"],
            "confounded": ["student_t", "ar1", "heteroscedastic", "level_dependent", "mixture"],
        }[complexity]
        kind = str(rng.choice(choices))
    spec: dict[str, Any] = {"type": kind, "scale": float(rng.uniform(0.8, 2.5))}
    if kind == "student_t":
        spec["degrees_of_freedom"] = float(rng.uniform(3.2, 7.5))
    elif kind == "ar1":
        spec["phi"] = float(rng.uniform(0.35, 0.88))
    elif kind == "heteroscedastic":
        mode = "local_window" if complexity == "confounded" or rng.random() < 0.45 else "ramp"
        spec["mode"] = mode
        if mode == "local_window":
            start = float(rng.uniform(0.18, 0.62))
            spec.update({"start_fraction": start, "end_fraction": min(.92, start + float(rng.uniform(.12, .30))), "inside_multiplier": float(rng.uniform(2.5, 5.5))})
        else:
            spec["end_multiplier"] = float(rng.uniform(2.0, 5.0))
    elif kind == "level_dependent":
        spec["level_strength"] = float(rng.uniform(0.6, 1.8))
    elif kind == "mixture":
        spec["spike_probability"] = float(rng.uniform(0.008, 0.035))
        spec["spike_multiplier"] = float(rng.uniform(4.0, 9.0))
    return spec


def sample_signal_spec(
    rng: np.random.Generator,
    n: int,
    complexity: str,
    *,
    allow_trend: bool = True,
    seasonal_count: int | None = None,
    forced_noise: str | None = None,
) -> dict[str, Any]:
    if complexity not in COMPLEXITY_LEVELS:
        raise ValueError(f"Unknown signal complexity: {complexity}")
    return {
        "level": float(rng.uniform(30.0, 80.0)),
        "trend": _sample_trend_spec(rng, n, complexity, allow_trend),
        "seasonality": _sample_seasonal_specs(rng, n, complexity, seasonal_count),
        "noise": _sample_noise_spec(rng, complexity, forced_noise),
    }


def render_trend(n: int, spec: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=float)
    u = t / max(1, n - 1)
    kind = spec["type"]
    if kind == "none":
        return np.zeros(n)
    if kind == "linear":
        return float(spec["slope"]) * t
    if kind == "quadratic":
        return float(spec["slope"]) * t + float(spec["curvature"]) * t**2
    if kind == "saturating":
        return float(spec["amplitude"]) * np.tanh((u - float(spec["midpoint"])) / float(spec["width"]))
    if kind == "piecewise":
        point = int(n * float(spec["break_fraction"]))
        before = float(spec["first_slope"]) * np.minimum(t, point)
        after = float(spec["second_slope"]) * np.maximum(0, t - point)
        return before + after
    if kind == "local_reversal":
        bump = float(spec["amplitude"]) * np.exp(-0.5 * ((u - float(spec["center"])) / float(spec["width"])) ** 2)
        return float(spec["slope"]) * t + bump
    # The realization seed is part of the hidden component specification so the
    # component can be reproduced independently of surrounding RNG consumption.
    innovation_rng = np.random.default_rng(int(spec["innovation_seed"]))
    raw = np.cumsum(innovation_rng.normal(0, float(spec["step_std"]), n))
    window = max(3, int(spec["smooth_window"]))
    kernel = np.ones(window) / window
    return np.convolve(raw, kernel, mode="same")


def _waveform(phase: np.ndarray, kind: str) -> np.ndarray:
    sine = np.sin(phase)
    if kind == "triangle":
        return 2.0 / np.pi * np.arcsin(sine)
    if kind == "soft_square":
        return np.tanh(2.7 * sine)
    return sine


def render_seasonality(n: int, specs: list[dict[str, Any]], phase_offset: float = 0.0) -> np.ndarray:
    t = np.arange(n, dtype=float)
    u = t / max(1, n - 1)
    result = np.zeros(n)
    for spec in specs:
        base_period = float(spec["period"])
        drift = float(spec.get("period_drift", 0.0))
        instantaneous_period = np.maximum(2.0, base_period * (1.0 + drift * (u - 0.5)))
        phase = np.cumsum(2 * np.pi / instantaneous_period) + float(spec["phase"]) + phase_offset
        amplitude = np.full(n, float(spec["amplitude"]))
        modulation = float(spec.get("amplitude_modulation", 0.0))
        if modulation:
            amplitude *= 1.0 + modulation * np.sin(2 * np.pi * t / float(spec["modulation_period"]))
        regime_fraction = spec.get("regime_fraction")
        if regime_fraction is not None:
            amplitude[int(n * float(regime_fraction)):] *= float(spec["regime_multiplier"])
        result += amplitude * _waveform(phase, str(spec["waveform"]))
    return result


def render_noise(rng: np.random.Generator, clean: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    n = len(clean)
    scale = float(spec["scale"])
    kind = spec["type"]
    if kind == "student_t":
        df = float(spec["degrees_of_freedom"])
        innovations = rng.standard_t(df, n) * scale / math.sqrt(df / (df - 2))
    else:
        innovations = rng.normal(0, scale, n)
    if kind == "ar1":
        phi = float(spec["phi"])
        noise = np.zeros(n)
        for index in range(1, n):
            noise[index] = phi * noise[index - 1] + innovations[index]
        return noise
    if kind == "heteroscedastic":
        if spec.get("mode", "ramp") == "local_window":
            multiplier = np.ones(n)
            start = int(n * float(spec["start_fraction"])); end = max(start + 1, int(n * float(spec["end_fraction"])))
            multiplier[start:end] = float(spec["inside_multiplier"])
        else:
            multiplier = np.linspace(0.65, float(spec["end_multiplier"]), n)
        return innovations * multiplier
    if kind == "level_dependent":
        centered = np.abs(clean - np.nanmedian(clean))
        normalized = centered / (np.nanpercentile(centered, 90) + 1e-8)
        return innovations * (0.65 + float(spec["level_strength"]) * normalized)
    if kind == "mixture":
        mask = rng.random(n) < float(spec["spike_probability"])
        innovations[mask] += rng.normal(0, scale * float(spec["spike_multiplier"]), int(mask.sum()))
    return innovations


def render_signal(
    rng: np.random.Generator,
    n: int,
    spec: dict[str, Any],
    *,
    scale: float = 1.0,
    offset: float = 0.0,
    phase_offset: float = 0.0,
    idiosyncratic_scale: float = 1.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    trend = render_trend(n, spec["trend"], rng)
    seasonal = render_seasonality(n, spec["seasonality"], phase_offset)
    clean = float(spec["level"]) + trend + seasonal
    noise = render_noise(rng, clean, spec["noise"]) * idiosyncratic_scale
    observed = offset + scale * (clean + noise)
    return observed, {"trend": scale * trend, "seasonal": scale * seasonal, "noise": scale * noise, "clean": offset + scale * clean}


def legacy_trend_slope(n: int, trend_spec: dict[str, Any], seed: int) -> float:
    trend = render_trend(n, trend_spec, np.random.default_rng(seed + 19_991))
    return float((trend[-1] - trend[0]) / max(1, n - 1))
