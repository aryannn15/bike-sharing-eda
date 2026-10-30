"""Shared helpers for the Bike Sharing EDA notebooks.

Keeps paths, plot styling and data loading in one place so every notebook
behaves the same way.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

# ---------------------------------------------------------------- palette
# Colorblind-checked categorical order (blue, orange, aqua, yellow, magenta,
# green, violet, red). Always assigned in this fixed order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_CMAP = "Blues"      # magnitude -> one hue, light to dark
DIV_CMAP = "RdBu_r"     # polarity (e.g. correlation) -> two hues, neutral middle

SEASONS = ["Spring", "Summer", "Fall", "Winter"]
WEATHER = ["Clear", "Mist/Cloudy", "Rain/Snow"]
WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


def set_style():
    """Apply a clean, consistent matplotlib style."""
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "figure.dpi": 100,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.prop_cycle": cycler(color=PALETTE),
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "lines.linewidth": 2,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })


def save_fig(fig, name):
    """Save a figure to reports/figures/<name>.png."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=120)


def load_raw_hourly():
    return pd.read_csv(RAW / "hour.csv", parse_dates=["dteday"])


def load_clean_hourly():
    """Cleaned hourly data (output of notebook 01) with a DatetimeIndex."""
    df = pd.read_csv(PROCESSED / "hour_clean.csv", parse_dates=["datetime"])
    return df.set_index("datetime")


def load_clean_daily():
    """Cleaned daily data (output of notebook 01) with a DatetimeIndex."""
    df = pd.read_csv(PROCESSED / "day_clean.csv", parse_dates=["date"])
    return df.set_index("date")
