import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import joblib, os
from sklearn.metrics import r2_score, mean_squared_error

DROP   = ["city", "timestamp", "aqi_category", "weather", "aqi"]
TARGET = "aqi"

COLORS = {
    "beijing":  "#E05C5C",
    "delhi":    "#F0A050",
    "london":   "#5BAD8F",
    "new york": "#5B8FD4",
    "sydney":   "#A87DC8",
}

BG      = "#12141C"
PANEL   = "#1C1F2E"
GRID    = "#2A2D3E"
TEXT    = "#E0E0E0"
SUBTEXT = "#8A8FA8"

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=SUBTEXT, labelsize=8)
    ax.spines[:].set_color(GRID)
    ax.spines[:].set_linewidth(0.6)
    ax.set_xlabel(xlabel, color=SUBTEXT, fontsize=8, labelpad=6)
    ax.set_ylabel(ylabel, color=SUBTEXT, fontsize=8, labelpad=6)
    ax.set_title(title,   color=TEXT,    fontsize=10, fontweight="bold", pad=10)
    ax.grid(color=GRID, linewidth=0.4, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

def load():
    df    = pd.read_parquet("data/features/features.parquet")
    model = joblib.load("models/xgboost_aqi.pkl")
    cols  = [c for c in df.columns if c not in DROP]
    df    = df[cols + [TARGET, "city", "timestamp"]].dropna()
    X     = df[cols]
    df["predicted"] = model.predict(X)
    df["hour"]      = pd.to_datetime(df["timestamp"]).dt.hour
    return df, model, cols

def plot(df, model, cols):
    os.makedirs("outputs", exist_ok=True)
    cities = list(COLORS.keys())

    fig = plt.figure(figsize=(20, 13), facecolor=BG)
    gs  = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.52, wspace=0.35,
        left=0.06, right=0.97, top=0.91, bottom=0.07
    )

    # ── Header metrics ──────────────────────────────────────────────
    r2   = r2_score(df["aqi"], df["predicted"])
    rmse = np.sqrt(mean_squared_error(df["aqi"], df["predicted"]))
    mape = np.mean(np.abs((df["aqi"] - df["predicted"]) / (df["aqi"] + 1e-8))) * 100

    fig.text(0.5, 0.965, "Air Quality Monitoring & Forecasting",
             ha="center", color=TEXT, fontsize=17, fontweight="bold")
    fig.text(0.5, 0.942,
             f"XGBoost Model  ·  R² = {r2:.4f}  ·  RMSE = {rmse:.2f}  ·  MAPE = {mape:.2f}%  ·  5 cities  ·  60 days",
             ha="center", color=SUBTEXT, fontsize=9)

    # ── 1. Actual vs Predicted — last 72h (full width) ───────────────
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1, "Actual vs Predicted AQI — last 72 hours", "Hours", "AQI")
    for city in cities:
        sub  = df[df["city"] == city].tail(72).reset_index(drop=True)
        col  = COLORS[city]
        ax1.plot(sub.index, sub["aqi"],       color=col, lw=2,   alpha=0.95)
        ax1.plot(sub.index, sub["predicted"], color=col, lw=1.2, alpha=0.5, linestyle="--")
    solid  = [Line2D([0],[0], color=COLORS[c], lw=2,   label=f"{c}")          for c in cities]
    dashed = [Line2D([0],[0], color="white",   lw=1.2, linestyle="--", label="predicted", alpha=0.5)]
    ax1.legend(handles=solid+dashed, fontsize=7.5, ncol=6,
               facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
               loc="upper right", framealpha=0.9)
    ax1.set_xlim(0, 71)

    # ── 2. Scatter: actual vs predicted ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2, "Predicted vs Actual", "Actual AQI", "Predicted AQI")
    for city in cities:
        sub = df[df["city"] == city]
        ax2.scatter(sub["aqi"], sub["predicted"],
                    alpha=0.12, s=5, color=COLORS[city], rasterized=True)
    lo, hi = df["aqi"].min(), df["aqi"].max()
    ax2.plot([lo, hi], [lo, hi], color=SUBTEXT, lw=1, linestyle="--", alpha=0.7)
    ax2.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax2.transAxes,
             color=TEXT, fontsize=8.5, fontweight="bold")

    # ── 3. Feature importance ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3, "Top 10 Feature Importances", "Importance Score", "")
    imp  = pd.Series(model.feature_importances_, index=cols).nlargest(10)
    ypos = range(len(imp))
    bars = ax3.barh(list(ypos), imp.values[::-1],
                    color="#5B8FD4", edgecolor="none", height=0.6)
    ax3.set_yticks(list(ypos))
    ax3.set_yticklabels([f.replace("aqi_","").replace("_"," ") for f in imp.index[::-1]],
                        fontsize=7.5, color=TEXT)
    for bar, val in zip(bars, imp.values[::-1]):
        ax3.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", color=SUBTEXT, fontsize=7)
    ax3.grid(axis="y", alpha=0)

    # ── 4. AQI by hour of day ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    style_ax(ax4, "Mean AQI by Hour of Day", "Hour (UTC)", "Mean AQI")
    for city in cities:
        sub = df[df["city"] == city].groupby("hour")["aqi"].mean()
        ax4.plot(sub.index, sub.values, color=COLORS[city], lw=2,
                 marker="o", ms=3, label=city)
    ax4.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID,
               labelcolor=TEXT, framealpha=0.9)
    ax4.set_xticks(range(0, 24, 4))

    # ── 5. City AQI distribution (violin) ───────────────────────────
    ax5 = fig.add_subplot(gs[2, 0:2])
    style_ax(ax5, "AQI Distribution by City", "", "AQI")
    data = [df[df["city"] == c]["aqi"].values for c in cities]
    parts = ax5.violinplot(data, positions=range(len(cities)),
                           showmedians=True, showextrema=False)
    for i, (pc, city) in enumerate(zip(parts["bodies"], cities)):
        pc.set_facecolor(COLORS[city])
        pc.set_edgecolor(COLORS[city])
        pc.set_alpha(0.5)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(1.5)
    ax5.set_xticks(range(len(cities)))
    ax5.set_xticklabels([c.title() for c in cities], color=TEXT, fontsize=9)

    # ── 6. Residuals ─────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    style_ax(ax6, "Prediction Residuals", "Residual (Actual − Predicted)", "Frequency")
    residuals = df["aqi"] - df["predicted"]
    ax6.hist(residuals, bins=60, color="#5B8FD4", edgecolor="none", alpha=0.85)
    ax6.axvline(0, color="white", lw=1.2, linestyle="--", alpha=0.7)
    ax6.text(0.97, 0.93, f"μ = {residuals.mean():.2f}\nσ = {residuals.std():.2f}",
             transform=ax6.transAxes, ha="right", color=TEXT, fontsize=8,
             bbox=dict(facecolor=GRID, edgecolor="none", alpha=0.8, pad=4))

    out = "outputs/dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"Dashboard saved → {out}")

if __name__ == "__main__":
    df, model, cols = load()
    plot(df, model, cols)