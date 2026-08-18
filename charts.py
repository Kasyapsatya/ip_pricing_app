"""
Chart rendering for the IP Pricing app — ported directly from the notebook's
policyholder/actuary dashboards (Section 11), trimmed to the 3 charts that
belong alongside the quote results:
  1. Premium build-up waterfall
  2. Incidence heatmap (age x occupation), current quote's cell highlighted
  3. Deferred-period frequency/severity trade-off, current choice highlighted

Every function returns a matplotlib Figure, rendered via st.pyplot(fig) in app.py.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

COLOR_PRIMARY = '#005696'
COLOR_ACCENT = '#E5383B'
COLOR_NEUTRAL = '#CBD5E1'

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 10


def render_premium_waterfall(quote):
    """3-bar waterfall: pooled base -> after income scale -> final (after loading)."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    base = quote['pooled_base_premium']
    after_income = base * quote['income_scale']
    final = quote['final_annual_premium']
    bars = ax.bar(
        ['Base\n(pooled)', '× Income\nScale', '× Loading\n= Final'],
        [base, after_income, final],
        color=[COLOR_NEUTRAL, COLOR_PRIMARY, COLOR_ACCENT],
    )
    for b, v in zip(bars, [base, after_income, final]):
        ax.text(b.get_x() + b.get_width() / 2, v, f'Rs {v:,.0f}', ha='center', va='bottom', fontsize=8)
    ax.set_title("How your premium builds up", fontsize=10)
    ax.set_ylabel("Annual amount (Rs)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    fig.tight_layout()
    return fig


def render_incidence_heatmap(base_table, age_band_labels, occupation_classes,
                              highlight_age_band=None, highlight_occupation=None):
    """Tool 1 heatmap. If a quote is active, its cell gets a highlighted border."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    grid = np.array([[base_table['incidence_table'][(b, o)] for o in occupation_classes]
                      for b in age_band_labels])
    im = ax.imshow(grid, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(occupation_classes)))
    ax.set_xticklabels(occupation_classes)
    ax.set_yticks(range(len(age_band_labels)))
    ax.set_yticklabels(age_band_labels.values())
    for i, b in enumerate(age_band_labels):
        for j, o in enumerate(occupation_classes):
            is_hl = highlight_age_band is not None and b == highlight_age_band and o == highlight_occupation
            text_color = 'white' if grid[i, j] > grid.max() * 0.6 else COLOR_PRIMARY
            ax.text(j, i, f'{grid[i, j]:.1%}', ha='center', va='center',
                     color=text_color, fontsize=9, fontweight='bold' if is_hl else 'normal')
            if is_hl:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                            edgecolor=COLOR_ACCENT, linewidth=3))
    ax.set_title("Incidence — age × occupation", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def render_deferred_tradeoff(deferred_period_table, highlight_weeks=None):
    """Tool 2 dual-axis: P(cross) declining vs avg claim cost rising. If a quote is
    active, its chosen deferred period is marked on the frequency line."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax2 = ax.twinx()
    opts = deferred_period_table.index.tolist()
    ax.plot(opts, deferred_period_table['p_cross_to_claiming'], 'o-', color=COLOR_PRIMARY,
             label='P(cross) — freq')
    ax2.plot(opts, deferred_period_table['avg_claim_cost'], 's--', color=COLOR_ACCENT,
              label='Avg claim cost — sev')
    if highlight_weeks is not None and highlight_weeks in opts:
        idx = opts.index(highlight_weeks)
        ax.plot(highlight_weeks, deferred_period_table['p_cross_to_claiming'].iloc[idx], 'o',
                 color=COLOR_ACCENT, markersize=13, markeredgecolor='black', zorder=5)
    ax.set_xlabel("Deferred period (weeks)")
    ax.set_ylabel("P(cross)", color=COLOR_PRIMARY, fontsize=9)
    ax2.set_ylabel("Avg claim cost (Rs)", color=COLOR_ACCENT, fontsize=9)
    ax.set_title("Frequency ↓ vs severity ↑ trade-off", fontsize=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='center right')
    fig.tight_layout()
    return fig
