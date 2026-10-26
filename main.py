import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUT = "output"
os.makedirs(OUT, exist_ok=True)

PRIORS = {
    "Flat": (1, 1),
    "Jeffreys": (0.5, 0.5),
    "Baseline": (2, 2),
    "Moderate": (10, 10),
    "Strong": (50, 50),
    "Skeptical": (2, 8),
}

STYLE = {
    "Flat": ("tab:blue", "-"),
    "Jeffreys": ("tab:cyan", "--"),
    "Baseline": ("tab:red", "--"),
    "Moderate": ("tab:green", "-."),
    "Strong": ("tab:orange", ":"),
    "Skeptical": ("tab:purple", "-."),
}

N_MAX = 300
SEED = 42
P_TRUE = 0.62


def periodic(pattern, n):
    """Return the first n flips of a repeating H/T pattern."""
    return np.array(
        [
            1 if pattern[i % len(pattern)] == "H" else 0
            for i in range(n)
        ]
    )


SEQ = {
    "S1": periodic("HHTHTHHT", N_MAX),  # p_hat -> 0.625
    "S2": periodic("HT", N_MAX),         # alternating, 0.5
    "S3": periodic("HHHHT", N_MAX),      # 0.8
    "S4": periodic("H", N_MAX),          # all heads
    "S5": np.random.default_rng(SEED).binomial(1, P_TRUE, size=N_MAX),
}

LABEL = {
    "S1": "S1 (HHTHTHHT..., 0.625)",
    "S2": "S2 (HT..., 0.5)",
    "S3": "S3 (HHHHT..., 0.8)",
    "S4": "S4 (all H)",
    "S5": f"S5 (NumPy, p={P_TRUE}, seed {SEED})",
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def post_params(a, b, x):
    """Beta(a,b) prior + flips x -> posterior parameters."""
    s = int(np.sum(x))
    return a + s, b + len(x) - s


def summary(a, b):
    """Return common summary statistics for Beta(a,b)."""
    d = beta(a, b)
    lo, hi = d.ppf([0.025, 0.975])

    return {
        "mean": a / (a + b),
        "sd": d.std(),
        "lo": lo,
        "hi": hi,
        "width": hi - lo,
        "p_gt_half": d.sf(0.5),
    }


def mean_path(prior, x, n_max):
    """Posterior mean after each n = 0..n_max flips."""
    a, b = PRIORS[prior]
    c = np.concatenate([[0], np.cumsum(x[:n_max])])
    n = np.arange(n_max + 1)
    return (a + c) / (a + b + n)


def width_path(prior, x, n_max):
    """95% credible-interval width after each n = 0..n_max flips."""
    a0, b0 = PRIORS[prior]
    c = np.concatenate([[0], np.cumsum(x[:n_max])])

    out = []
    for n in range(n_max + 1):
        d = beta(a0 + c[n], b0 + n - c[n])
        lo, hi = d.ppf([0.025, 0.975])
        out.append(hi - lo)

    return np.array(out)


def tv(p, q):
    """Total-variation distance between two Beta distributions."""
    f = lambda t: abs(beta.pdf(t, *p) - beta.pdf(t, *q))
    pts = np.linspace(0, 1, 11)

    return 0.5 * sum(
        quad(f, pts[i], pts[i + 1], limit=200)[0]
        for i in range(10)
    )


def save_csv(name, header, rows):
    """Save table data as a CSV file inside OUT."""
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def show(title, header, rows):
    """Print a simple text version of a table."""
    print(f"\n=== {title} ===")
    print(" | ".join(header))
    for row in rows:
        print(" | ".join(str(value) for value in row))


def flip_to_char(value):
    """Convert binary flip encoding back to H/T."""
    return "HT"[1 - int(value)]


xs = np.linspace(0.001, 0.999, 600)


# ---------------------------------------------------------------------------
# Table 1: priors
# ---------------------------------------------------------------------------

rows = []

for name, (a, b) in PRIORS.items():
    k = a + b
    sd = np.sqrt(a * b / (k**2 * (k + 1)))

    rows.append(
        [
            name,
            f"Beta({a:g},{b:g})",
            f"{a / k:.3f}",
            f"{sd:.3f}",
            f"{k:g}",
            f"{k:g}",
            f"{9 * k:g}",
        ]
    )

H = [
    "Prior",
    "Beta",
    "mean",
    "SD",
    "kappa",
    "n for w=1/2",
    "n for w=0.9",
]

show("Table 1: priors", H, rows)
save_csv("table1_priors.csv", H, rows)


# ---------------------------------------------------------------------------
# Table 2: first 8 steps, Beta(2,2), S1
# ---------------------------------------------------------------------------

rows = []
prev = None

for n in range(9):
    a, b = post_params(2, 2, SEQ["S1"][:n])
    s = summary(a, b)

    rows.append(
        [
            n,
            "--" if n == 0 else flip_to_char(SEQ["S1"][n - 1]),
            f"({a:g},{b:g})",
            f"{s['mean']:.4f}",
            "--" if prev is None else f"{s['mean'] - prev:+.4f}",
            f"{s['sd']:.4f}",
            f"[{s['lo']:.3f},{s['hi']:.3f}]",
            f"{s['p_gt_half']:.3f}",
        ]
    )

    prev = s["mean"]

H = [
    "n",
    "flip",
    "(alpha_n,beta_n)",
    "mean",
    "change",
    "SD",
    "95% CI",
    "P(p>0.5)",
]

show(
    "Table 2: sequential update, Beta(2,2), first 8 flips of S1",
    H,
    rows,
)
save_csv("table2_steps.csv", H, rows)


# ---------------------------------------------------------------------------
# Table 3: priors x n on S1
# ---------------------------------------------------------------------------

rows = []

for name, (a0, b0) in PRIORS.items():
    for n in (10, 20, 50, 100):
        a, b = post_params(a0, b0, SEQ["S1"][:n])
        s = summary(a, b)

        rows.append(
            [
                name,
                n,
                f"{s['mean']:.3f}",
                f"[{s['lo']:.3f},{s['hi']:.3f}]",
                f"{s['width']:.3f}",
                f"{s['p_gt_half']:.3f}",
            ]
        )

H = [
    "Prior",
    "n",
    "mean",
    "95% CI",
    "width",
    "P(p>0.5)",
]

show("Table 3: posterior summaries on S1", H, rows)
save_csv("table3_grid.csv", H, rows)

print(
    "Sample proportion of S1:",
    {n: round(SEQ["S1"][:n].mean(), 3) for n in (10, 20, 50, 100)},
)


# ---------------------------------------------------------------------------
# Table 5: posterior mean at n=100, prior x sequence
# ---------------------------------------------------------------------------

rows = [
    ["Sample proportion"] + [f"{SEQ[s][:100].mean():.2f}" for s in SEQ]
]

for name in PRIORS:
    rows.append(
        [name]
        + [f"{mean_path(name, SEQ[s], 100)[-1]:.3f}" for s in SEQ]
    )

H = ["Prior"] + list(SEQ)

show("Table 5: posterior mean at n=100", H, rows)
save_csv("table5_prior_by_sequence.csv", H, rows)

print(
    "S5 first 20 flips:",
    "".join(flip_to_char(v) for v in SEQ["S5"][:20]),
    "| heads in first 10/40/100:",
    [int(SEQ["S5"][:n].sum()) for n in (10, 40, 100)],
)

print(
    "S2, Beta(2,2): posterior SD at n=0 and n=40:",
    round(summary(2, 2)["sd"], 3),
    round(summary(*post_params(2, 2, SEQ["S2"][:40]))["sd"], 3),
)


# ---------------------------------------------------------------------------
# Order experiment
# ---------------------------------------------------------------------------

A = np.array([1] * 5 + [0] * 5)
B = np.array([1, 0] * 5)

print("\n=== Order experiment (prior Beta(2,2)) ===")

for name, x in (
    ("A=HHHHHTTTTT", A),
    ("B=HTHTHTHTHT", B),
):
    print(
        name,
        "final posterior Beta%s" % (post_params(2, 2, x),),
        "mean path:",
        np.round(mean_path("Baseline", x, 10), 3),
    )


# ---------------------------------------------------------------------------
# Table 4: total variation to flat posterior
# ---------------------------------------------------------------------------

tv_ns = sorted(set(range(0, N_MAX + 1, 3)) | {1, 10, 50, 100})

TV = {}

for name, (a0, b0) in PRIORS.items():
    if name == "Flat":
        continue

    TV[name] = {
        n: tv(
            post_params(a0, b0, SEQ["S1"][:n]),
            post_params(1, 1, SEQ["S1"][:n]),
        )
        for n in tv_ns
    }

rows = []

for name in PRIORS:
    a, b = PRIORS[name]
    k = a + b

    row = [
        name,
        f"{k:g}",
        f"{50 / (50 + k):.3f}",
        f"{100 / (100 + k):.3f}",
    ]

    if name == "Flat":
        row += ["--"] * 4
    else:
        row += [f"{TV[name][n]:.3f}" for n in (10, 50, 100, 300)]

    rows.append(row)

H = [
    "Prior",
    "kappa",
    "w_50",
    "w_100",
    "TV_10",
    "TV_50",
    "TV_100",
    "TV_300",
]

show(
    "Table 4: data weight and TV to flat-prior posterior (S1)",
    H,
    rows,
)
save_csv("table4_tv.csv", H, rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.fontsize": 8,
    }
)


# Fig 1: evolution of posterior, Beta(2,2), S1

fig, ax = plt.subplots(figsize=(7, 4))

for n, c in zip(
    (0, 1, 5, 10, 20, 50),
    ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "k"),
):
    a, b = post_params(2, 2, SEQ["S1"][:n])
    ax.plot(
        xs,
        beta.pdf(xs, a, b),
        color=c,
        label=f"n={n}: Beta({a:g},{b:g})",
    )

ax.set(
    xlabel="p",
    ylabel="posterior density",
    xlim=(0, 1),
    ylim=(0, None),
    title="Prior -> posterior after 1, 5, 10, 20, 50 flips (S1, Beta(2,2))",
)

ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_evolution.png", dpi=200)
plt.close(fig)


# Fig 2: posteriors from all priors at n=10 and n=50

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

for ax, n in zip(axs, (10, 50)):
    for name, (a0, b0) in PRIORS.items():
        a, b = post_params(a0, b0, SEQ["S1"][:n])
        c, ls = STYLE[name]

        ax.plot(
            xs,
            beta.pdf(xs, a, b),
            color=c,
            ls=ls,
            label=name,
        )

    ax.set(
        xlabel="p",
        ylabel="posterior density",
        xlim=(0, 1),
        ylim=(0, None),
        title=f"n = {n} (S1)",
    )

    ax.legend()

fig.tight_layout()
fig.savefig(f"{OUT}/fig2_prior_comparison.png", dpi=200)
plt.close(fig)


# Fig 3: posterior mean and 95% CI width vs n (S1)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
n100 = np.arange(101)

for name in PRIORS:
    c, ls = STYLE[name]

    axs[0].plot(
        n100,
        mean_path(name, SEQ["S1"], 100),
        color=c,
        ls=ls,
        label=name,
    )

    axs[1].plot(
        n100,
        width_path(name, SEQ["S1"], 100),
        color=c,
        ls=ls,
        label=name,
    )

axs[0].plot(
    n100[1:],
    np.cumsum(SEQ["S1"][:100]) / n100[1:],
    "k:",
    lw=1,
    label="sample proportion",
)

axs[1].plot(
    n100[3:],
    3.92 * np.sqrt(0.625 * 0.375 / n100[3:]),
    "k:",
    lw=1,
    label="data-only width",
)

axs[0].set(
    xlabel="n",
    ylabel="posterior mean",
    ylim=(0.1, 0.9),
    title="Posterior mean (S1)",
)

axs[1].set(
    xlabel="n",
    ylabel="95% CI width",
    ylim=(0, 1),
    title="Posterior spread (S1)",
)

axs[0].legend(ncol=2)
axs[1].legend()

fig.tight_layout()
fig.savefig(f"{OUT}/fig3_mean_and_spread.png", dpi=200)
plt.close(fig)


# Fig 4: TV to flat posterior and posterior mean for five sequences

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

for name in PRIORS:
    if name == "Flat":
        continue

    c, ls = STYLE[name]

    axs[0].plot(
        tv_ns,
        [TV[name][n] for n in tv_ns],
        color=c,
        ls=ls,
        label=name,
    )

axs[0].set(
    xlabel="n",
    ylabel="TV to flat-prior posterior",
    ylim=(0, 0.85),
    title="Prior sensitivity (S1)",
)

axs[0].legend()

n40 = np.arange(41)

for seq_name in SEQ:
    axs[1].plot(
        n40,
        mean_path("Baseline", SEQ[seq_name], 40),
        label=LABEL[seq_name],
    )

axs[1].set(
    xlabel="n",
    ylabel="posterior mean (prior Beta(2,2))",
    ylim=(0.3, 1),
    title="Five data sequences",
)

axs[1].legend()

fig.tight_layout()
fig.savefig(f"{OUT}/fig4_tv_and_sequences.png", dpi=200)
plt.close(fig)


# Fig 5: order experiment

fig, ax = plt.subplots(figsize=(6, 3.5))

ax.plot(
    range(11),
    mean_path("Baseline", A, 10),
    "o-",
    label="A: HHHHHTTTTT",
)

ax.plot(
    range(11),
    mean_path("Baseline", B, 10),
    "s--",
    label="B: HTHTHTHTHT",
)

ax.set(
    xlabel="n",
    ylabel="posterior mean",
    title="Same counts, different order: same endpoint Beta(7,7)",
)

ax.legend()

fig.tight_layout()
fig.savefig(f"{OUT}/fig5_order.png", dpi=200)
plt.close(fig)


print(f"\nDone. Figures and CSV tables are in ./{OUT}/")
