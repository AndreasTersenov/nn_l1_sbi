"""fig:embed -- 2-D embedding of the 10-D summaries, one panel per compressor, coloured by cosmology.

Near-sufficiency, made visual: if the hand-crafted (l1+product, joint l1) and the learned (CNN)
summaries organise the cosmology space the same way, that is the near-sufficiency claim you can see.
Each panel is a t-SNE of the 10-D VMIM/CNN summary (z-scored), one point per (subsampled) mock,
coloured by sigma8. Illustrative (t-SNE geometry is non-quantitative), not evidential.

Reads the committed 10-D summary caches. sklearn t-SNE (no UMAP). Double column (1x3).
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from sklearn.manifold import TSNE
from pathlib import Path
import sys

from nn_l1_sbi.figures.style import use_style, WIDTH_DOUBLE
from nn_l1_sbi.figures._data import A, CNNP, DERIVED, ensure_output_dir, output_root, require
HERE = output_root() / "figures"    # $NN_L1_SBI_OUT/figures -- never the installed package
use_style()
GEN_SUMM = ("nn_l1_sbi.compression.vmim_from_cache (l1 arms) / "
            "nn_l1_sbi.cnn.npe_cnn_nbody_tomo (CNN arm) -- the 10-D summary caches")

N_SUB = 6000          # subsample for t-SNE (same indices across compressors -> same cosmologies)
ARMS_EMB = [
    ("l1_product", r"$\ell_1$+product", f"{A}/l1product_vmim_s41/cache/l1_train.npz"),
    ("jointl1",    r"joint $\ell_1$",   f"{A}/jointl1_nobnt/cache/l1_train.npz"),
    ("cnn",        "CNN",               f"{CNNP}/arch_sweep_2026_06_13/cnn_resnet18_s41/cache/cnn_train.npz"),
]


CACHE = DERIVED / "embed_tsne.npz"


def main(recompute=False):
    ensure_output_dir("figures")    # create HERE on use, not as an import side effect
    if CACHE.exists() and not recompute:
        z = np.load(CACHE)
        embs = {k: z[f"emb_{k}"] for k, _, _ in ARMS_EMB}
        S8, Om = z["S8"], z["Om"]
    else:
        rng = np.random.default_rng(0)
        th = np.load(require(ARMS_EMB[0][2], GEN_SUMM))["theta"]
        idx = rng.choice(th.shape[0], N_SUB, replace=False)
        Om, s8 = th[idx, 0], th[idx, 1]
        S8 = s8 * np.sqrt(Om / 0.3)                                    # lensing-constrained combination
        embs = {}                                                     # one t-SNE per compressor (reused both rows)
        for key, lab, path in ARMS_EMB:
            x = np.load(require(path, GEN_SUMM))["x"][idx].astype(np.float64)
            x = (x - x.mean(0)) / (x.std(0) + 1e-12)
            embs[key] = TSNE(n_components=2, perplexity=40, init="pca", random_state=0,
                             learning_rate="auto").fit_transform(x)
            print("t-SNE done:", key, flush=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, S8=S8, Om=Om, **{f"emb_{k}": embs[k] for k in embs})

    ROWS = [(r"$S_8$", S8, "viridis"),
            (r"$\Omega_{\rm m}$", Om, "plasma")]
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH_DOUBLE, 0.70 * WIDTH_DOUBLE), constrained_layout=False)
    fig.subplots_adjust(left=0.015, right=0.88, top=0.945, bottom=0.04, wspace=0.05, hspace=0.06)
    for r, (rlab, cval, cmap) in enumerate(ROWS):
        norm = Normalize(vmin=np.percentile(cval, 1), vmax=np.percentile(cval, 99))
        for c, (key, lab, path) in enumerate(ARMS_EMB):
            ax = axes[r, c]
            ax.scatter(embs[key][:, 0], embs[key][:, 1], c=cval, cmap=cmap, norm=norm, s=3, lw=0,
                       alpha=0.7, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(lab, fontsize=9.5)
        pos0, pos1 = axes[r, 2].get_position(), axes[r, 0].get_position()
        cax = fig.add_axes([0.895, pos0.y0 + 0.02, 0.014, pos0.height - 0.04])
        sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(rlab, fontsize=8.5, labelpad=2); cb.ax.tick_params(labelsize=7.5)
    fig.text(0.45, 0.008, "t-SNE of the 10-D summary (each point a mock)", ha="center", fontsize=7.5,
             color="0.4")
    out = HERE / "fig_embed.pdf"
    fig.savefig(out); fig.savefig(HERE / "_preview" / "fig_embed.png", dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    (HERE / "_preview").mkdir(parents=True, exist_ok=True)
    main(recompute="--recompute" in sys.argv)
