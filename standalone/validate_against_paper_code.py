"""Check joint_l1.py against the code that produced the paper's numbers.

Runs the paper's own modules next to joint_l1.py on the same maps and compares:
  1. the starlet transform  vs wl_stats_torch @ 77eafc84 (WLStatistics, coarse mean removed)
  2. the joint l1 features  vs nn_l1_sbi/flatsky/flatsky_joint_stats.py
     (pair2d_features(weighted=True, ranges=...), i.e. --stat jointl1 --adaptive-ranges),
     without BNT and in the BNT basis, with the paper's frozen noise-sigma tables
  3. the SNR-range fit      vs calibrate_joint_ranges from the same module

Needs torch, plus a source checkout of wl_stats_torch at the pinned commit; the
nn_l1_sbi sources default to this repository's src/:
    python validate_against_paper_code.py \
        --wl-stats-torch-src <dir containing wl_stats_torch/ at commit 77eafc84> \
        --sigma-nobnt <flatsky_cross_noise_sigma.npz> --sigma-bnt <flatsky_cross_noise_sigma_bnt.npz>

The maps are toy maps (joint_l1._toy_tomographic_maps) with the paper's pixel noise:
this compares two implementations, so any input with realistic SNR values will do.
"""
import argparse
import ast
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import joint_l1  # noqa: E402


def read_paper_bnt_matrix(nn_l1_sbi_src):
    """BNT_MATRIX literal from bnt_utils.py, read without importing tensorflow."""
    path = os.path.join(nn_l1_sbi_src, "nn_l1_sbi", "flatsky", "bnt_utils.py")
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "BNT_MATRIX":
            rows = ast.literal_eval(node.value.args[0])
            # the paper stores it as float32 and torch casts it to the map dtype
            return np.asarray(rows, dtype=np.float32).astype(np.float64)
    raise RuntimeError(f"BNT_MATRIX not found in {path}")


def load_sigma(path):
    table = np.load(path)
    return np.asarray(table["sigma"], dtype=np.float64)[:4], float(table["sigma_pix"])


def report(name, ours, theirs, tolerance):
    difference = np.abs(ours - theirs)
    scale = np.abs(theirs).max()
    relative = difference.max() / scale
    status = "OK " if relative < tolerance else "FAIL"
    print(f"  [{status}] {name}: max |diff| = {difference.max():.3e} "
          f"(relative to max |value| {relative:.2e}, tolerance {tolerance:.0e})")
    return relative < tolerance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nn-l1-sbi-src", default=os.path.join(os.path.dirname(HERE), "src"))
    parser.add_argument("--wl-stats-torch-src", required=True)
    parser.add_argument("--sigma-nobnt", required=True)
    parser.add_argument("--sigma-bnt", required=True)
    parser.add_argument("--n-maps", type=int, default=600)
    args = parser.parse_args()

    sys.path.insert(0, args.wl_stats_torch_src)
    sys.path.insert(0, args.nn_l1_sbi_src)
    import torch
    from wl_stats_torch import WLStatistics
    from nn_l1_sbi.flatsky import flatsky_joint_stats as paper

    torch.set_num_threads(16)
    stats = WLStatistics(n_scales=5, device="cpu", pixel_arcmin=7.5, dtype=torch.float64)
    bnt = read_paper_bnt_matrix(args.nn_l1_sbi_src)
    all_ok = True

    sigma_nobnt, sigma_pix = load_sigma(args.sigma_nobnt)
    sigma_bnt, _ = load_sigma(args.sigma_bnt)
    rng = np.random.default_rng(7)
    signal = joint_l1._toy_tomographic_maps(args.n_maps, 4, 80, amplitude=0.02, seed=3)
    maps = signal + sigma_pix * rng.standard_normal(signal.shape)   # (n, 4, 80, 80)
    maps_channels_last = np.ascontiguousarray(maps.transpose(0, 2, 3, 1))
    n_train = args.n_maps // 2

    print("1. starlet transform vs wl_stats_torch @ 77eafc84")
    images = maps[:64].reshape(-1, 80, 80)
    stats.compute_wavelet_transform(torch.from_numpy(images), 1.0, subtract_coarse_mean=True)
    theirs = stats.wavelet_coeffs.numpy()
    ours = joint_l1.starlet_transform(images, n_scales=5)
    all_ok &= report("coefficients", ours, theirs, 1e-12)

    for basis_name, basis, sigma in (("no BNT", False, sigma_nobnt), ("BNT", bnt, sigma_bnt)):
        print(f"2./3. joint l1, {basis_name}")
        sigma_torch = torch.from_numpy(sigma)

        def train_batches():
            for start in range(0, n_train, 100):
                yield maps_channels_last[start:start + 100]

        paper_ranges = paper.calibrate_joint_ranges(
            train_batches(), basis, sigma_torch, stats, 10, n_examples=n_train)

        mixing = None if basis is False else basis
        ours_stat = joint_l1.JointL1(sigma, n_bins=10, mixing_matrix=mixing)

        # features on the held-out maps, with the paper's own fitted ranges
        ours_stat.snr_range = ours_stat._as_range_table(paper_ranges.numpy())
        held_out = maps_channels_last[n_train:]
        wc = paper.wavelet_stack(held_out, basis, stats)
        theirs = paper.pair2d_features(wc, sigma_torch, 10, weighted=True,
                                       ranges=paper_ranges).numpy()
        ours = ours_stat(maps[n_train:])
        all_ok &= report(f"features {ours.shape}", ours, theirs, 1e-10)
        cells_differing = np.mean(np.abs(ours - theirs) > 1e-9 * np.abs(theirs).max())
        print(f"         cells differing at all: {cells_differing:.2e} of {ours.size}")

        # range fit: the paper's code mixes all bins and scales in each column, so its
        # per-(bin, scale) table is one pooled range plus sampling jitter
        pooled = ours_stat.fit_snr_range(maps[:n_train])[0, 0]
        table = paper_ranges.numpy()
        print(f"         SNR range, joint_l1 pooled fit: [{pooled[0]:.2f}, {pooled[1]:.2f}]")
        print(f"         SNR range, paper code, all (bin, scale): lo {table[..., 0].min():.2f}"
              f" .. {table[..., 0].max():.2f}, hi {table[..., 1].min():.2f}"
              f" .. {table[..., 1].max():.2f}")
        per_bin = ours_stat.fit_snr_range(maps[:n_train], per_bin_and_scale=True)
        print("         SNR range per (bin, scale) if fitted separately, hi edge:\n"
              + np.array2string(per_bin[..., 1], precision=2, prefix="         "))

    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
