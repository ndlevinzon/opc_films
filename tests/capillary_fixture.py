"""Fake cpptraj surftension outputs in the layout of results/capillary/<row_id>/."""

from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST_COLUMNS = ["row_id", "kind", "replica", "rate_K_per_ns", "source_rate_K_per_ns", "temperature_K",
                    "t_start_K", "t_end_K", "files", "topology", "frame_ps", "ntwx", "cut_A",
                    "source_restart", "discard_ps", "duration_ps", "mdout"]
KB = 1.380649e-23


def write_hold(directory, temperature, replica_index, drift=0.0, seed=0):
    """Write summary, spectrum, roughness, blocks, and log files for one hold."""
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    gamma0 = 60.0 + 0.08 * (250.0 - temperature) + rng.normal(0, 0.5)
    kappa_prime = 400.0
    area = 194.567 * 188.781
    q = np.arange(1, 25) * 2 * np.pi / 194.567
    gamma_q = gamma0 + kappa_prime * q ** 2
    # S(q) in Å^4 from gamma(q) in mN/m: S = kT / (A q^2 gamma), gamma in J/m^2 -> J/Å^2 * 1e-20
    s = KB * temperature / (area * q ** 2 * gamma_q * 1e-3 * 1e-20)
    columns = {"S": s, "Stop": s, "Sbot": s, "q2S": q ** 2 * s, "q2Stop": q ** 2 * s, "q2Sbot": q ** 2 * s,
               "gammaq": gamma_q, "gammaqtop": gamma_q, "gammaqbot": gamma_q,
               "kappaq": np.full_like(q, 5.0), "kappaqtop": np.full_like(q, 5.0), "kappaqbot": np.full_like(q, 5.0)}
    header = "#q (Ang^-1) " + " ".join(f"ST_00001[{k}]" for k in columns)
    lines = [header] + [f"{qi:12.6f} " + " ".join(f"{columns[k][i]:14.6e}" for k in columns) for i, qi in enumerate(q)]
    (directory / "capwave_spectrum.dat").write_text("\n".join(lines) + "\n")
    w = 3.0 * np.sqrt(temperature / 250.0)
    frames = 200
    rough = ["#Frame ST_00001[wtop] ST_00001[wbot] ST_00001[wmean] ST_00001[rhobulk]"]
    for f in range(frames):
        a, b = w + rng.normal(0, 0.1), w + rng.normal(0, 0.1)
        rough.append(f"{f + 1:8d} {a:10.4f} {b:10.4f} {(a + b) / 2:10.4f} {0.0331:10.4f}")
    (directory / "capwave_roughness.dat").write_text("\n".join(rough) + "\n")
    blocks = ["#Frame ST_00001[bgamma] ST_00001[bkappa] ST_00001[bwmean] ST_00001[bwtop] ST_00001[bwbot]"]
    for i in range(5):
        g = gamma0 + drift * i + rng.normal(0, 0.2)
        blocks.append(f"{i + 1:8d} {g:10.4f} {5.0:10.4f} {w:10.4f} {w:10.4f} {w:10.4f}")
    (directory / "capwave_blocks.dat").write_text("\n".join(blocks) + "\n")
    summary = {"temperature": temperature, "nsurf": 2, "area": area, "q_fundamental": q[0], "qmin": q[0],
               "qmax": 0.30, "frames": frames, "skipped": 0, "gamma": gamma0 * 1.05, "gamma_upper": gamma0 * 1.05,
               "gamma_lower": gamma0 * 1.05, "kappa": 5.0, "roughness": w, "roughness_upper": w,
               "roughness_lower": w, "block_mean_gamma": gamma0, "block_sd_gamma": 0.2}
    (directory / "capwave_summary.dat").write_text(
        "# surftension summary\n" + "".join(f"{k} {v}\n" for k, v in summary.items()))
    (directory / "capwave.log").write_text(f"\tgamma, Helfrich intercept = {gamma0:.4f} mN/m\n")
    return gamma0


def make_capillary_tree(root, temperatures=(250, 240, 230, 220, 210, 200, 190, 180, 170, 160, 150),
                        drifting=(160, 150)):
    """A repo-shaped tree: config/manifest.tsv and results/capillary/<row_id>/."""
    root = Path(root)
    rows, truth = [], {}
    for k, temperature in enumerate(temperatures):
        for r in range(3):
            row_id = f"hold-r{r + 1}-4.44-{temperature}"
            rows.append({"row_id": row_id, "kind": "hold", "replica": f"r{r + 1}", "rate_K_per_ns": "",
                         "source_rate_K_per_ns": "4.44", "temperature_K": str(temperature),
                         "t_start_K": str(temperature), "t_end_K": str(temperature), "files": "x.nc",
                         "topology": "x.prmtop", "frame_ps": "10", "ntwx": "5000", "cut_A": "9",
                         "source_restart": "", "discard_ps": "1000", "duration_ps": "10000", "mdout": ""})
            truth[row_id] = write_hold(root / "results" / "capillary" / row_id, temperature, r,
                                       drift=2.0 if temperature in drifting else 0.0, seed=100 * k + r)
    (root / "config").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(root / "config" / "manifest.tsv", sep="\t", index=False)
    return truth
