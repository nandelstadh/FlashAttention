#!/usr/bin/env python3
"""Plot the FlashAttention benchmark results in results/ into figures/.

  time.png          time per forward pass vs N (CPU vs GPU, scaling)
  gflops.png        effective FLOP rate vs N, with the L40s peaks
  speedup.png       FA kernel relative to the FA1-order kernel, unfused GPU and the CPU
  memory.png        extra memory vs N with the O(N^2) and O(Nd) models
  dram_traffic.png  measured DRAM traffic vs N with traffic models and the L2 limit
  bandwidth.png     achieved DRAM bandwidth vs N
  roofline.png      L40s roofline with every GPU configuration

All csv files per implementation are read and, per configuration, the fastest
run is kept (best of runs removes interference from other jobs on the node);
single runs are drawn as faint markers when there are several. Implementations
without csv files are left out; fa1_*.csv holds the earlier kernel with the
FlashAttention-1 loop order (O, l, m in global memory). Colour and
marker identify the implementation (same in every figure), line style the head
dimension. The numbers quoted in the report go to figures/summary.txt.
"""
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")

# NVIDIA L40s datasheet: 864 GB/s GDDR6, 96 MB L2, 91.6 TFLOP/s fp32, 362 TFLOP/s
# dense fp16 tensor (fp32 accumulate); usable memory as reported by the job logs
BW = 864.0                  # GB/s
L2 = 96e6                   # bytes
GPU_MEM = 44.47 * 2**30     # bytes
PEAK_FP32 = 91.6e3          # GFLOP/s
PEAK_FP16 = 362e3           # GFLOP/s
BC = 64                     # K/V tile rows of the FA1-order kernel on the L40s (d = 64 and 128)
FLOAT = 4
MIB = 2**20

IMPLS = ("fa", "unfused", "cpu", "fa1")
LABEL = {"fa": "FA kernel, GPU (fp16 tensor cores)",
         "unfused": "unfused libtorch, GPU (fp32)",
         "cpu": "unfused libtorch, CPU (8 cores)",
         "fa1": "FA kernel, FA1 loop order (O in global memory)"}
# categorical slots 1-4 of the validated reference palette, in fixed order (as in Lab3)
COLOR = {"fa": "#2a78d6", "unfused": "#eb6834", "cpu": "#1baf7a", "fa1": "#eda100"}
MARKER = {"fa": "o", "unfused": "s", "cpu": "^", "fa1": "D"}
FA_KERNELS = ("fa", "fa1")
D_LS = {64: "-", 128: "--"}
HEADS = ((1, 1), (4, 8))
HEADS_TITLE = {(1, 1): "single head (B = 1, nh = 1)", (4, 8): "32 heads (B = 4, nh = 8)"}
FIGSIZE = (14, 5.5)
REF_STYLE = dict(color="0.4", ls="--", lw=1.1, zorder=0)
MODEL_STYLE = dict(color="0.45", ls=":", lw=1.4, zorder=1)
MARK_STYLE = dict(color="0.55", ls=":", lw=1.0, zorder=0)

SUMMARY = []
NFILES = {}   # number of csv files (runs) per implementation


def note(line=""):
    SUMMARY.append(line)


def load():
    """best[impl][(B, nh, N, d)] = fastest row (or 'failed'/'skipped'); runs[impl] = all rows."""
    best, runs = {}, {}
    note("Data (best of runs per configuration)")
    for impl in IMPLS:
        paths = sorted(glob.glob(os.path.join(RESULTS, f"{impl}_*.csv")),
                       key=lambda p: int(re.search(r"_(\d+)\.csv$", p)[1]))
        if not paths:
            continue
        NFILES[impl] = len(paths)
        best[impl], runs[impl] = {}, []
        for path in paths:
            with open(path) as fh:
                for r in csv.DictReader(fh):
                    key = tuple(int(r[k]) for k in ("B", "nh", "N", "d"))
                    if r["iters"] in ("failed", "skipped"):
                        best[impl].setdefault(key, r["iters"])
                        continue
                    row = {k: float(v) for k, v in r.items() if v != ""}
                    runs[impl].append((key, row))
                    old = best[impl].get(key)
                    if not isinstance(old, dict) or row["ms"] < old["ms"]:
                        best[impl][key] = row
        spread = 0.0
        for key in best[impl]:
            times = [row["ms"] for k, row in runs[impl] if k == key]
            if len(times) > 1:
                spread = max(spread, max(times) / min(times) - 1)
        note(f"  {impl:8s} {len(paths)} run(s): "
             + ", ".join(os.path.relpath(p, ROOT) for p in paths)
             + (f"; largest run-to-run spread {100 * spread:.0f} %" if len(paths) > 1 else ""))
    if not best:
        raise SystemExit(f"no csv files in {RESULTS}")
    return best, runs


def present(best, impls=IMPLS):
    """The implementations among impls that have results."""
    return [i for i in impls if i in best]


def configs(best, impl, heads, d):
    """Sorted (N, row) of valid rows and N of failed/skipped ones."""
    ok, bad = [], []
    for (b, nh, n, dd), row in best[impl].items():
        if (b, nh) == heads and dd == d:
            (ok if isinstance(row, dict) else bad).append((n, row))
    return sorted(ok, key=lambda t: t[0]), sorted(n for n, _ in bad)


def flops(heads, n, d):
    return 4.0 * heads[0] * heads[1] * n * n * d


def plot_impl(ax, best, runs, impl, heads, y):
    """One line per d for impl; faint markers for single runs; x near the axis floor for failures."""
    xs_bad = []
    for d in (64, 128):
        ok, bad = configs(best, impl, heads, d)
        if ok:
            ax.plot([n for n, _ in ok], [y(row, heads, n, d) for n, row in ok],
                    color=COLOR[impl], ls=D_LS[d], marker=MARKER[impl], lw=1.8, ms=6)
        if NFILES[impl] > 1:
            pts = [(k[2], y(row, heads, k[2], d)) for k, row in runs[impl]
                   if k[:2] == heads and k[3] == d]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls="none",
                    marker=MARKER[impl], ms=4, color=COLOR[impl], alpha=0.3, zorder=1)
        xs_bad += bad
    for n in sorted(set(xs_bad)):
        ax.plot(n, 0.04 + 0.06 * IMPLS.index(impl), marker="x", ms=9, mew=2, color=COLOR[impl],
                transform=ax.get_xaxis_transform(), clip_on=False, ls="none")
    return xs_bad


def ref_line(ax, value, text, x=0.01):
    ax.axhline(value, **REF_STYLE)
    ax.annotate(text, xy=(x, value), xycoords=("axes fraction", "data"),
                fontsize=8, color="0.3", va="bottom", ha="right" if x > 0.5 else "left")


def log_axes(ax, title, ylabel, subs=(1.0, 2.0, 5.0)):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("sequence length N")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.12)


def legend(fig, impls, model=None, failures=True, guide=None):
    """One legend below the panels: implementations (colour + marker), d (line style), extras."""
    handles = [Line2D([], [], color=COLOR[i], marker=MARKER[i], lw=1.8, ms=6, label=LABEL[i])
               for i in impls]
    handles += [Line2D([], [], color="0.3", ls=D_LS[d], lw=1.8, label=f"d = {d}") for d in (64, 128)]
    if guide:
        handles.append(Line2D([], [], **REF_STYLE, label=guide))
    if model:
        handles.append(Line2D([], [], **MODEL_STYLE, label=model))
    if failures:
        handles.append(Line2D([], [], ls="none", marker="x", ms=8, mew=2, color="0.3",
                              label="× near the axis: out of memory / skipped"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), fontsize=9,
               frameon=False)


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, name)
    fig.tight_layout(rect=(0, 0.12 if fig.legends else 0, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_time(best, runs):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    note()
    note("Time per forward pass at the largest N that ran [ms, GFLOP/s]")
    for ax, heads in zip(axes, HEADS):
        for impl in present(best):
            plot_impl(ax, best, runs, impl, heads, lambda r, *_: r["ms"])
            for d in (64, 128):
                ok, _ = configs(best, impl, heads, d)
                n, row = ok[-1]
                note(f"  {heads[0] * heads[1]:2d} head(s) d = {d:3d} {impl:8s} N = {n:5d}: "
                     f"{row['ms']:9.2f} ms, {row['gflops']:8.1f} GFLOP/s")
        # N^2 guide through the slowest GPU d = 64 curve at its largest N, shifted down
        ok, _ = configs(best, present(best)[0], heads, 64)
        n_ref, row = ok[-1]
        ns = [1024, 16384]
        ax.plot(ns, [0.25 * row["ms"] * (n / n_ref) ** 2 for n in ns], **REF_STYLE)
        log_axes(ax, HEADS_TITLE[heads], "time per forward pass [ms]")
    legend(fig, present(best), guide="∝ N² (guide)")
    fig.suptitle("Time per forward pass on the L40s and 8 CPU cores", fontsize=12)
    save(fig, "time.png")


def plot_gflops(best, runs):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    for ax, heads in zip(axes, HEADS):
        for impl in present(best):
            plot_impl(ax, best, runs, impl, heads, lambda r, *_: r["gflops"])
        ref_line(ax, PEAK_FP16, "L40s fp16 tensor-core peak (362 TFLOP/s)")
        ref_line(ax, PEAK_FP32, "L40s fp32 peak (91.6 TFLOP/s)")
        log_axes(ax, HEADS_TITLE[heads], "effective performance [GFLOP/s] (4·N²·d per head)",
                 subs=(1.0,))
    legend(fig, present(best))
    fig.suptitle("Effective FLOP rate", fontsize=12)
    save(fig, "gflops.png")


def plot_speedup(best):
    base = present(best, FA_KERNELS)[0]
    others = [o for o in present(best, ("fa1", "unfused", "cpu")) if o != base]
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    note()
    note(f"Speedup of {LABEL[base]} (time of other / time of FA)")
    for ax, heads in zip(axes, HEADS):
        for other in others:
            for d in (64, 128):
                fa, _ = configs(best, base, heads, d)
                ref = dict(configs(best, other, heads, d)[0])
                pts = [(n, ref[n]["ms"] / row["ms"]) for n, row in fa if n in ref]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLOR[other],
                        ls=D_LS[d], marker=MARKER[other], lw=1.8, ms=6)
                s = [p[1] for p in pts]
                note(f"  {heads[0] * heads[1]:2d} head(s) d = {d:3d} vs {other:8s}: "
                     f"{min(s):7.2f}x .. {max(s):7.2f}x, at N = {pts[-1][0]}: {s[-1]:.2f}x")
        ref_line(ax, 1.0, "equal time", x=0.99)
        log_axes(ax, HEADS_TITLE[heads], "speedup of the FA kernel [×] (time of other / time of FA)",
                 subs=(1.0, 3.0))
    handles = [Line2D([], [], color=COLOR[o], marker=MARKER[o], lw=1.8, ms=6, label=f"vs {LABEL[o]}")
               for o in others]
    handles += [Line2D([], [], color="0.3", ls=D_LS[d], lw=1.8, label=f"d = {d}") for d in (64, 128)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("FA kernel relative to unfused attention (> 1: FA is faster)", fontsize=12)
    save(fig, "speedup.png")


def plot_memory(best, runs):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    note()
    note("Memory beyond the inputs [MiB]: measured vs model")
    ns = [1024 * 2 ** i for i in range(5)]
    for ax, heads in zip(axes, HEADS):
        h = heads[0] * heads[1]
        for impl in present(best):
            plot_impl(ax, best, runs, impl, heads,
                      lambda r, *_: r["peak_rss_mib"] if "peak_rss_mib" in r else r["extra_mib"])
        ax.plot(ns, [2 * h * n * n * FLOAT / MIB for n in ns], **MODEL_STYLE)
        ax.annotate("2·B·nh·N²·4 B (two copies of S)", xy=(ns[2], 2 * h * ns[2] ** 2 * FLOAT / MIB),
                    xytext=(10, -4), textcoords="offset points", fontsize=8, color="0.35", va="top")
        for d in (64, 128):
            ax.plot(ns, [h * n * d * FLOAT / MIB for n in ns], **MODEL_STYLE)
        ax.annotate("B·nh·N·d·4 B (O; FA1 order also l, m)", xy=(ns[2], h * ns[2] * 64 * FLOAT / MIB),
                    xytext=(0, -14), textcoords="offset points", fontsize=8, color="0.35", va="top")
        ref_line(ax, GPU_MEM / MIB, "L40s memory (44.5 GiB usable)")
        for d in (64, 128):
            for impl in present(best, ("fa", "fa1", "unfused")):
                ok, _ = configs(best, impl, heads, d)
                n, row = ok[-1]
                note(f"  {h:2d} head(s) d = {d:3d} {impl:8s} N = {n:5d}: {row['extra_mib']:9.1f} MiB")
        log_axes(ax, HEADS_TITLE[heads], "memory beyond Q, K, V [MiB] (CPU: whole-process peak RSS)",
                 subs=(1.0,))
    legend(fig, present(best), model="models")
    fig.suptitle("Memory footprint: unfused grows as N², the FA kernel as N", fontsize=12)
    save(fig, "memory.png")


def fa_traffic_model(heads, n, d):
    """Bytes: compulsory, Q, K, V read and O written once (K/V re-reads by the Q tiles of a
    head are served by L2)."""
    return heads[0] * heads[1] * 4 * n * d * FLOAT


def fa1_traffic_model(heads, n, d):
    """Bytes: compulsory while Q and O (re-touched for every K/V tile) fit in L2,
    otherwise the FA1 loop order reads Q and reads + writes O once per K/V tile."""
    h = heads[0] * heads[1]
    tile = n * d * FLOAT                    # one N x d matrix of one head
    if 2 * h * tile <= L2:
        return h * 4 * tile                 # read Q, K, V, write O
    return h * ((n / BC) * 3 * tile + 2 * tile)


TRAFFIC_MODEL = {"fa": fa_traffic_model, "fa1": fa1_traffic_model}


def unfused_traffic_model(heads, n, d):
    """Bytes: six passes over S (gemm write, scale r/w, softmax r/w, gemm read) + Q, K, V, O."""
    h = heads[0] * heads[1]
    return h * (6 * n * n * FLOAT + 4 * n * d * FLOAT)


def plot_traffic(best, runs):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    note()
    note(f"DRAM traffic: measured / model (FA1-order model uses bc = {BC}, L2 = {L2 / 1e6:.0f} MB)")
    gpu = present(best, ("fa", "fa1", "unfused"))
    dense = [int(1024 * 2 ** (i / 16)) for i in range(65)]
    for ax, heads in zip(axes, HEADS):
        h = heads[0] * heads[1]
        for impl in gpu:
            plot_impl(ax, best, runs, impl, heads, lambda r, *_: r["dram_gb"])
        ax.plot(dense, [unfused_traffic_model(heads, n, 64) / 1e9 for n in dense], **MODEL_STYLE)
        ax.annotate("6 passes over S", xy=(2048, unfused_traffic_model(heads, 2048, 64) / 1e9),
                    xytext=(0, 10), textcoords="offset points", fontsize=8, color="0.35", ha="right")
        for d in (64, 128):
            for impl in present(best, FA_KERNELS):
                ax.plot(dense, [TRAFFIC_MODEL[impl](heads, n, d) / 1e9 for n in dense], **MODEL_STYLE)
            n_l2 = L2 / (2 * h * d * FLOAT)
            if "fa1" in best and dense[0] <= n_l2 <= dense[-1]:
                ax.axvline(n_l2, **MARK_STYLE)
                ax.annotate(f"FA1 order: Q + O = L2, d = {d}", xy=(n_l2, 0.03),
                            xycoords=("data", "axes fraction"), rotation=90, fontsize=8, color="0.35",
                            ha="right", va="bottom")
            for impl in gpu:
                model = TRAFFIC_MODEL.get(impl, unfused_traffic_model)
                ok, _ = configs(best, impl, heads, d)
                ratios = " ".join(f"{row['dram_gb'] * 1e9 / model(heads, n, d):4.2f}" for n, row in ok)
                note(f"  {h:2d} head(s) d = {d:3d} {impl:8s} N = {ok[0][0]}..{ok[-1][0]}: {ratios}")
        ax.annotate("FA model: compulsory (Q, K, V, O once)\nFA1-order model: compulsory while Q + O\n"
                    "fit in L2, then Q + O read/write per K/V tile",
                    xy=(0.02, 0.97), xycoords="axes fraction", fontsize=8, color="0.35", va="top")
        log_axes(ax, HEADS_TITLE[heads], "DRAM traffic per forward pass [GB] (ncu)", subs=(1.0,))
    legend(fig, gpu, model="traffic models")
    fig.suptitle("DRAM traffic: Θ(N²) for unfused, Θ(N·d) for the FA kernel, Θ(N²·d/bc) in FA1 order",
                 fontsize=12)
    save(fig, "dram_traffic.png")


def plot_bandwidth(best, runs):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    for ax, heads in zip(axes, HEADS):
        for impl in present(best, ("fa", "fa1", "unfused")):
            plot_impl(ax, best, runs, impl, heads, lambda r, *_: r["dram_gbps"])
        ref_line(ax, BW, "L40s DRAM peak (864 GB/s)")
        log_axes(ax, HEADS_TITLE[heads], "achieved DRAM bandwidth [GB/s] (ncu)", subs=(1.0, 3.0))
    legend(fig, present(best, ("fa", "fa1", "unfused")))
    fig.suptitle("Achieved DRAM bandwidth", fontsize=12)
    save(fig, "bandwidth.png")


def l2_resident(impl, heads, n, d):
    """Whether L2 hides traffic the implementation would otherwise send to DRAM. The FA kernel
    only moves compulsory traffic, so its DRAM intensity is never inflated."""
    h = heads[0] * heads[1]
    if impl == "fa":
        return False
    if impl == "fa1":
        return 2 * h * n * d * FLOAT <= L2
    return h * n * n * FLOAT <= L2


def plot_roofline(best):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ai_axis = [0.5 * 1.05 ** i for i in range(260)]
    for peak, ls, text in ((PEAK_FP16, "-", "fp16 tensor-core roof"), (PEAK_FP32, "--", "fp32 roof")):
        ax.plot(ai_axis, [min(BW * a, peak) for a in ai_axis], color="0.35", ls=ls, lw=1.3, label=text)
        ax.annotate(f"{text}, ridge at {peak / BW:.0f} flop/B", xy=(2e4, peak), xytext=(0, 4),
                    textcoords="offset points", fontsize=8, color="0.3", ha="center")
    ax.annotate("DRAM 864 GB/s", xy=(3, 3 * BW), xytext=(-4, 4), textcoords="offset points",
                fontsize=8, color="0.3", rotation=37, ha="right")
    note()
    note(f"Roofline: ridge points fp32 {PEAK_FP32 / BW:.0f} flop/B, fp16 tensor {PEAK_FP16 / BW:.0f} flop/B")
    note("  (AI = 4·B·nh·N²·d / measured DRAM bytes; % of the attainable roof of that implementation)")
    # From the traffic models: FA moves 4·N·d·4 B for 4·N²·d flop -> N/4; the FA1 order moves
    # 3·N·d·4 B per K/V tile for 4·N·bc·d flop -> bc/3; unfused moves 6·N²·4 B -> d/6
    note(f"  model AI: FA N/4 flop/B, FA1 order bc/3 = {BC / 3:.1f} flop/B, unfused d/6 = "
         f"{64 / 6:.1f} (d = 64), {128 / 6:.1f} (d = 128) flop/B")
    peaks = {"fa": PEAK_FP16, "fa1": PEAK_FP16, "unfused": PEAK_FP32}
    for impl in present(best, ("fa", "fa1", "unfused")):
        peak = peaks[impl]
        for resident in (False, True):
            xs, ys = [], []
            for heads in HEADS:
                for d in (64, 128):
                    for n, row in configs(best, impl, heads, d)[0]:
                        if l2_resident(impl, heads, n, d) != resident:
                            continue
                        ai = flops(heads, n, d) / (row["dram_gb"] * 1e9)
                        xs.append(ai)
                        ys.append(row["gflops"])
                        if not resident:
                            roof = min(BW * ai, peak)
                            note(f"  {impl:8s} {heads[0] * heads[1]:2d} head(s) d = {d:3d} N = {n:5d}: "
                                 f"AI {ai:7.1f} flop/B, {row['gflops']:8.1f} GFLOP/s = "
                                 f"{100 * row['gflops'] / roof:5.1f} % of roof")
            if not xs:
                continue
            ax.plot(xs, ys, ls="none", marker=MARKER[impl], ms=8, color=COLOR[impl],
                    mfc="none" if resident else COLOR[impl], mew=1.6, zorder=3,
                    label=f"{LABEL[impl]}, " + ("only compulsory traffic" if impl == "fa"
                                                else "working set in L2 (AI inflated)" if resident
                                                else "working set larger than L2"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("arithmetic intensity [flop / DRAM byte]")
    ax.set_ylabel("performance [GFLOP/s]")
    ax.set_title("L40s roofline: 864 GB/s, 91.6 TFLOP/s fp32, 362 TFLOP/s fp16 tensor", fontsize=11)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(fontsize=8, framealpha=0.95, loc="upper left")
    save(fig, "roofline.png")


def main():
    best, runs = load()
    plot_time(best, runs)
    plot_gflops(best, runs)
    plot_speedup(best)
    plot_memory(best, runs)
    plot_traffic(best, runs)
    plot_bandwidth(best, runs)
    plot_roofline(best)

    text = "\n".join(SUMMARY) + "\n"
    with open(os.path.join(FIGDIR, "summary.txt"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
