#!/usr/bin/env python3
"""Write report/appendix_tables.tex: one longtable per implementation, plus the correctness
test results when a results/test_gpu_*.log is present.

The csvs are read the same way as in plot_results.py (best of runs per configuration), so the
appendix can be regenerated whenever the data is.
"""
import glob
import os
import re

from plot_results import LABEL, RESULTS, ROOT, load

OUT = os.path.join(ROOT, "report", "appendix_tables.tex")

STATUS = {"failed": "out of memory", "skipped": "skipped (does not fit in the job's memory)"}
ORDER = ("fa", "unfused", "cpu")


def title(impl):
    return LABEL[impl][0].upper() + LABEL[impl][1:]


def num(x, digits):
    return f"{x:,.{digits}f}".replace(",", r"\,")


def longtable(caption, label, columns, rows):
    """columns: header cells; rows: lists of already formatted cells."""
    head = " & ".join(columns) + r" \\" "\n" r"\hline"
    out = [
        r"\begingroup\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{1.2}",
        f"\\begin{{longtable}}{{{'r' * len(columns)}}}",
        f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
        r"\hline", head,
        r"\endfirsthead",
        f"\\multicolumn{{{len(columns)}}}{{l}}{{\\emph{{Table \\thetable{{}} continued}}}}\\\\",
        r"\hline", head,
        r"\endhead",
        r"\hline",
        r"\endfoot",
    ]
    for cells in rows:
        out.append(" & ".join(cells) + r" \\")
    out += [r"\end{longtable}", r"\endgroup", ""]
    return "\n".join(out)


def table_rows(best, impl, columns, cells_of):
    """Rows ordered by number of heads, d and N; failed/skipped configurations span the data columns."""
    out = []
    for key in sorted(best[impl], key=lambda k: (k[0] * k[1], k[3], k[2])):
        b, nh, n, d = key
        row = best[impl][key]
        cells = [f"${b}\\times{nh}$", num(n, 0), str(d)]
        if isinstance(row, dict):
            cells += cells_of(row)
        else:
            cells.append(f"\\multicolumn{{{len(columns) - 3}}}{{l}}{{{STATUS[row]}}}")
        out.append(cells)
    return out


def gpu_table(best, impl):
    has_occ = any(isinstance(r, dict) and "occupancy_pct" in r for r in best[impl].values())
    columns = ["$B\\times n_h$", "$N$", "$d$", "$n$", "$t$ [ms]", "GFLOP/s", "mem. [MiB]",
               "DRAM [GB]", "GB/s"] + (["occ. [\\%]"] if has_occ else [])

    def cells(r):
        out = [num(r["iters"], 0), num(r["ms"], 3), num(r["gflops"], 1), num(r["extra_mib"], 1),
               num(r["dram_gb"], 3), num(r["dram_gbps"], 1)]
        if has_occ:
            out.append(num(r["occupancy_pct"], 1) if "occupancy_pct" in r else "--")
        return out

    caption = (f"{title(impl)}, NVIDIA L40s. Mem.: memory beyond the inputs; "
               "DRAM and GB/s: traffic and achieved bandwidth from Nsight Compute"
               + ("; occ.: achieved occupancy." if has_occ else "."))
    return longtable(caption, f"tab:{impl}", columns, table_rows(best, impl, columns, cells))


def cpu_table(best):
    columns = ["$B\\times n_h$", "$N$", "$d$", "$n$", "$t$ [ms]", "GFLOP/s", "peak RSS [MiB]"]

    def cells(r):
        return [num(r["iters"], 0), num(r["ms"], 3), num(r["gflops"], 1), num(r["peak_rss_mib"], 1)]

    return longtable(f"{title('cpu')}.", "tab:cpu", columns, table_rows(best, "cpu", columns, cells))


def test_section():
    """Table of the newest correctness log, if any."""
    logs = glob.glob(os.path.join(RESULTS, "test_gpu_*.log"))
    if not logs:
        return []
    path = max(logs, key=lambda p: int(re.search(r"_(\d+)\.log$", p)[1]))
    pattern = re.compile(r"B=(\d+) nh=(\d+) N=(\d+) d=(\d+): max abs diff: (\S+) -> (\w+)")
    rows = []
    with open(path) as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                b, nh, n, d, diff, result = m.groups()
                rows.append([b, nh, num(int(n), 0), d, f"${float(diff):.2e}$".replace("e-0", "e-"), result])
    if not rows:
        return []
    columns = ["$B$", "$n_h$", "$N$", "$d$", "max $|O - O_\\text{ref}|$", "result"]
    log = os.path.relpath(path, ROOT).replace("_", r"\_")
    return [
        r"\subsection{Correctness}",
        "",
        "The FA kernel is compared with unfused libtorch attention on the GPU, using "
        "standard-normal random $Q$, $K$ and $V$, for a single head ($B = n_h = 1$) and a "
        "batch of heads ($B = 2$, $n_h = 4$), with $d \\in \\{32, 64, 128\\}$ and "
        "$N \\in \\{1, 63, 64, 65, 1000, 4096\\}$. Every candidate tile size is a multiple "
        "of 16, so $N = 1$ is smaller than any tile, and $N = 1$, 63, 65 and 1000 always "
        "leave the last $Q$ tile and the last $K/V$ tile partially filled. This exercises "
        "the zero padding and the masking of the padded scores. $N = 4096$ is a multiple "
        "of both candidate $B_r$, so every $Q$ tile is full. A configuration passes when "
        "\\texttt{torch::allclose} holds with relative and absolute tolerance "
        "$5\\cdot10^{-3}$, which allows for the fp16 inputs of the tensor-core products. "
        f"Data from \\texttt{{{log}}}.",
        "",
        longtable("Correctness of the FA kernel against unfused libtorch on the GPU.",
                  "tab:test", columns, rows),
    ]


def main():
    best, _ = load()
    parts = [
        r"\clearpage",
        r"\appendix",
        r"\section{Measurement tables}\label{app:tables}",
        r"\sloppy",
        "",
        "This appendix lists every measurement behind the figures. Each configuration "
        "$(B, n_h, N, d)$ runs in its own process on random single-precision inputs. The "
        "attention is computed once as an untimed warm-up and once more to calibrate; then "
        "$n = \\min(100, \\max(1, \\lfloor 1000\\,\\text{ms}/t_\\text{cal} \\rfloor))$ calls are "
        "timed together (for $n = 1$ the calibration call is the measurement), and $t$ is the "
        "average time of one call. GPU calls are timed with CUDA events and include the small "
        "kernels that allocate and initialise the output; CPU calls are timed with "
        "\\texttt{std::chrono::steady\\_clock}. GFLOP/s counts $4 B n_h N^2 d$ flops. Memory is "
        "the peak allocated by the libtorch caching allocator during one further call minus "
        "what was allocated before it ($Q$, $K$, $V$ and the cuBLAS workspace). DRAM traffic "
        "comes from a separate run in which Nsight Compute profiles only that one call: "
        "\\texttt{dram\\_\\_bytes\\_read.sum} plus \\texttt{dram\\_\\_bytes\\_write.sum} over "
        "all of its kernels, divided by their summed durations for the bandwidth. Occupancy "
        "is \\texttt{sm\\_\\_warps\\_active.avg.pct\\_of\\_peak\\_sustained\\_active}, weighted "
        "by kernel duration; it was only recorded for the final FA kernel. GPU runs used one "
        "NVIDIA L40s (48\\,GB); CPU runs used 8 CPUs of a Pelle node (Slurm \\texttt{-c 8}, "
        "8 OpenMP threads, 32\\,GB), and each CPU entry is the fastest of the available runs. "
        "Configurations whose unfused score matrices do not fit are marked out of memory or "
        "skipped.",
        "",
    ]
    parts += test_section()
    for impl in ORDER:
        if impl not in best:
            continue
        parts += [r"\clearpage", f"\\subsection{{{title(impl)}}}", ""]
        parts.append(cpu_table(best) if impl == "cpu" else gpu_table(best, impl))

    with open(OUT, "w") as fh:
        fh.write("% generated by make_appendix.py -- do not edit by hand\n")
        fh.write("\n".join(parts))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
