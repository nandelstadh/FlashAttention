"""Sum DRAM traffic and achieved occupancy over all kernels in an ncu --csv --page raw log.

Prints dram_gb,dram_gbps,dram_pct_peak,occupancy_pct, where the percentage of
peak bandwidth and the achieved occupancy are duration-weighted means over kernels.
"""
import csv
import sys


def number(s):
    return float(s.replace(",", ""))


def main(path):
    with open(path) as f:
        lines = f.readlines()
    # ncu writes ==PROF== messages to the log before the CSV header
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))

    total_bytes = total_ns = weighted_pct = weighted_occ = 0.0
    for row in csv.DictReader(lines[start:]):
        if not row["ID"].isdigit():  # Units row
            continue
        ns = number(row["gpu__time_duration.sum"])
        total_bytes += number(row["dram__bytes_read.sum"]) + number(row["dram__bytes_write.sum"])
        total_ns += ns
        weighted_pct += ns * number(row["dram__throughput.avg.pct_of_peak_sustained_elapsed"])
        weighted_occ += ns * number(row["sm__warps_active.avg.pct_of_peak_sustained_active"])

    if total_ns == 0:
        sys.exit("no kernels found in " + path)
    # bytes per ns is GB/s
    print(f"{total_bytes / 1e9:.3f},{total_bytes / total_ns:.1f},{weighted_pct / total_ns:.1f},"
          f"{weighted_occ / total_ns:.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
