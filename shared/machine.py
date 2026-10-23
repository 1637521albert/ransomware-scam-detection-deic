# Load of the machine while a runtime is measured (Linux, from /proc).
#
# A runtime is only comparable with another one measured on the same machine without
# other work competing for its cores and memory. These helpers measure the CPU time used
# by the other processes of the machine (all the CPU time of the machine, not idle or
# waiting for the disk, minus the time of this process and of the children it waited
# for, such as the forked workers of an expansion) and the available memory, and name the
# processes that use the machine. Without /proc (not Linux) they return None.

import os
import time

CLOCK_TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def busy_cpu_seconds():
    """CPU time used by the whole machine since it started (not idle or waiting for the
    disk; time stolen by the hypervisor counts as used, by other machines)."""
    try:
        with open("/proc/stat") as f:
            fields = [int(v) for v in f.readline().split()[1:]]
    except (OSError, ValueError):
        return None
    # user nice system idle iowait irq softirq steal (guest time is included in user)
    return (sum(fields[:8]) - fields[3] - fields[4]) / CLOCK_TICKS


def own_cpu_seconds():
    """CPU time of this process and of the children it waited for (finished workers)."""
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def memory():
    """Total and available memory in bytes (the page cache counts as available)."""
    values = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, value = line.split(":", 1)
                values[key] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        return None
    if "MemTotal" not in values or "MemAvailable" not in values:
        return None
    return {"total": values["MemTotal"], "available": values["MemAvailable"]}


def snapshot():
    """State to measure the use of the machine from (usage_since)."""
    return {"time": time.perf_counter(), "busy": busy_cpu_seconds(), "own": own_cpu_seconds(), "memory": memory(),
            "loadavg": os.getloadavg()[0] if hasattr(os, "getloadavg") else None}


def usage_since(start):
    """Use of the machine since the snapshot `start`: average cores used by the other
    processes and by this one (with its finished workers), and the memory at the start."""
    wall = time.perf_counter() - start["time"]
    busy = busy_cpu_seconds()
    own = own_cpu_seconds() - start["own"]
    mem = start["memory"]
    usage = {
        "wall_s": wall,
        "own_cores": own / wall if wall > 0 else None,
        "other_cores": None,
        "loadavg_1min_start": start["loadavg"],
        "mem_available_gb_start": mem["available"] / 2 ** 30 if mem else None,
        "mem_available_pct_start": 100 * mem["available"] / mem["total"] if mem else None,
    }
    if busy is not None and start["busy"] is not None and wall > 0:
        usage["other_cores"] = max(0.0, (busy - start["busy"] - own) / wall)
    return usage


def _process_times():
    # CPU ticks and resident memory of every process
    page = os.sysconf("SC_PAGE_SIZE")
    processes = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as f:
                stat = f.read()
            with open(f"/proc/{pid}/statm") as f:
                rss = int(f.read().split()[1]) * page
        except (OSError, ValueError, IndexError):
            continue
        # The fields after the command name (which can contain spaces) start at the state;
        # utime and stime are the 14th and 15th fields
        fields = stat[stat.rfind(")") + 2:].split()
        processes[int(pid)] = (int(fields[11]) + int(fields[12]), rss)
    return processes


def _command(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            command = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return "?"
    return command[:100] or "?"


def machine_load(seconds, top=5):
    """Use of the machine by the other processes over `seconds` while this process waits:
    average cores, available memory and the processes that use the most CPU or memory
    (at least 0.1 cores or 1 GB). None without /proc."""
    if busy_cpu_seconds() is None or memory() is None:
        return None
    start = snapshot()
    before = _process_times()
    time.sleep(seconds)
    after = _process_times()
    usage = usage_since(start)

    me = os.getpid()
    processes = []
    for pid, (ticks, rss) in after.items():
        if pid == me:
            continue
        # A process started during the measurement used all its CPU time in it
        cores = (ticks - before.get(pid, (0, 0))[0]) / CLOCK_TICKS / usage["wall_s"]
        if cores >= 0.1 or rss >= 2 ** 30:
            processes.append({"pid": pid, "cores": round(cores, 2), "rss_gb": round(rss / 2 ** 30, 1),
                              "command": _command(pid)})
    processes.sort(key=lambda p: (p["cores"], p["rss_gb"]), reverse=True)
    return {"seconds": usage["wall_s"], "other_cores": usage["other_cores"],
            "mem_available_gb": usage["mem_available_gb_start"],
            "mem_available_pct": usage["mem_available_pct_start"],
            "busiest_processes": processes[:top]}
