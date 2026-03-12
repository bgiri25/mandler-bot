import math, random, time, statistics, os
from multiprocessing import Pool
import matplotlib.pyplot as plt


def estimate_pi_serial(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    return 4 * inside_circle / num_samples

def estimate_pi_chunk(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    return inside_circle

def estimate_pi_parallel(num_samples, num_processes=4):
    samples_per_process = num_samples // num_processes
    tasks = [samples_per_process] * num_processes

    remainder = num_samples % num_processes
    # distribute the remainder so total samples sum to num_samples
    tasks = [samples_per_process + (1 if i < remainder else 0) for i in range(num_processes)]

    with Pool(processes=num_processes) as pool:
        results = pool.map(estimate_pi_chunk, tasks)
    return 4 * sum(results) / num_samples

if __name__ == '__main__':
    num_samples = 10_000_000
      # Measure serial time (T1) using the pure-serial function
    serial_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        pi_serial = estimate_pi_serial(num_samples)
        serial_times.append(time.perf_counter() - t0)
    t_serial = statistics.median(serial_times)
    print(f"Serial (1 worker): {t_serial:.3f}s pi={pi_serial:.6f}")
    for num_proc in range(1, os.cpu_count() + 1):
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            pi_est = estimate_pi_parallel(num_samples, num_proc)
            times.append(time.perf_counter() - t0)
        t = statistics.median(times)
        # Speedup relative to serial and parallel efficiency
        speedup = t_serial / t if t > 0 else float('inf')
        efficiency = speedup / num_proc
        print(
            f"{num_proc:2d} workers: {t:.3f}s pi={pi_est:.6f} | "
            f"Speedup={speedup:.2f}x Efficiency={efficiency*100:.1f}%"
        )
        # collect for plotting
        if 'procs' not in locals():
            procs = []
            times_list = []
            speedups = []
            efficiencies = []
        procs.append(num_proc)
        times_list.append(t)
        speedups.append(speedup)
        efficiencies.append(efficiency)

    # After loop: plot speedup and overhead curves
    # relative overhead: (p * T_p - T1) / T1
    overhead_rel = [(p * tp - t_serial) / t_serial for p, tp in zip(procs, times_list)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.plot(procs, speedups, 'o-', label='Measured speedup')
    ax1.plot(procs, procs, 'k--', label='Ideal speedup')
    ax1.set_xlabel('Number of workers (p)')
    ax1.set_ylabel('Speedup S_p')
    ax1.set_title('Speedup vs workers')
    ax1.legend()

    ax2.plot(procs, overhead_rel, 's-r', label='Relative overhead')
    ax2.axhline(0.0, color='k', linestyle='--')
    ax2.set_xlabel('Number of workers (p)')
    ax2.set_ylabel('Relative overhead ( (p T_p - T1)/T1 )')
    ax2.set_title('Parallel overhead vs workers')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('monte_carlo_overhead.png', dpi=150)
    plt.show()