# Load Testing Results — Locust Flood Simulation

## Methodology

`locustfile.py` simulates a field-agent user: mostly submitting leaf photos for
prediction (`POST /predict`, weight 8), occasionally checking status (`GET /health`,
weight 2) and model metrics (`GET /model-info`, weight 1) — matching realistic usage
of the UI's Diagnose and Overview tabs.

Three scenarios were run, each for 45 seconds against a real, running instance of the
API (not simulated/estimated numbers):

| Scenario | Server config | Concurrent users | Spawn rate |
|---|---|---|---|
| A | 1 uvicorn worker | 10 | 2/s |
| B | 2 uvicorn workers | 10 | 2/s |
| C | 1 uvicorn worker | 20 | 5/s |

**How "different numbers of containers" was simulated:** this environment doesn't
have Docker or multiple machines available, so container-count scaling was
approximated with `uvicorn --workers N` (multiple worker processes behind one
listener — architecturally similar to multiple container replicas behind a load
balancer, minus the load balancer and separate CPU allocation). **Important
limitation, stated plainly:** this test machine has a single CPU core. Real
horizontal scaling (multiple Docker containers, each with their own CPU) on Render
or any real multi-core host should show a much clearer throughput improvement from
scenario A to B than what's reported here — see "Recommendation" below.

## Results

| Scenario | Requests | Failures | Median | p95 | p99 | Throughput |
|---|---|---|---|---|---|---|
| A — 1 worker, 10 users | 217 | 0 (0%) | 150ms | 380ms | 1100ms | 4.93 req/s |
| B — 2 workers, 10 users | 227 | 0 (0%) | 150ms | 340ms | 2500ms | 5.02 req/s |
| C — 1 worker, 20 users | 366 | 0 (0%) | 400ms | 820ms | 1400ms | 8.12 req/s |

![Load test comparison](data/locust_comparison.png)

## Reading these results

**A vs. B (worker/"container" count, same load):** throughput is essentially
identical (4.93 → 5.02 req/s, +1.8%) and p99 latency actually got *worse*
(1100ms → 2500ms). This is the expected, honest outcome on a single-CPU host: two
TensorFlow-serving processes contending for one core doesn't add capacity, it adds
context-switching overhead. This result is a direct consequence of the test
hardware, not a flaw in the API or the horizontal-scaling approach itself — it's
exactly why real deployments scale via separate containers/cores, not extra
processes crammed onto one core.

**A vs. C (load level, same server config):** this is the more meaningful
comparison available in this environment. Doubling concurrent users (10→20) roughly
doubled throughput (4.93→8.12 req/s) but latency degraded non-linearly — median
jumped 150ms→400ms (2.7x) and p95 380ms→820ms (2.2x). Zero failures in all three
scenarios — the API queues gracefully under load rather than dropping or erroring
on requests, but response times climb as the single CPU core becomes the
bottleneck. This is classic queueing behavior for a CPU-bound (not I/O-bound)
workload: TensorFlow inference genuinely occupies the CPU per request, so more
concurrent requests must wait their turn rather than being served in parallel.

## Recommendation for the real deployment

Once deployed to Render (see `DEPLOYMENT.md`), re-run this exact test against the
live URL with genuinely separate container instances (Render's autoscaling, or
manually comparing a single instance vs. a scaled-up instance count) to get a
throughput-vs-container-count curve that reflects real infrastructure rather than
this single-core sandbox. The command is identical, just point `--host` at the
deployed URL:

```bash
locust -f locustfile.py --host https://<your-app>.onrender.com \
    --users 10 --spawn-rate 2 --run-time 60s --headless --csv locust_results/render_1instance
```

Expect real multi-instance/multi-core results to show clear throughput gains from
scenario A→B (unlike this sandbox), since each container gets its own CPU rather
than sharing one. Also expect a cold-start outlier on the very first request if the
Render free-tier instance had spun down from inactivity (see `DEPLOYMENT.md`).

## Raw data

Full percentile breakdowns and per-request CSVs are in `locust_results/`
(`*_stats.csv`, `*_stats_history.csv`) for each of the three scenarios above.
