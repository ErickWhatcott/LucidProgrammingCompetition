# LPC 2026 bot (C#)

This is a C# port of the Java starter, with four learned engines that share one planner, a local
simulator, and a self-play training league.

- Build: `dotnet build -c Release`. The binary is `bin/Release/net10.0/lpc`.
- Run it as `./lpc ...` from either folder. Plain `lpc` on macOS is the CUPS printer tool.
- Run every command from this folder. It reads and writes `models/` and `data/`.
- The team key goes in `api.key` (plain text). `lpc whoami` checks it without joining a match.

## Engines

All four engines use the same planner. It handles remembered map, path costs, claims, assignment, building workers and bases, gold-for-iron trades and the end-game return. The engines differ only in how they score each worker's candidate targets.

| Mode | Engine | Learns from |
|---|---|---|
| `mlp` | Neural value scorer (28-64-64-1) | The shared pool: each decision's realized gold per turn until the next deposit |
| `gbdt` | Gradient-boosted trees value scorer, LightGBM-style histogram GBDT in pure C# | The same shared pool |
| `ppo` | Policy net that samples targets with softmax | First imitation of the top-placing seats from any engine, then PPO on its own sampled decisions. Its critic is built from the shared mlp value model. |
| `es` | CMA-ES over 16 planner settings plus 28 linear scorer weights | Game results. Fitness is seat gold minus the match average. Generation 0 reproduces the hand-written scorer. |

The hand-written rate is still a feature, and it is the fallback whenever a model is missing or a turn is short on time.

## Simulator opponents

`intermediate`, `beginner` and `novice` approximate the website's practice bots.

## Commands

```sh
lpc whoami                                   # check api.key
lpc run --record                             # play match after match, rotating es -> ppo -> gbdt -> mlp
lpc run --mode es --record                   # same loop, one engine (or --mode es,ppo to rotate two)
lpc run --mode es --once                     # a single match
lpc train pretrain --minutes 45              # self-play league (all engines, shared pool)
lpc train pretrain --minutes 20 --resume     # continue from saved models + CMA-ES state
lpc train finetune                           # adapt mlp/gbdt/ppo to recorded scrimmages
lpc bench --modes mlp,gbdt,ppo,es --seeds 44 # paired comparison in the simulator
lpc bench --h2h                              # all engines in the same matches
lpc latency                                  # per-engine decision time vs the 100 ms budget
lpc sim --mode es --seed 3 --render          # one match, standings + ASCII map
```

## Training workflow

1. **Pretrain in the simulator** (`train pretrain`). Each round plays 36 matches with 16 seats each:
   - 4 seats for CMA-ES candidates
   - 2 seats each for mlp, gbdt and ppo
   - 6 practice bots

   Every engine seat records into one pool, and every engine then trains from that pool. The latest model for each engine goes to `models/<engine>.json`. A snapshot of the version that had its best round goes to `models/<engine>.best.json`. Replay samples go to `data/sim/`.
2. **Pick the engine** with `lpc bench`.
3. **Scrimmage** with `lpc run --record`. The loop keeps playing new matches until one of three things happens:
   - you press Ctrl+C,
   - another bot takes over the team's session,
   - a new build appears (checked between matches; exit code 3).

   Each match reloads the latest models from disk and is logged to `logs/scrimmages.csv` with engine, gold and placement.

   With `--record`, each match writes `data/live/trips-*.jsonl` and `decisions-*.jsonl` (training data) plus `states-*.jsonl.gz` (raw server states, for checking the simulator).
4. **Fine-tune** with `./lpc train finetune`.
   - It starts from the latest league model and mixes live rows with about 30% sim replay.
   - It writes `models/<engine>.live.json`. Live play prefers that file, and the league never overwrites it.
   - A fine-tuned model that does more than 5% worse in the simulator check is kept as `*.live.rejected.json` and not used. That is likely while there are only a few scrimmages of data.
   - CMA-ES is not fine-tuned on scrimmages, because it needs many games per generation.

## Real-time matches

Scrimmages and ranked matches advance on the server's schedule (`multi-client-action` or `timed`). A web request to lpc.sh takes about 230 ms. In these matches the client reads the server's turn stream (`/api/events/matches/{id}`), the same push feed the official Python and JS clients use, so the bot always acts on the newest turn without polling. If the stream drops, it falls back to 20 ms polling. `--no-stream` forces polling.

Before each match the engine also warms up on a simulated board, so the first real turns aren't slowed by first-call compilation. Practice matches (`client-action`) wait for the bot and keep the simple request/response loop.

## Time budget and hardware (M3 Pro)

- Every turn has a 100 ms budget. Scorers estimate their own cost and fall back to the heuristic rather than run over. `lpc latency` measures whole matches.
- Matrix multiplies run on Apple Accelerate (AMX units). In testing that was about 0.1 ms per 1000 candidates and about 1M training rows/s. The GPU is not used: for networks this small, dispatch overhead would dominate.
- Self-play runs one game per core across all 11 cores.

## Simulator caveats

The real map generator is not published. The simulator's obstacle blobs, ring of bases and uniform deposit placement are assumptions (see `Sim/Simulator.cs`). The game mechanics follow `lpc2026-help.md`.
