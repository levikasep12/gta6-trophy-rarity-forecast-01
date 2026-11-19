# gta6-trophy-rarity-forecast-01 — Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/levikasep12/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/levikasep12/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's earn rate on
[Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at 2026-11-26T00:00:00Z, as five quantiles per trophy, from the house's
recorded reads up to the deadline.

`main.ts` (Node 22, standard library only, no network, deterministic) works
on the logit scale: it dedupes Exophase's stale plateaus, fits a per-trophy
logit-linear trend in log-time with empirical-Bayes shrinkage across
trophies, forecasts completion trophies (platinum / 100% completion) by
player-count dilution instead of growth, anchors the overall level on the
mean trophies earned per tracked player, and widens its spread with the
horizon and the fit's uncertainty. A flat history (the practice cases) gets
a dilution shift down and a wide band, because the one refresh on record
cut every rate except the 100% one.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `*.in` case in the input directory and writes `<id>.out` per
case: `{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]}`.

Entered on [make.fun](https://make.fun/?ref=4qtem8kg&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
