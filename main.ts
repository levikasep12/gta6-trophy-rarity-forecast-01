// GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).
//
// Forecasts, for every trophy, five quantiles of its Exophase earn rate at the
// case's target, as percentages. Node 22, standard library only, no network,
// deterministic: the output is a pure function of the case file.
//
// Model (logit scale throughout, rates clamped to [0.01, 99.99] before logit):
//
// 1. Dedupe the history. Exophase refreshes a game's counts on its own
//    schedule, hours apart, while the house reads every 10-30 minutes, so
//    long runs of identical reads are one observation, first seen at the
//    first read that shows it. Fitting the raw reads would treat a stale
//    plateau as many independent observations and flatten every slope.
// 2. Split each rate into counts: rate = 100 * E / N, where E is the number
//    of tracked players who earned the trophy and N the players tracked.
//    E only grows (players keep earning) and N only grows (players keep
//    syncing), so fit ln E and ln N separately, each linear in
//    x = ln(1 + hours since the release day began), weighted toward the
//    newest distinct read. The rate's slope is then bE - bN: this keeps a
//    refresh from looking like a collapse in earning, because at the one
//    refresh on record E grew 1.34x while N grew 6.06x, and the rate fell
//    only through the N side. Per-trophy E-slopes are shrunk
//    empirically-Bayes style toward the cross-trophy mean slope, then
//    blended 50/50 with the last-window slope (clipped to +1.0 above the
//    shrunk slope), because earner growth is convex in x while players are
//    still reaching a trophy and a whole-history slope understates it.
//    Median: logit(median) = logit(last) + (bE* - bN) * (x_target - x_last),
//    floored at the dilution floor 100 * E_last / N_target (earners do not
//    decrease) and, for completion trophies, capped at the last rate.
// 3. Completion trophies (platinum, or "100% Completion" in the
//    description) are special: nobody can play 200 hours in the seven days
//    to the target, so their earners are essentially fixed; their E-slope
//    is capped at 0 and they are forecast by dilution alone.
// 4. A cross-trophy level anchor. The mean trophies earned per tracked
//    player, m = trophies_earned / players_tracked, is fitted linearly in x
//    and extrapolated to the target; the sum of the forecast medians must
//    equal 100 * m*. All non-fixed medians are shifted by one common logit
//    delta (damped 0.7x, clipped to +/-0.8) to meet it, preserving the
//    trophies' relative order while stopping a systematic level miss.
// 5. Spread. sigma^2 = 0.20^2 (base) + (0.45*sqrt(days))^2 (random walk)
//    + residual_E^2 + residual_N^2 + (se_bE * dx)^2 + (se_bN * dx)^2,
//    floored at 0.30 and capped at 2.0, then scaled by 0.85: the raw
//    formula over-covers (90% coverage came out 0.91-0.98 in simulation),
//    so the band is narrowed to sit near honest 90% coverage. Quantiles
//    are the normal quantiles on the logit scale. Completion trophies get
//    sigma >= 0.9.
// 6. Flat history (one distinct read: the practice cases, launch night).
//    No slope is estimable, but the one refresh on record cut the non-100%
//    rates to 0.221x of their launch-night values (log -1.51: 3,868 players
//    tracked and 4,276 trophies earned at 04:30Z, against 638 and 943 before,
//    so earners grew 1.34x while players grew 6.06x). So the median is
//    shifted down 1.45 logit (decaying with horizon, as real growth takes
//    over from dilution) with a 0.65-0.9 spread, instead of repeating the
//    stale rate with a narrow band the way the house baseline does (which
//    scores ~1.51 on the practice because the refresh lands outside it).
// 7. A trophy at 100% in every distinct read stays near the top:
//    [99.0, 99.8, 99.99, 99.99, 99.99]. Every tracked player has it, through
//    a 6x jump in players tracked, so it is the first mission everyone
//    completes, not a rate that can fall.
//
// Usage (local): mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
/** Release day start; the house curve baseline uses the same origin. */
const RELEASE_START_MS = Date.parse('2026-11-19T00:00:00Z');

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; release_date?: string };
  trophies: Trophy[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;

interface Fit {
  a: number;
  b: number;
  /** Standard error of the slope (Infinity when unidentified). */
  seB: number;
  /** Residual standard deviation. */
  s: number;
  n: number;
}

/** Weighted least squares of y on x; null when the slope is unidentified. */
function wls(xs: number[], ys: number[], ws: number[]): Fit | null {
  const n = xs.length;
  if (n < 2) return null;
  let sw = 0;
  let swx = 0;
  let swy = 0;
  let swxx = 0;
  let swxy = 0;
  for (let i = 0; i < n; i++) {
    const w = ws[i];
    sw += w;
    swx += w * xs[i];
    swy += w * ys[i];
    swxx += w * xs[i] * xs[i];
    swxy += w * xs[i] * ys[i];
  }
  const denom = sw * swxx - swx * swx;
  if (!Number.isFinite(denom) || Math.abs(denom) < 1e-12) return null;
  const b = (sw * swxy - swx * swy) / denom;
  const a = (swy - b * swx) / sw;
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    rss += ws[i] * r * r;
  }
  const dof = n - 2;
  const s2 = dof > 0 ? rss / dof / (sw / n) : 0;
  const s = Math.sqrt(Math.max(0, s2));
  const seB = dof > 0 ? Math.sqrt(Math.max(0, s2) / (swxx - (swx * swx) / sw)) : Infinity;
  return { a, b, seB, s, n };
}

/** Intercept refitted for a fixed slope (used when a fitted slope is clipped). */
function refitIntercept(xs: number[], ys: number[], ws: number[], b: number): number {
  let sw = 0;
  let s = 0;
  for (let i = 0; i < xs.length; i++) {
    sw += ws[i];
    s += ws[i] * (ys[i] - b * xs[i]);
  }
  return sw > 0 ? s / sw : 0;
}

function main(): void {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
  for (const file of inputs) {
    const id = file.slice(0, -'.in'.length);
    const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
    const trophies = c.trophies;
    const nT = trophies.length;
    const days = Math.max(0, c.horizon_minutes / 1440);

    // Time origin: the release date the case carries, else launch day.
    let t0 = RELEASE_START_MS;
    if (c.game && typeof c.game.release_date === 'string') {
      const p = Date.parse(`${c.game.release_date}T00:00:00Z`);
      if (Number.isFinite(p)) t0 = p;
    }
    const xOf = (iso: string): number => {
      const tHours = (Date.parse(iso) - t0) / 3600000;
      return Math.log(1 + Math.max(tHours, -0.9));
    };
    const xTarget = xOf(c.target);

    // Dedupe consecutive identical reads (same players, same rates) to the
    // first read showing each distinct page; keep first-appearance times.
    const hist = c.history ?? [];
    const distinct: Read[] = [];
    let prevKey = '';
    for (const r of hist) {
      const key = `${r.players_tracked}|${(r.rates ?? []).join(',')}`;
      if (key !== prevKey) distinct.push(r);
      prevKey = key;
    }
    const lastRead = hist[hist.length - 1];
    if (lastRead === undefined) throw new Error(`${file} has no history`);
    const lastRates: Array<number | null> = lastRead.rates ?? [];
    // x of the first appearance of the last distinct page (when the value
    // the anchor uses was actually fresh).
    const lastDistinct = distinct[distinct.length - 1] ?? lastRead;
    const xAnchor = xOf(lastDistinct.at);
    const dx = Math.max(0, xTarget - xAnchor);

    // Per-trophy series over distinct reads. Weights decay with age in x:
    // the launch-night regime (selected early syncers, pre-launch reviewers)
    // does not sit on the same trend as the last two days, and a fit over
    // the whole history would drag every slope toward the flat plateau and
    // the first refresh. Half-life ~0.83 in x (a factor ~2.3 in 1+t).
    const xs = distinct.map((r) => xOf(r.at));
    const xLastD = xs[xs.length - 1] ?? 0;
    const ws = distinct.map((_, j) => Math.max(0.05, Math.exp(-(xLastD - xs[j]) / 1.2)));

    // Players-tracked power law: ln N = aN + bN * x.
    const lnNs = distinct.map((r) => Math.log(Math.max(1, r.players_tracked)));
    const nFit = wls(xs, lnNs, ws);
    let bN = 0;
    let seBN = 1;
    let sN = 0;
    if (nFit) {
      bN = Math.max(0, Math.min(1.2, nFit.b));
      seBN = nFit.seB;
      sN = nFit.s;
    }
    const aN = refitIntercept(xs, lnNs, ws, bN);
    let nPred = Math.exp(aN + bN * xTarget);
    nPred = Math.min(Math.max(nPred, lastRead.players_tracked), lastRead.players_tracked * 50);

    interface PerTrophy {
      lastLogit: number;
      lastRate: number;
      eLast: number;
      bE: number;
      seBE: number;
      sE: number;
      fitN: number;
      isComplete: boolean;
      isFull: boolean;
      xs: number[];
      ys: number[];
      ws: number[];
    }
    const per: PerTrophy[] = trophies.map((t, i) => {
      const lastRate = typeof lastRates[i] === 'number' ? (lastRates[i] as number) : 0;
      const ys: number[] = [];
      const xsi: number[] = [];
      const wsi: number[] = [];
      distinct.forEach((r, j) => {
        const v = r.rates ? r.rates[i] : null;
        if (typeof v === 'number' && Number.isFinite(v) && v > 0) {
          const e = (v * r.players_tracked) / 100;
          if (e > 0) {
            ys.push(Math.log(e));
            xsi.push(xs[j]);
            wsi.push(ws[j]);
          }
        }
      });
      const fit = wls(xsi, ys, wsi);
      const isComplete =
        t.tier === 'platinum' || /100%\s*completion/i.test(t.description ?? '');
      const isFull = lastRate >= 99.99 && ys.length > 0;
      return {
        lastLogit: logit(lastRate),
        lastRate,
        eLast: (lastRate * lastRead.players_tracked) / 100,
        bE: fit ? fit.b : 0,
        seBE: fit ? fit.seB : Infinity,
        sE: fit ? fit.s : 0,
        fitN: ys.length,
        isComplete,
        isFull,
        xs: xsi,
        ys,
        ws: wsi,
      };
    });

    // Empirical-Bayes shrinkage of earner-slopes toward the cross-trophy mean.
    const fitted = per.filter((p) => p.fitN >= 2 && Number.isFinite(p.seBE) && p.seBE < 10);
    if (fitted.length >= 3) {
      let wSum = 0;
      let wbSum = 0;
      for (const p of fitted) {
        const w = 1 / (p.seBE * p.seBE + 1e-6);
        wSum += w;
        wbSum += w * p.bE;
      }
      const bBar = wbSum / wSum;
      const meanB = fitted.reduce((s, p) => s + p.bE, 0) / fitted.length;
      const tau = Math.min(2, Math.max(0.05, Math.sqrt(
        fitted.reduce((s, p) => s + (p.bE - meanB) * (p.bE - meanB), 0) / fitted.length,
      )));
      for (const p of per) {
        if (!Number.isFinite(p.seBE) || p.seBE >= 10) {
          p.bE = bBar;
          p.seBE = 1;
        } else {
          const wData = 1 / (p.seBE * p.seBE + 1e-6);
          const wPrior = 1 / (tau * tau);
          p.bE = (wData * p.bE + wPrior * bBar) / (wData + wPrior);
        }
      }
    } else {
      for (const p of per) if (!Number.isFinite(p.seBE)) p.seBE = 1;
    }
    for (const p of per) {
      p.bE = Math.max(-0.5, Math.min(3, p.bE));
      if (p.isComplete) p.bE = Math.min(p.bE, 0);
      if (!Number.isFinite(p.seBE)) p.seBE = 1;
    }
    // Recent-slope blend. A trophy's earner growth is convex in x while
    // players are still reaching it (takeoff after a delay, then saturation),
    // so the whole-history slope understates where it is heading and a
    // last-window slope alone would explode off the takeoff. Blend the two:
    // the recent slope (last 1.0 of x, at least 3 points) clipped to at most
    // +1.0 above the shrunk full-window slope, averaged 50/50.
    for (const p of per) {
      if (p.isFull || p.isComplete) continue;
      const rx: number[] = [];
      const ry: number[] = [];
      const rw: number[] = [];
      for (let j = 0; j < p.xs.length; j++) {
        if (p.xs[j] >= xLastD - 1.0) {
          rx.push(p.xs[j]);
          ry.push(p.ys[j]);
          rw.push(p.ws[j]);
        }
      }
      const rFit = rx.length >= 3 ? wls(rx, ry, rw) : null;
      if (rFit) {
        const bRec = Math.max(p.bE - 0.5, Math.min(p.bE + 1.0, rFit.b));
        p.bE = (p.bE + bRec) / 2;
      }
    }

    const flat = distinct.length <= 1;
    const medLogit: number[] = new Array(nT).fill(0);
    const sigma: number[] = new Array(nT).fill(0.8);
    // Dilution floor per trophy (earners do not decrease), re-applied after
    // the level shift below so the shift can never push a rate under it.
    const floorLogit: number[] = per.map((p) =>
      logit((100 * p.eLast) / Math.max(1, nPred)),
    );

    if (flat) {
      // No movement to fit: expect the next refresh to dilute rates the way
      // the only refresh on record did (non-100% rates to 0.221x), with a
      // spread covering how unevenly trophies dilute.
      const shift = -1.45 * Math.exp(-days / 3);
      const sig = 0.65 + 0.25 * Math.sqrt(days);
      for (let i = 0; i < nT; i++) {
        const p = per[i];
        if (p.isFull) {
          medLogit[i] = logit(99.95);
          sigma[i] = 0.35;
        } else {
          medLogit[i] = p.lastLogit + shift;
          sigma[i] = sig;
        }
      }
    } else {
      // Moving history: anchor on the last rate, extend by (bE - bN).
      for (let i = 0; i < nT; i++) {
        const p = per[i];
        if (p.isFull) {
          medLogit[i] = logit(99.95);
          sigma[i] = 0.35;
          continue;
        }
        const bRate = p.bE - bN;
        let m = p.lastLogit + bRate * dx;
        // Dilution floor: earners do not decrease, so the rate cannot fall
        // below E_last / N_target.
        if (m < floorLogit[i]) m = floorLogit[i];
        // Completion trophies never rise above their last rate.
        if (p.isComplete && m > p.lastLogit) m = p.lastLogit;
        medLogit[i] = m;
        const sETerm = Math.min(p.sE, 0.8);
        const sNTerm = Math.min(sN, 0.5);
        const seETerm = Math.min(p.seBE, 1) * dx;
        const seNTerm = Math.min(Number.isFinite(seBN) ? seBN : 1, 1) * dx;
        const v =
          0.2 * 0.2 +
          (0.45 * Math.sqrt(days)) ** 2 +
          sETerm * sETerm +
          sNTerm * sNTerm +
          seETerm * seETerm +
          seNTerm * seNTerm;
        let sig = 0.85 * Math.sqrt(Math.max(0.09, v));
        if (p.isComplete) sig = Math.max(sig, 0.9);
        sigma[i] = Math.min(2, Math.max(0.3, sig));
      }

      // Cross-trophy level anchor: mean trophies per player. m grows
      // multiplicatively (a sum of power laws), so extrapolate it two ways
      // -- linear in x and linear in ln m -- and average them; either alone
      // leans one way when the curve bends. Keep most of the fitted move.
      const mVals = distinct.map((r) =>
        r.players_tracked > 0 ? r.trophies_earned / r.players_tracked : 0,
      );
      const mFit = wls(xs, mVals, ws);
      const lnMVals = mVals.map((m) => Math.log(Math.max(0.05, m)));
      const lnMFit = wls(xs, lnMVals, ws);
      if (mFit && lnMFit && distinct.length >= 3) {
        const mLast = mVals[mVals.length - 1];
        const mLin = mFit.a + mFit.b * xTarget;
        const mLog = Math.exp(lnMFit.a + lnMFit.b * xTarget);
        let mStar = (mLin + mLog) / 2;
        mStar = mLast + 0.85 * (mStar - mLast);
        mStar = Math.min(Math.max(mStar, 0.3 * mLast), 4 * mLast);
        mStar = Math.max(mStar, 0.1);
        const sTarget = 100 * mStar;
        const fixedIdx = new Set<number>();
        let fixedSum = 0;
        for (let i = 0; i < nT; i++) {
          if (per[i].isFull || per[i].isComplete) {
            fixedIdx.add(i);
            fixedSum += invLogit(medLogit[i]);
          }
        }
        const sumAt = (delta: number): number => {
          let s = fixedSum;
          for (let i = 0; i < nT; i++) {
            if (!fixedIdx.has(i)) s += invLogit(medLogit[i] + delta);
          }
          return s;
        };
        let lo = -2;
        let hi = 2;
        if (sumAt(lo) > sTarget || sumAt(hi) < sTarget) {
          // Target outside the reachable range: take the closest end.
          const d = Math.abs(sumAt(lo) - sTarget) < Math.abs(sumAt(hi) - sTarget) ? lo : hi;
          const shift = Math.max(-0.5, Math.min(0.5, 0.5 * d));
          for (let i = 0; i < nT; i++) if (!fixedIdx.has(i)) medLogit[i] += shift;
        } else {
          for (let k = 0; k < 60; k++) {
            const mid = (lo + hi) / 2;
            if (sumAt(mid) < sTarget) lo = mid;
            else hi = mid;
          }
          const delta = (lo + hi) / 2;
          const shift = Math.max(-0.5, Math.min(0.5, 0.5 * delta));
          for (let i = 0; i < nT; i++) if (!fixedIdx.has(i)) medLogit[i] += shift;
        }
        // The level shift must not push a rate under its dilution floor or
        // a completion trophy above its last rate.
        for (let i = 0; i < nT; i++) {
          if (per[i].isFull) continue;
          if (medLogit[i] < floorLogit[i]) medLogit[i] = floorLogit[i];
          if (per[i].isComplete && medLogit[i] > per[i].lastLogit) {
            medLogit[i] = per[i].lastLogit;
          }
        }
      }
    }

    // Emit rows: medians capped, quantiles from the logit-normal band.
    const rows = trophies.map((t, i) => {
      const p = per[i];
      if (p.isFull) {
        return { id: t.id, q: [99, 99.8, 99.99, 99.99, 99.99] };
      }
      let med = invLogit(medLogit[i]);
      med = Math.min(99, Math.max(0.01, med));
      const lMed = logit(med);
      const sig = sigma[i];
      const q = Z.map((z) => round4(clampPct(invLogit(lMed + z * sig))));
      // Keep the row non-decreasing after rounding.
      for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
      return { id: t.id, q };
    });
    writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: rows })}\n`);
  }
}

main();
