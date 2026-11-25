"""Phase 3: Coverage-first Loto 6 ticket optimizer (Wheeling for 4th/5th prize hit rate).

Design priorities:
  1. Hit-rate maximization (4th/5th prize focus):
     - Given a pool of top M (10-15) predicted numbers from Phase 2, constructs B tickets
       (default B=5) that optimize number overlap via combinatorial & score-weighted wheeling.
     - Directly maximizes exact coverage of 3-hit and 4-hit scenarios inside the pool so that
       whenever 3 or 4 winning numbers fall within the top pool, at least one ticket reliably
       captures 5th prize (3+ matches) or 4th prize (4+ matches).
  2. Gentle, configurable structural filters:
     - Sum range (default 80..185, ~95% of real draws), odd/even range (1..5), and max
       consecutive run (<=4) are intentionally lenient so high-scoring numbers are adopted
       naturally, and can be tuned or disabled (`None`).
  3. Crowd-psychology penalty (DEFAULT OFF):
     - `use_crowd_penalty=False` by default to prioritize pure hit rate.
     - Can be toggled ON via config/UI to penalize birthday bias, arithmetic patterns,
       same last-digit clusters, and mark-sheet line patterns for expected-value play.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """Configuration for coverage-first wheeling ticket optimization."""

    pool_size: int = 12
    n_tickets: int = 5
    # Crowd psychology penalty: DEFAULT OFF (pure hit-rate priority)
    use_crowd_penalty: bool = False
    crowd_penalty_weight: float = 0.25
    # Gentle structural filters (set to None to disable)
    sum_range: Optional[tuple[int, int]] = (80, 185)
    odd_count_range: Optional[tuple[int, int]] = (1, 5)
    max_consecutive_run: Optional[int] = 4
    # Coverage optimization weights
    cov_4_for_3_weight: float = 0.45  # 4 hits in pool -> >=5th prize (3+ matches)
    cov_3_for_3_weight: float = 0.25  # 3 hits in pool -> >=5th prize (3 matches)
    cov_4_for_4_weight: float = 0.20  # 4 hits in pool -> >=4th prize (4 matches)
    ticket_score_weight: float = 0.15  # Favor higher Phase 2 score tickets
    overlap_penalty_weight: float = 0.08  # Discourage redundant 4-5 number overlap
    score_weighting_strength: float = 0.50  # 0.0=unweighted wheel, 1.0=strong score bias
    max_local_search_iters: int = 12


@dataclass
class OptimizationResult:
    """Optimized ticket portfolio and exact wheeling coverage metrics."""

    pool_numbers: list[int]
    tickets: list[tuple[int, int, int, int, int, int]]
    tickets_df: pd.DataFrame
    coverage_metrics: dict[str, float]
    number_usage: pd.Series
    config: OptimizerConfig


def _max_consecutive_run(sorted_nums: Sequence[int]) -> int:
    """Return length of the longest consecutive integer run in a sorted sequence."""
    if not sorted_nums:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(sorted_nums)):
        if sorted_nums[i] == sorted_nums[i - 1] + 1:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best


def compute_crowd_penalty(ticket: Sequence[int]) -> float:
    """Compute a normalized crowd-popularity penalty score (0.0 = unpopular/unique, ~1.0 = crowded).

    Penalizes patterns disproportionately chosen by human lottery players:
      - Birthday concentration (1..31, especially 1..12)
      - Long consecutive sequences (e.g., 1-2-3-4)
      - Constant-step arithmetic progressions / multiples of 5 or 7
      - Heavy clustering on the same last digit
      - Visual row/column concentration on a standard 7-column Loto 6 mark sheet
    """
    nums = sorted(int(x) for x in ticket)
    penalty = 0.0

    # 1. Birthday range bias (1..31)
    n_le31 = sum(1 for x in nums if x <= 31)
    if n_le31 == 6:
        penalty += 0.35
    elif n_le31 == 5:
        penalty += 0.15
    n_le12 = sum(1 for x in nums if x <= 12)
    if n_le12 >= 4:
        penalty += 0.15

    # 2. Consecutive runs
    max_run = _max_consecutive_run(nums)
    if max_run >= 4:
        penalty += 0.35
    elif max_run == 3:
        penalty += 0.15

    # 3. Arithmetic progression / multiples of 5 or 7
    diffs = [nums[i] - nums[i - 1] for i in range(1, len(nums))]
    if len(set(diffs)) == 1:
        penalty += 0.50  # Exact arithmetic sequence (e.g., 5,10,15,20,25,30)
    elif len(set(diffs)) == 2 and max(diffs) - min(diffs) <= 2:
        penalty += 0.15

    if sum(1 for x in nums if x % 5 == 0) >= 4 or sum(1 for x in nums if x % 7 == 0) >= 4:
        penalty += 0.25

    # 4. Same last-digit clustering
    last_digits = [x % 10 for x in nums]
    max_same_last = max(last_digits.count(d) for d in set(last_digits))
    if max_same_last >= 4:
        penalty += 0.25
    elif max_same_last == 3:
        penalty += 0.08

    # 5. Mark-sheet row/column alignment (7 numbers per column layout)
    rows = [(x - 1) // 7 for x in nums]
    cols = [(x - 1) % 7 for x in nums]
    if max(rows.count(r) for r in set(rows)) >= 4 or max(cols.count(c) for c in set(cols)) >= 4:
        penalty += 0.15

    return float(min(1.5, penalty))


def _subset_weights(
    subsets: np.ndarray,
    pool_scores: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Compute normalized subset likelihood weights w(S) proportional to prod_{i in S} p_i^strength."""
    if strength <= 0.0:
        return np.full(len(subsets), 1.0 / len(subsets), dtype=np.float64)
    norm_scores = np.clip(pool_scores / np.mean(pool_scores), 1e-4, None)
    log_s = np.log(norm_scores) * float(strength)
    log_w = log_s[subsets].sum(axis=1)
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    return w / w.sum()


def _passes_filters(ticket: Sequence[int], cfg: OptimizerConfig) -> bool:
    """Check whether a 6-number ticket satisfies the configured gentle structural filters."""
    if cfg.sum_range is not None:
        s = sum(ticket)
        if not (cfg.sum_range[0] <= s <= cfg.sum_range[1]):
            return False
    if cfg.odd_count_range is not None:
        n_odd = sum(1 for x in ticket if x % 2 == 1)
        if not (cfg.odd_count_range[0] <= n_odd <= cfg.odd_count_range[1]):
            return False
    if cfg.max_consecutive_run is not None:
        if _max_consecutive_run(ticket) > cfg.max_consecutive_run:
            return False
    return True


def evaluate_portfolio_coverage(
    tickets: Sequence[Sequence[int]],
    pool_numbers: Sequence[int],
    score_map: Optional[dict[int, float]] = None,
    score_weighting_strength: float = 0.50,
) -> dict[str, float]:
    """Compute exact combinatorial & score-weighted coverage metrics for a ticket portfolio.

    Evaluates what fraction of all possible 3-hit, 4-hit, and 5-hit winning subsets inside
    `pool_numbers` are guaranteed to win 5th prize (3+ matches) or 4th prize (4+ matches)
    by at least one ticket in `tickets`.
    """
    pool = [int(x) for x in pool_numbers]
    m = len(pool)
    pool_idx = {num: idx for idx, num in enumerate(pool)}
    pool_scores = np.array(
        [float(score_map.get(num, 6.0 / 43.0)) if score_map else (6.0 / 43.0) for num in pool],
        dtype=np.float64,
    )

    n_tickets = len(tickets)
    ticket_mat = np.zeros((n_tickets, m), dtype=np.uint8)
    for t_idx, t in enumerate(tickets):
        for num in t:
            if int(num) in pool_idx:
                ticket_mat[t_idx, pool_idx[int(num)]] = 1

    # Helper to compute unweighted & weighted coverage for k-subsets at match threshold r
    def _cov_stats(k: int, req_match: int) -> tuple[float, float]:
        if m < k or n_tickets == 0:
            return 0.0, 0.0
        subs = np.array(list(itertools.combinations(range(m), k)), dtype=np.int32)
        sub_mat = np.zeros((len(subs), m), dtype=np.uint8)
        sub_mat[np.repeat(np.arange(len(subs)), k), subs.ravel()] = 1
        matches = ticket_mat @ sub_mat.T  # (n_tickets, n_subs)
        covered = (matches.max(axis=0) >= req_match).astype(np.float64)
        w = _subset_weights(subs, pool_scores, score_weighting_strength)
        return float(covered.mean()), float((covered * w).sum())

    cov3_5th, cov3_5th_w = _cov_stats(3, 3)
    cov4_5th, cov4_5th_w = _cov_stats(4, 3)
    cov4_4th, cov4_4th_w = _cov_stats(4, 4)
    cov5_4th, _ = _cov_stats(5, 4)
    cov5_5th, _ = _cov_stats(5, 3)

    # Pairwise overlap statistics across tickets
    overlaps = []
    for i in range(n_tickets):
        s_i = set(tickets[i])
        for j in range(i + 1, n_tickets):
            overlaps.append(len(s_i.intersection(tickets[j])))
    avg_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    max_overlap = float(np.max(overlaps)) if overlaps else 0.0

    usage = ticket_mat.sum(axis=0)
    crowd_scores = [compute_crowd_penalty(t) for t in tickets]

    return {
        "cov_3_hits_ge_5th": cov3_5th,
        "cov_3_hits_ge_5th_weighted": cov3_5th_w,
        "cov_4_hits_ge_5th": cov4_5th,
        "cov_4_hits_ge_5th_weighted": cov4_5th_w,
        "cov_4_hits_ge_4th": cov4_4th,
        "cov_4_hits_ge_4th_weighted": cov4_4th_w,
        "cov_5_hits_ge_5th": cov5_5th,
        "cov_5_hits_ge_4th": cov5_4th,
        "avg_pairwise_overlap": avg_overlap,
        "max_pairwise_overlap": max_overlap,
        "pool_numbers_used": float((usage > 0).sum()),
        "pool_size": float(m),
        "min_number_usage": float(usage.min()) if m > 0 else 0.0,
        "max_number_usage": float(usage.max()) if m > 0 else 0.0,
        "avg_crowd_penalty": float(np.mean(crowd_scores)) if crowd_scores else 0.0,
    }


def optimize_tickets(
    number_scores_df: pd.DataFrame,
    config: Optional[OptimizerConfig] = None,
) -> OptimizationResult:
    """Optimize B Loto 6 tickets from the top M predicted numbers via wheeling coverage.

    Uses a two-phase exact combinatorial optimizer:
      Phase A: Greedy marginal-gain selection maximizing weighted 4-hit & 3-hit coverage.
      Phase B: Vectorized 1-opt coordinate-ascent refinement until convergence.

    Args:
        number_scores_df: Phase 2 `PredictionResult.number_scores` DataFrame containing
            at least `number` and `final_score` columns.
        config: Optional `OptimizerConfig`.
    """
    cfg = config or OptimizerConfig()
    if not (6 <= cfg.pool_size <= 16):
        raise ValueError("pool_size は 6〜16 の範囲で指定してください（推奨: 10〜15）。")
    if cfg.n_tickets <= 0:
        raise ValueError("n_tickets は 1 以上を指定してください。")

    required = {"number", "final_score"}
    if not required.issubset(number_scores_df.columns):
        raise ValueError(f"number_scores_df に必須カラム {required} が不足しています。")

    ranked = (
        number_scores_df[["number", "final_score"]]
        .drop_duplicates(subset=["number"])
        .sort_values(["final_score", "number"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if len(ranked) < cfg.pool_size:
        raise ValueError(f"数字数が pool_size={cfg.pool_size} より少ないです（{len(ranked)}個）。")

    pool_df = ranked.head(cfg.pool_size)
    pool_numbers = pool_df["number"].astype(int).tolist()
    pool_scores = pool_df["final_score"].to_numpy(dtype=np.float64)
    score_map = dict(zip(pool_numbers, pool_scores))

    m = len(pool_numbers)

    # Generate candidate 6-number tickets from the top M pool (index representation 0..m-1)
    all_idx_combos = list(itertools.combinations(range(m), 6))
    if len(all_idx_combos) < cfg.n_tickets:
        raise ValueError(
            f"pool_size={m} から作成可能な組み合わせ数 ({len(all_idx_combos)}) が "
            f"n_tickets={cfg.n_tickets} より少ないです。"
        )

    filtered_idx_combos = [
        idx_c for idx_c in all_idx_combos if _passes_filters(sorted(pool_numbers[i] for i in idx_c), cfg)
    ]
    filters_relaxed = False
    if len(filtered_idx_combos) < cfg.n_tickets:
        logger.warning(
            "構造フィルタ後の候補数 (%d) が n_tickets (%d) 未満のためフィルタを自動緩和します。",
            len(filtered_idx_combos),
            cfg.n_tickets,
        )
        cand_idx_combos = all_idx_combos
        filters_relaxed = True
    else:
        cand_idx_combos = filtered_idx_combos

    n_cand = len(cand_idx_combos)
    cand_arr = np.array(cand_idx_combos, dtype=np.int32)  # (n_cand, 6)

    # Binary candidate matrix (n_cand, m)
    cand_bin = np.zeros((n_cand, m), dtype=np.uint8)
    cand_bin[np.repeat(np.arange(n_cand), 6), cand_arr.ravel()] = 1

    # Normalized ticket score (0..1 range across candidates)
    raw_ticket_scores = pool_scores[cand_arr].sum(axis=1)
    score_span = float(raw_ticket_scores.max() - raw_ticket_scores.min())
    if score_span > 1e-12:
        norm_ticket_scores = (raw_ticket_scores - raw_ticket_scores.min()) / score_span
    else:
        norm_ticket_scores = np.ones(n_cand, dtype=np.float64)

    # Crowd psychology penalty (computed for reporting; applied ONLY if use_crowd_penalty=True)
    crowd_penalties = np.array(
        [compute_crowd_penalty(sorted(pool_numbers[i] for i in idx_c)) for idx_c in cand_idx_combos],
        dtype=np.float64,
    )

    # Precompute exact subset coverage matrices for 3-subsets and 4-subsets
    subs3 = np.array(list(itertools.combinations(range(m), 3)), dtype=np.int32)
    sub3_bin = np.zeros((len(subs3), m), dtype=np.uint8)
    sub3_bin[np.repeat(np.arange(len(subs3)), 3), subs3.ravel()] = 1
    cov3_mat = (cand_bin @ sub3_bin.T) >= 3  # (n_cand, n_sub3) bool
    w3 = _subset_weights(subs3, pool_scores, cfg.score_weighting_strength)

    subs4 = np.array(list(itertools.combinations(range(m), 4)), dtype=np.int32)
    sub4_bin = np.zeros((len(subs4), m), dtype=np.uint8)
    sub4_bin[np.repeat(np.arange(len(subs4)), 4), subs4.ravel()] = 1
    match4 = cand_bin @ sub4_bin.T  # (n_cand, n_sub4) uint8
    cov4_ge3_mat = match4 >= 3
    cov4_eq4_mat = match4 >= 4
    w4 = _subset_weights(subs4, pool_scores, cfg.score_weighting_strength)

    # Per-candidate intrinsic score bonus/penalty
    intrinsic = (cfg.ticket_score_weight / cfg.n_tickets) * norm_ticket_scores
    if cfg.use_crowd_penalty:
        intrinsic = intrinsic - (cfg.crowd_penalty_weight / cfg.n_tickets) * crowd_penalties

    # Precompute overlap penalty scaling denominator
    max_pairs = max(1, cfg.n_tickets * (cfg.n_tickets - 1) // 2)

    def _candidate_marginal_values(
        cur_selected: list[int],
    ) -> np.ndarray:
        """Vectorized marginal objective gain for every candidate given `cur_selected`."""
        if not cur_selected:
            uncov3 = np.ones(len(subs3), dtype=bool)
            uncov4_3 = np.ones(len(subs4), dtype=bool)
            uncov4_4 = np.ones(len(subs4), dtype=bool)
            overlap_pen = np.zeros(n_cand, dtype=np.float64)
        else:
            sel = np.array(cur_selected, dtype=np.int32)
            uncov3 = ~cov3_mat[sel].any(axis=0)
            uncov4_3 = ~cov4_ge3_mat[sel].any(axis=0)
            uncov4_4 = ~cov4_eq4_mat[sel].any(axis=0)
            # Pairwise overlap with already-selected tickets (penalize >=4 shared numbers quadratically)
            ov = (cand_bin @ cand_bin[sel].T).astype(np.float64)  # (n_cand, len(sel))
            # Soft penalty on squared overlap above 2 numbers
            excess = np.maximum(0.0, ov - 2.0) ** 2 / 16.0
            overlap_pen = (cfg.overlap_penalty_weight / max_pairs) * excess.sum(axis=1)

        gain_4_3 = cov4_ge3_mat[:, uncov4_3] @ w4[uncov4_3] if uncov4_3.any() else 0.0
        gain_3_3 = cov3_mat[:, uncov3] @ w3[uncov3] if uncov3.any() else 0.0
        gain_4_4 = cov4_eq4_mat[:, uncov4_4] @ w4[uncov4_4] if uncov4_4.any() else 0.0

        values = (
            cfg.cov_4_for_3_weight * gain_4_3
            + cfg.cov_3_for_3_weight * gain_3_3
            + cfg.cov_4_for_4_weight * gain_4_4
            + intrinsic
            - overlap_pen
        )
        if cur_selected:
            values[cur_selected] = -1e9  # prevent duplicate tickets
        return values

    # Phase A: Greedy initialization
    selected: list[int] = []
    for _ in range(cfg.n_tickets):
        vals = _candidate_marginal_values(selected)
        best_c = int(np.argmax(vals))
        selected.append(best_c)

    # Phase B: 1-opt coordinate-ascent local search refinement
    for _ in range(cfg.max_local_search_iters):
        improved = False
        for slot in range(cfg.n_tickets):
            cur_c = selected[slot]
            others = [selected[i] for i in range(cfg.n_tickets) if i != slot]
            vals = _candidate_marginal_values(others)
            best_c = int(np.argmax(vals))
            if best_c != cur_c and vals[best_c] > vals[cur_c] + 1e-12:
                selected[slot] = best_c
                improved = True
        if not improved:
            break

    # Build final sorted tickets & summary DataFrame
    tickets: list[tuple[int, int, int, int, int, int]] = []
    ticket_rows = []
    for t_idx, c_idx in enumerate(selected, start=1):
        nums = tuple(sorted(pool_numbers[i] for i in cand_idx_combos[c_idx]))
        tickets.append(nums)
        n_odd = sum(1 for x in nums if x % 2 == 1)
        ticket_rows.append(
            {
                "ticket_id": t_idx,
                "num1": nums[0],
                "num2": nums[1],
                "num3": nums[2],
                "num4": nums[3],
                "num5": nums[4],
                "num6": nums[5],
                "numbers_str": "-".join(f"{x:02d}" for x in nums),
                "sum_value": int(sum(nums)),
                "odd_count": int(n_odd),
                "even_count": int(6 - n_odd),
                "max_consecutive": int(_max_consecutive_run(nums)),
                "score_sum": float(sum(score_map[x] for x in nums)),
                "crowd_penalty": float(crowd_penalties[c_idx]),
            }
        )

    tickets_df = pd.DataFrame(ticket_rows).sort_values("score_sum", ascending=False).reset_index(drop=True)
    tickets_df["ticket_id"] = np.arange(1, len(tickets_df) + 1, dtype=int)
    tickets = [
        (
            int(r.num1),
            int(r.num2),
            int(r.num3),
            int(r.num4),
            int(r.num5),
            int(r.num6),
        )
        for r in tickets_df.itertuples(index=False)
    ]

    # Exact coverage evaluation
    metrics = evaluate_portfolio_coverage(
        tickets=tickets,
        pool_numbers=pool_numbers,
        score_map=score_map,
        score_weighting_strength=cfg.score_weighting_strength,
    )
    metrics["filters_relaxed"] = float(filters_relaxed)

    # Per-number usage counts across the selected tickets
    usage_counts = {num: 0 for num in pool_numbers}
    for t in tickets:
        for num in t:
            usage_counts[num] += 1
    number_usage = pd.Series(usage_counts, name="usage_count")

    return OptimizationResult(
        pool_numbers=pool_numbers,
        tickets=tickets,
        tickets_df=tickets_df,
        coverage_metrics=metrics,
        number_usage=number_usage,
        config=cfg,
    )


def generate_random_tickets(
    n_tickets: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> list[tuple[int, int, int, int, int, int]]:
    """Generate `n_tickets` distinct uniform random Loto 6 tickets (for benchmarking)."""
    gen = rng or np.random.default_rng()
    seen: set[tuple[int, int, int, int, int, int]] = set()
    tickets: list[tuple[int, int, int, int, int, int]] = []
    while len(tickets) < n_tickets:
        picked = tuple(sorted(int(x) for x in gen.choice(np.arange(1, 44), size=6, replace=False)))
        if picked not in seen:
            seen.add(picked)
            tickets.append(picked)
    return tickets


if __name__ == "__main__":
    from src.model import Loto6PredictionEngine, PredictionConfig
    from src.scraper import DEFAULT_OUTPUT_PATH, load_history_csv

    history = load_history_csv(DEFAULT_OUTPUT_PATH)
    pred = Loto6PredictionEngine(PredictionConfig()).fit_predict_next(history)
    opt = optimize_tickets(pred.number_scores, OptimizerConfig(pool_size=12, n_tickets=5))

    print(f"対象開催回: 第{pred.target_draw_no}回")
    print(f"採用上位プール ({len(opt.pool_numbers)}数字): {opt.pool_numbers}")
    print(f"大衆心理ペナルティ: {'ON' if opt.config.use_crowd_penalty else 'OFF'}")
    print("\n--- 最適化買い目 (5口) ---")
    print(
        opt.tickets_df[
            ["ticket_id", "numbers_str", "sum_value", "odd_count", "even_count", "score_sum", "crowd_penalty"]
        ].to_string(index=False)
    )
    print("\n--- カバレッジ指標（プール内的中時の4〜5等拾い切り率） ---")
    m = opt.coverage_metrics
    print(f"  プール内3個的中 -> 5等以上カバー率: {m['cov_3_hits_ge_5th']*100:.1f}% (スコア重み付き: {m['cov_3_hits_ge_5th_weighted']*100:.1f}%)")
    print(f"  プール内4個的中 -> 5等以上カバー率: {m['cov_4_hits_ge_5th']*100:.1f}% (スコア重み付き: {m['cov_4_hits_ge_5th_weighted']*100:.1f}%)")
    print(f"  プール内4個的中 -> 4等以上カバー率: {m['cov_4_hits_ge_4th']*100:.1f}% (スコア重み付き: {m['cov_4_hits_ge_4th_weighted']*100:.1f}%)")
    print(f"  プール内5個的中 -> 4等以上カバー率: {m['cov_5_hits_ge_4th']*100:.1f}%")
    print(f"  買い目間の平均重複数字数: {m['avg_pairwise_overlap']:.2f} 個 (最大: {int(m['max_pairwise_overlap'])} 個)")
    print(f"  プール数字の使用回数: { {int(k): int(v) for k, v in opt.number_usage.items()} }")
