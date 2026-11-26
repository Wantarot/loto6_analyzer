"""Strict leak-free feature engineering for Loto 6 draw prediction.

Data-leakage prevention guarantee:
  For every historical draw N and candidate number k (1..43), ALL features are
  computed strictly from draws 1..(N-1). Neither draw N's winning numbers nor any
  future draw (> N) is ever accessed when computing features for draw N.

Features generated per (draw_no=N, number=k):
  - Set-ball appearance rate & deviation from global average (6/43) and from the
    number's own overall appearance rate, referenced against either:
      * "prev"  (default, 100% leak-free): set ball used in draw N-1
      * "fixed" (100% leak-free): user-specified candidate set ball A-J
      * "actual" (opt-in for Phase 2 conditional training P(y_{N,k}=1 | set_ball_N=s)):
        set ball of draw N, still computing all counts strictly from draws 1..(N-1)
  - Interval: consecutive missed draws immediately preceding draw N (0 if drawn in N-1)
  - Recent appearance counts over the last 5, 10, and 30 draws (N-w .. N-1)
  - Pull (引っ張り) flag: whether number k appeared in draw N-1
  - Slide (スライド ±1) flags: whether k-1 or k+1 appeared in draw N-1
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd

from src.scraper import MAIN_NUM_COLS, VALID_SET_BALLS, validate_history_df

NUM_MIN = 1
NUM_MAX = 43
N_NUMBERS = 43
N_MAIN_DRAW = 6
GLOBAL_MEAN_RATE = N_MAIN_DRAW / N_NUMBERS  # 6 / 43 ≈ 0.139535

SET_BALL_TO_IDX = {sb: idx for idx, sb in enumerate(VALID_SET_BALLS)}
N_SET_BALLS = len(VALID_SET_BALLS)

RECENT_WINDOWS = (5, 10, 24, 30)
SET_BALL_RECENT_WINDOWS = (30, 50)

_NUMBERS_1_TO_43 = np.arange(NUM_MIN, NUM_MAX + 1, dtype=np.int32)
_LAST_DIGITS = _NUMBERS_1_TO_43 % 10
_DECADES = _NUMBERS_1_TO_43 // 10  # 0:1..9, 1:10..19, 2:20..29, 3:30..39, 4:40..43
_LAST_DIGIT_COUNTS = np.array([(int((_LAST_DIGITS == d).sum())) for d in range(10)], dtype=np.float64)

FEATURE_COLS = [
    "overall_appearance_rate",
    "set_ball_past_draws",
    "set_ball_appearance_rate",
    "set_ball_recent_30_rate",
    "set_ball_recent_50_rate",
    "set_ball_rate_diff",
    "set_ball_rate_diff_vs_num_avg",
    "interval",
    "recent_5_count",
    "recent_10_count",
    "recent_24_count",
    "recent_30_count",
    "is_pull",
    "is_lag2_repeat",
    "is_slide",
    "is_slide_minus1",
    "is_slide_plus1",
    "is_prev_consec_pull",
    "is_prev_consec_slide",
    "last_digit_recent_24_rate",
    "last_digit_gap",
    "prev_decade_count",
]

FLOAT_FEATURE_COLS = {
    "overall_appearance_rate",
    "set_ball_appearance_rate",
    "set_ball_recent_30_rate",
    "set_ball_recent_50_rate",
    "set_ball_rate_diff",
    "set_ball_rate_diff_vs_num_avg",
    "last_digit_recent_24_rate",
}

SetBallMode = Literal["prev", "fixed", "actual"]


def _build_appearance_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build (T, 43) binary appearance matrix for main winning numbers (1..43)."""
    n_draws = len(df)
    mat = np.zeros((n_draws, N_NUMBERS), dtype=np.int32)
    main_nums = df[MAIN_NUM_COLS].to_numpy(dtype=np.int32)
    row_idx = np.repeat(np.arange(n_draws), N_MAIN_DRAW)
    col_idx = (main_nums - 1).ravel()
    mat[row_idx, col_idx] = 1
    return mat


def _compute_prefix_states(
    appear_mat: np.ndarray,
    set_ball_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """Compute strictly-before-i cumulative states for indices i = 0..T.

    Index i represents the state AFTER draws 0..(i-1) and strictly BEFORE draw i.
    Index T represents the state after ALL T historical draws (for predicting draw T+1).

    Returns:
        cum_all:        (T+1, 43) cumulative appearances across all set balls before i
        cum_sb_appear:  (T+1, 10, 43) cumulative appearances per set ball before i
        cum_sb_draws:   (T+1, 10) cumulative draw counts per set ball before i
        intervals:      (T+1, 43) consecutive miss counts strictly before i
        sb_seq_cum:     list of 10 arrays, each (n_s + 1, 43) cumulative appearances
                        indexed by that set ball's own occurrence sequence (for O(1) recent W)
        tail_intervals: (T+1, 10) consecutive miss counts for each last digit 0..9 before i
    """
    n_draws = appear_mat.shape[0]

    # Prefix sum shifted by 1: cum_all[i] = sum(appear_mat[0:i])
    cum_all = np.zeros((n_draws + 1, N_NUMBERS), dtype=np.int32)
    cum_all[1:] = np.cumsum(appear_mat, axis=0, dtype=np.int32)

    # Per-set-ball cumulative appearances & draw counts shifted by 1
    cum_sb_appear = np.zeros((n_draws + 1, N_SET_BALLS, N_NUMBERS), dtype=np.int32)
    cum_sb_draws = np.zeros((n_draws + 1, N_SET_BALLS), dtype=np.int32)
    for t in range(n_draws):
        cum_sb_appear[t + 1] = cum_sb_appear[t]
        cum_sb_draws[t + 1] = cum_sb_draws[t]
        sb_idx = set_ball_indices[t]
        cum_sb_appear[t + 1, sb_idx] += appear_mat[t]
        cum_sb_draws[t + 1, sb_idx] += 1

    # Per-set-ball sequence prefix sums for O(1) recent-W window rates
    sb_seq_cum: list[np.ndarray] = []
    for s in range(N_SET_BALLS):
        sub = appear_mat[set_ball_indices == s]
        seq_c = np.zeros((len(sub) + 1, N_NUMBERS), dtype=np.int32)
        if len(sub) > 0:
            seq_c[1:] = np.cumsum(sub, axis=0, dtype=np.int32)
        sb_seq_cum.append(seq_c)

    # Consecutive miss intervals strictly before draw i (per number & per last digit 0..9)
    intervals = np.zeros((n_draws + 1, N_NUMBERS), dtype=np.int32)
    tail_intervals = np.zeros((n_draws + 1, 10), dtype=np.int32)
    cur_interval = np.zeros(N_NUMBERS, dtype=np.int32)
    cur_tail_interval = np.zeros(10, dtype=np.int32)
    for t in range(n_draws):
        intervals[t] = cur_interval
        tail_intervals[t] = cur_tail_interval
        appeared = appear_mat[t] == 1
        cur_interval = np.where(appeared, 0, cur_interval + 1)
        tail_appeared = np.array([bool(appeared[_LAST_DIGITS == d].any()) for d in range(10)])
        cur_tail_interval = np.where(tail_appeared, 0, cur_tail_interval + 1)
    intervals[n_draws] = cur_interval
    tail_intervals[n_draws] = cur_tail_interval

    return cum_all, cum_sb_appear, cum_sb_draws, intervals, sb_seq_cum, tail_intervals


def _resolve_ref_set_balls(
    set_balls: np.ndarray,
    set_ball_mode: SetBallMode,
    target_set_ball: Optional[str],
) -> list[Optional[str]]:
    """Determine the reference set ball for each historical draw index i=0..T-1."""
    n_draws = len(set_balls)
    if set_ball_mode == "fixed":
        if target_set_ball is None or target_set_ball.upper() not in SET_BALL_TO_IDX:
            raise ValueError("set_ball_mode='fixed' の場合は target_set_ball に A〜J を指定してください。")
        sb = target_set_ball.upper()
        return [sb] * n_draws

    if set_ball_mode == "prev":
        # Draw 0 has no previous draw; draw i >= 1 references draw i-1's set ball.
        return [None, *[str(sb) for sb in set_balls[:-1]]]

    if set_ball_mode == "actual":
        # Opt-in only for Phase 2 conditional training P(y_{N,k}=1 | set_ball_N=s).
        return [str(sb) for sb in set_balls]

    raise ValueError(f"未対応の set_ball_mode です: {set_ball_mode}")


def _extract_features_for_step(
    step_idx: int,
    appear_mat: np.ndarray,
    cum_all: np.ndarray,
    cum_sb_appear: np.ndarray,
    cum_sb_draws: np.ndarray,
    intervals: np.ndarray,
    sb_seq_cum: list[np.ndarray],
    tail_intervals: np.ndarray,
    ref_set_ball: Optional[str],
) -> dict[str, np.ndarray]:
    """Extract all 43-element leak-free feature arrays for a single step index `step_idx`."""
    # 1. Overall appearance rate over draws 0 .. step_idx-1
    if step_idx > 0:
        overall_rate = cum_all[step_idx].astype(np.float64) / float(step_idx)
    else:
        overall_rate = np.full(N_NUMBERS, np.nan, dtype=np.float64)

    # 2. Set-ball-specific appearance rates (all-time + recent 30/50 of that set ball)
    if ref_set_ball is not None:
        sb_idx = SET_BALL_TO_IDX[ref_set_ball]
        sb_draws = int(cum_sb_draws[step_idx, sb_idx])
        if sb_draws > 0:
            sb_rate = cum_sb_appear[step_idx, sb_idx].astype(np.float64) / float(sb_draws)
            sb_diff_global = sb_rate - GLOBAL_MEAN_RATE
            sb_diff_vs_num = sb_rate - overall_rate
            seq_c = sb_seq_cum[sb_idx]
            w30 = min(sb_draws, 30)
            w50 = min(sb_draws, 50)
            sb_rec30_rate = (seq_c[sb_draws] - seq_c[sb_draws - w30]).astype(np.float64) / float(w30)
            sb_rec50_rate = (seq_c[sb_draws] - seq_c[sb_draws - w50]).astype(np.float64) / float(w50)
        else:
            sb_rate = np.full(N_NUMBERS, np.nan, dtype=np.float64)
            sb_rec30_rate = np.full(N_NUMBERS, np.nan, dtype=np.float64)
            sb_rec50_rate = np.full(N_NUMBERS, np.nan, dtype=np.float64)
            sb_diff_global = np.full(N_NUMBERS, np.nan, dtype=np.float64)
            sb_diff_vs_num = np.full(N_NUMBERS, np.nan, dtype=np.float64)
    else:
        sb_draws = 0
        sb_rate = np.full(N_NUMBERS, np.nan, dtype=np.float64)
        sb_rec30_rate = np.full(N_NUMBERS, np.nan, dtype=np.float64)
        sb_rec50_rate = np.full(N_NUMBERS, np.nan, dtype=np.float64)
        sb_diff_global = np.full(N_NUMBERS, np.nan, dtype=np.float64)
        sb_diff_vs_num = np.full(N_NUMBERS, np.nan, dtype=np.float64)

    # 3. Interval (consecutive misses strictly before step_idx)
    interval_vec = intervals[step_idx]

    # 4. Recent 5 / 10 / 24 / 30 draw appearance counts (draws max(0, step_idx-w) .. step_idx-1)
    recent_5 = cum_all[step_idx] - cum_all[max(0, step_idx - 5)]
    recent_10 = cum_all[step_idx] - cum_all[max(0, step_idx - 10)]
    recent_24 = cum_all[step_idx] - cum_all[max(0, step_idx - 24)]
    recent_30 = cum_all[step_idx] - cum_all[max(0, step_idx - 30)]

    # 5. Pull (引っ張り), Lag-2 Repeat (隔回復活), Slide (±1), and Prev-Consecutive signals
    if step_idx > 0:
        prev = appear_mat[step_idx - 1]
        is_pull = prev.copy()
        is_slide_minus1 = np.zeros(N_NUMBERS, dtype=np.int32)
        is_slide_plus1 = np.zeros(N_NUMBERS, dtype=np.int32)
        is_slide_minus1[1:] = prev[:-1]  # number k has k-1 in previous draw
        is_slide_plus1[:-1] = prev[1:]   # number k has k+1 in previous draw
        is_slide = np.maximum(is_slide_minus1, is_slide_plus1)

        # Previous draw consecutive pair membership & adjacent slides
        has_left_adj = np.zeros(N_NUMBERS, dtype=np.int32)
        has_right_adj = np.zeros(N_NUMBERS, dtype=np.int32)
        has_left_adj[1:] = prev[:-1]
        has_right_adj[:-1] = prev[1:]
        in_prev_consec = (prev == 1) & ((has_left_adj == 1) | (has_right_adj == 1))
        is_prev_consec_pull = in_prev_consec.astype(np.int32)

        consec_left = np.zeros(N_NUMBERS, dtype=bool)
        consec_right = np.zeros(N_NUMBERS, dtype=bool)
        consec_left[1:] = in_prev_consec[:-1]
        consec_right[:-1] = in_prev_consec[1:]
        is_prev_consec_slide = ((prev == 0) & (consec_left | consec_right)).astype(np.int32)

        # Previous draw decade density for each number's decade
        dec_counts = np.bincount(_DECADES, weights=prev, minlength=5).astype(np.int32)
        prev_decade_count = dec_counts[_DECADES]
    else:
        is_pull = np.zeros(N_NUMBERS, dtype=np.int32)
        is_slide_minus1 = np.zeros(N_NUMBERS, dtype=np.int32)
        is_slide_plus1 = np.zeros(N_NUMBERS, dtype=np.int32)
        is_slide = np.zeros(N_NUMBERS, dtype=np.int32)
        is_prev_consec_pull = np.zeros(N_NUMBERS, dtype=np.int32)
        is_prev_consec_slide = np.zeros(N_NUMBERS, dtype=np.int32)
        prev_decade_count = np.zeros(N_NUMBERS, dtype=np.int32)

    if step_idx >= 2:
        is_lag2_repeat = appear_mat[step_idx - 2].copy()
    else:
        is_lag2_repeat = np.zeros(N_NUMBERS, dtype=np.int32)

    # 6. Last-digit (下一桁) recent-24 normalized rate & consecutive miss gap
    w24 = min(step_idx, 24)
    if w24 > 0:
        tail_sums = np.bincount(_LAST_DIGITS, weights=recent_24, minlength=10)
        tail_rates = tail_sums / (_LAST_DIGIT_COUNTS * float(w24))
        last_digit_recent_24_rate = tail_rates[_LAST_DIGITS]
    else:
        last_digit_recent_24_rate = np.full(N_NUMBERS, np.nan, dtype=np.float64)
    last_digit_gap = tail_intervals[step_idx, _LAST_DIGITS].astype(np.int32)

    return {
        "overall_appearance_rate": overall_rate,
        "set_ball_past_draws": np.full(N_NUMBERS, sb_draws, dtype=np.int32),
        "set_ball_appearance_rate": sb_rate,
        "set_ball_recent_30_rate": sb_rec30_rate,
        "set_ball_recent_50_rate": sb_rec50_rate,
        "set_ball_rate_diff": sb_diff_global,
        "set_ball_rate_diff_vs_num_avg": sb_diff_vs_num,
        "interval": interval_vec,
        "recent_5_count": recent_5,
        "recent_10_count": recent_10,
        "recent_24_count": recent_24,
        "recent_30_count": recent_30,
        "is_pull": is_pull,
        "is_lag2_repeat": is_lag2_repeat,
        "is_slide": is_slide,
        "is_slide_minus1": is_slide_minus1,
        "is_slide_plus1": is_slide_plus1,
        "is_prev_consec_pull": is_prev_consec_pull,
        "is_prev_consec_slide": is_prev_consec_slide,
        "last_digit_recent_24_rate": last_digit_recent_24_rate,
        "last_digit_gap": last_digit_gap,
        "prev_decade_count": prev_decade_count,
    }


def generate_features(
    history_df: pd.DataFrame,
    set_ball_mode: SetBallMode = "prev",
    target_set_ball: Optional[str] = None,
    min_past_draws: int = 0,
) -> pd.DataFrame:
    """Generate leak-free features for every (draw_no=N, number=1..43) pair.

    Every feature for draw N uses strictly draws 1..(N-1) only.

    Args:
        history_df: Validated Loto 6 history DataFrame.
        set_ball_mode: How to pick the referenced set ball for set-ball appearance stats:
            - "prev" (default): previous draw's set ball (100% leak-free).
            - "fixed": fixed `target_set_ball` A-J (100% leak-free).
            - "actual": draw N's actual set ball (use ONLY for conditional modeling
              P(y_{N,k}=1 | set_ball_N=s); counts still use strictly draws 1..N-1).
        target_set_ball: Set ball A-J (required when `set_ball_mode="fixed"`).
            If provided while `set_ball_mode="prev"`, automatically switches to `"fixed"`.
        min_past_draws: Exclude initial warm-up draws with fewer than `min_past_draws`
            historical draws available (default 0 returns all draws).

    Returns:
        DataFrame with `len(filtered_draws) * 43` rows containing identifier columns,
        all leak-free `FEATURE_COLS`, and `target` (0/1 main-number appearance in draw N).
    """
    df = validate_history_df(history_df)
    if target_set_ball is not None and set_ball_mode == "prev":
        set_ball_mode = "fixed"

    n_draws = len(df)
    appear_mat = _build_appearance_matrix(df)
    set_balls = df["set_ball"].to_numpy()
    sb_indices = np.array([SET_BALL_TO_IDX[sb] for sb in set_balls], dtype=np.int32)

    cum_all, cum_sb_appear, cum_sb_draws, intervals, sb_seq_cum, tail_intervals = _compute_prefix_states(
        appear_mat, sb_indices
    )
    ref_set_balls = _resolve_ref_set_balls(set_balls, set_ball_mode, target_set_ball)

    draw_nos = df["draw_no"].to_numpy(dtype=np.int32)
    draw_dates = df["draw_date"].to_numpy()
    numbers_vec = np.arange(NUM_MIN, NUM_MAX + 1, dtype=np.int32)

    start_idx = max(0, int(min_past_draws))
    n_out_draws = max(0, n_draws - start_idx)
    total_rows = n_out_draws * N_NUMBERS

    # Pre-allocate contiguous arrays for fast assembly (~0.1s for 90k+ rows)
    out_draw_no = np.empty(total_rows, dtype=np.int32)
    out_draw_date = np.empty(total_rows, dtype=object)
    out_number = np.empty(total_rows, dtype=np.int32)
    out_ref_sb = np.empty(total_rows, dtype=object)
    out_past_draws = np.empty(total_rows, dtype=np.int32)
    out_target = np.empty(total_rows, dtype=np.int32)

    feature_arrays: dict[str, np.ndarray] = {
        col: np.empty(total_rows, dtype=np.float64 if col in FLOAT_FEATURE_COLS else np.int32)
        for col in FEATURE_COLS
    }

    for pos, i in enumerate(range(start_idx, n_draws)):
        sl = slice(pos * N_NUMBERS, (pos + 1) * N_NUMBERS)
        ref_sb = ref_set_balls[i]

        out_draw_no[sl] = draw_nos[i]
        out_draw_date[sl] = draw_dates[i]
        out_number[sl] = numbers_vec
        out_ref_sb[sl] = ref_sb
        out_past_draws[sl] = i
        out_target[sl] = appear_mat[i]

        feats = _extract_features_for_step(
            step_idx=i,
            appear_mat=appear_mat,
            cum_all=cum_all,
            cum_sb_appear=cum_sb_appear,
            cum_sb_draws=cum_sb_draws,
            intervals=intervals,
            sb_seq_cum=sb_seq_cum,
            tail_intervals=tail_intervals,
            ref_set_ball=ref_sb,
        )
        for col in FEATURE_COLS:
            feature_arrays[col][sl] = feats[col]

    result = pd.DataFrame(
        {
            "draw_no": out_draw_no,
            "draw_date": out_draw_date,
            "number": out_number,
            "ref_set_ball": out_ref_sb,
            "past_draws_count": out_past_draws,
            **feature_arrays,
            "target": out_target,
        }
    )
    return result


def build_next_draw_features(
    history_df: pd.DataFrame,
    target_set_ball: Optional[str] = None,
) -> pd.DataFrame:
    """Build 43-row feature matrix for the next unseen draw (max(draw_no) + 1).

    Uses ALL rows in `history_df` (which constitute draws 1..(N_next - 1)).

    Args:
        history_df: Validated Loto 6 history DataFrame.
        target_set_ball: Candidate set ball (A-J) for the next draw. If None,
            defaults to the most recent draw's set ball.
    """
    df = validate_history_df(history_df)
    if df.empty:
        raise ValueError("history_df が空のため次回特徴量を生成できません。")

    n_draws = len(df)
    appear_mat = _build_appearance_matrix(df)
    set_balls = df["set_ball"].to_numpy()
    sb_indices = np.array([SET_BALL_TO_IDX[sb] for sb in set_balls], dtype=np.int32)

    cum_all, cum_sb_appear, cum_sb_draws, intervals, sb_seq_cum, tail_intervals = _compute_prefix_states(
        appear_mat, sb_indices
    )

    if target_set_ball is not None:
        ref_sb = target_set_ball.strip().upper()
        if ref_sb not in SET_BALL_TO_IDX:
            raise ValueError(f"target_set_ball は A〜J を指定してください: {target_set_ball}")
    else:
        ref_sb = str(set_balls[-1])

    feats = _extract_features_for_step(
        step_idx=n_draws,
        appear_mat=appear_mat,
        cum_all=cum_all,
        cum_sb_appear=cum_sb_appear,
        cum_sb_draws=cum_sb_draws,
        intervals=intervals,
        sb_seq_cum=sb_seq_cum,
        tail_intervals=tail_intervals,
        ref_set_ball=ref_sb,
    )

    next_draw_no = int(df["draw_no"].max()) + 1
    return pd.DataFrame(
        {
            "draw_no": np.full(N_NUMBERS, next_draw_no, dtype=np.int32),
            "number": np.arange(NUM_MIN, NUM_MAX + 1, dtype=np.int32),
            "ref_set_ball": np.full(N_NUMBERS, ref_sb, dtype=object),
            "past_draws_count": np.full(N_NUMBERS, n_draws, dtype=np.int32),
            **feats,
        }
    )


def build_next_draw_features_all_set_balls(
    history_df: pd.DataFrame,
    already_validated: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build next-draw 43-row feature matrices for all 10 set balls (A-J) in one pass.

    Strictly uses all rows in `history_df` (which constitute draws 1..(N_next - 1)).
    """
    df = history_df if already_validated else validate_history_df(history_df)
    if df.empty:
        raise ValueError("history_df が空のため次回特徴量を生成できません。")

    n_draws = len(df)
    appear_mat = _build_appearance_matrix(df)
    set_balls = df["set_ball"].to_numpy()
    sb_indices = np.array([SET_BALL_TO_IDX[sb] for sb in set_balls], dtype=np.int32)
    cum_all, cum_sb_appear, cum_sb_draws, intervals, sb_seq_cum, tail_intervals = _compute_prefix_states(
        appear_mat, sb_indices
    )

    next_draw_no = int(df["draw_no"].max()) + 1
    draw_no_arr = np.full(N_NUMBERS, next_draw_no, dtype=np.int32)
    numbers_arr = np.arange(NUM_MIN, NUM_MAX + 1, dtype=np.int32)
    past_arr = np.full(N_NUMBERS, n_draws, dtype=np.int32)

    out: dict[str, pd.DataFrame] = {}
    for sb in VALID_SET_BALLS:
        feats = _extract_features_for_step(
            step_idx=n_draws,
            appear_mat=appear_mat,
            cum_all=cum_all,
            cum_sb_appear=cum_sb_appear,
            cum_sb_draws=cum_sb_draws,
            intervals=intervals,
            sb_seq_cum=sb_seq_cum,
            tail_intervals=tail_intervals,
            ref_set_ball=sb,
        )
        out[sb] = pd.DataFrame(
            {
                "draw_no": draw_no_arr,
                "number": numbers_arr,
                "ref_set_ball": np.full(N_NUMBERS, sb, dtype=object),
                "past_draws_count": past_arr,
                **feats,
            }
        )
    return out


def compute_previous_draw_profile(
    history_df: pd.DataFrame,
    already_validated: bool = False,
) -> dict[str, object]:
    """Summarize the latest draw's structure & 24-draw context for next-draw profiling."""
    df = history_df if already_validated else validate_history_df(history_df)
    if df.empty:
        raise ValueError("history_df が空のため前回プロファイルを作成できません。")

    latest = df.iloc[-1]
    prev_nums = sorted(int(latest[c]) for c in MAIN_NUM_COLS)
    prev_set = set(prev_nums)
    prev_sum = int(sum(prev_nums))
    prev_odd = int(sum(n % 2 for n in prev_nums))
    prev_consec = [(prev_nums[i], prev_nums[i + 1]) for i in range(5) if prev_nums[i + 1] == prev_nums[i] + 1]
    slide_nums = sorted({x + d for x in prev_set for d in (-1, 1) if NUM_MIN <= x + d <= NUM_MAX} - prev_set)
    lag2_nums = sorted(int(df.iloc[-2][c]) for c in MAIN_NUM_COLS) if len(df) >= 2 else []

    decade_labels = ["01-09", "10-19", "20-29", "30-39", "40-43"]
    decade_counts = {lbl: 0 for lbl in decade_labels}
    for n in prev_nums:
        decade_counts[decade_labels[n // 10]] += 1

    # Last-digit stats over past 24 draws & consecutive miss streaks
    last24 = df.tail(24)
    tail_24_counts = {d: 0 for d in range(10)}
    for _, r in last24.iterrows():
        for c in MAIN_NUM_COLS:
            tail_24_counts[int(r[c]) % 10] += 1

    tail_gaps: dict[int, int] = {}
    for d in range(10):
        gap = 0
        for idx in range(len(df) - 1, -1, -1):
            row_nums = [int(df.iloc[idx][c]) for c in MAIN_NUM_COLS]
            if any(x % 10 == d for x in row_nums):
                break
            gap += 1
        tail_gaps[d] = gap

    prev_tails = [n % 10 for n in prev_nums]
    prev_dup_tails = sorted({d for d in prev_tails if prev_tails.count(d) >= 2})

    # Empirical next-draw sum distribution conditioned on previous sum zone
    all_sums = df[MAIN_NUM_COLS].sum(axis=1).to_numpy(dtype=int)
    if prev_sum >= 160:
        zone_label = "前回160以上"
        mask = all_sums[:-1] >= 160
    elif prev_sum <= 105:
        zone_label = "前回105以下"
        mask = all_sums[:-1] <= 105
    else:
        zone_label = "前回標準域 (106〜159)"
        mask = (all_sums[:-1] > 105) & (all_sums[:-1] < 160)

    cond_next_sums = all_sums[1:][mask] if len(all_sums) > 1 else np.array([], dtype=int)
    if len(cond_next_sums) > 0:
        next_sum_mean = float(np.mean(cond_next_sums))
        next_in_120_150_pct = float(((cond_next_sums >= 120) & (cond_next_sums <= 150)).mean() * 100.0)
        next_in_100_165_pct = float(((cond_next_sums >= 100) & (cond_next_sums <= 165)).mean() * 100.0)
    else:
        next_sum_mean = 132.0
        next_in_120_150_pct = 40.0
        next_in_100_165_pct = 70.0

    return {
        "draw_no": int(latest["draw_no"]),
        "draw_date": str(latest["draw_date"]),
        "numbers": prev_nums,
        "bonus": int(latest["bonus"]),
        "set_ball": str(latest["set_ball"]),
        "sum_value": prev_sum,
        "odd_count": prev_odd,
        "even_count": 6 - prev_odd,
        "consecutive_pairs": prev_consec,
        "slide_candidates": slide_nums,
        "lag2_numbers": lag2_nums,
        "decade_counts": decade_counts,
        "prev_dup_tails": prev_dup_tails,
        "tail_24_counts": tail_24_counts,
        "tail_gaps": tail_gaps,
        "cond_zone_label": zone_label,
        "cond_sample_n": int(len(cond_next_sums)),
        "cond_next_sum_mean": next_sum_mean,
        "cond_next_120_150_pct": next_in_120_150_pct,
        "cond_next_100_165_pct": next_in_100_165_pct,
    }


def compute_set_ball_summary(
    history_df: pd.DataFrame,
    up_to_draw_no: Optional[int] = None,
) -> pd.DataFrame:
    """Compute per-(set_ball, number) appearance stats strictly up to `up_to_draw_no`.

    Useful for EDA and Phase 5 set-ball bias heatmaps without any future leakage.
    """
    df = validate_history_df(history_df)
    if up_to_draw_no is not None:
        df = df[df["draw_no"] <= up_to_draw_no].reset_index(drop=True)
    if df.empty:
        raise ValueError("指定された範囲にデータが存在しません。")

    appear_mat = _build_appearance_matrix(df)
    total_draws = len(df)
    overall_rates = appear_mat.sum(axis=0) / float(total_draws)

    rows = []
    for sb in VALID_SET_BALLS:
        mask = (df["set_ball"] == sb).to_numpy()
        sb_draws = int(mask.sum())
        counts = appear_mat[mask].sum(axis=0) if sb_draws > 0 else np.zeros(N_NUMBERS, dtype=np.int32)
        for idx in range(N_NUMBERS):
            rate = (counts[idx] / float(sb_draws)) if sb_draws > 0 else np.nan
            rows.append(
                {
                    "set_ball": sb,
                    "number": idx + 1,
                    "set_ball_draws": sb_draws,
                    "appearances": int(counts[idx]),
                    "appearance_rate": rate,
                    "rate_diff_from_global": (rate - GLOBAL_MEAN_RATE) if sb_draws > 0 else np.nan,
                    "overall_appearance_rate": float(overall_rates[idx]),
                    "rate_diff_vs_num_avg": (rate - float(overall_rates[idx])) if sb_draws > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def verify_no_data_leakage(
    history_df: pd.DataFrame,
    test_draw_nos: Optional[Iterable[int]] = None,
    set_ball_mode: SetBallMode = "prev",
    target_set_ball: Optional[str] = None,
) -> dict[str, object]:
    """Mathematically verify strict zero data leakage via truncation & mutation tests.

    For each checkpoint draw K:
      1. Computes features on the full history DataFrame.
      2. Truncates history to `draw_no <= K` (deleting all future draws > K).
      3. Corrupts draw K's winning numbers (and draw K's set_ball when mode != 'actual')
         so any accidental read of draw K would immediately change feature values.
      4. Asserts exact equality on every feature column for `draw_no == K`.

    Raises:
        AssertionError: If any data leakage is detected.
    """
    df = validate_history_df(history_df)
    if len(df) < 5:
        raise ValueError("リーク検証には最低5回分のデータが必要です。")

    if test_draw_nos is None:
        all_draws = df["draw_no"].tolist()
        candidate_indices = sorted({1, len(all_draws) // 4, len(all_draws) // 2, len(all_draws) - 1})
        test_draw_nos = [all_draws[idx] for idx in candidate_indices]
    else:
        test_draw_nos = list(test_draw_nos)

    full_features = generate_features(
        df, set_ball_mode=set_ball_mode, target_set_ball=target_set_ball
    )

    for draw_k in test_draw_nos:
        full_slice = (
            full_features[full_features["draw_no"] == draw_k]
            .sort_values("number")
            .reset_index(drop=True)
        )
        if full_slice.empty:
            raise ValueError(f"draw_no={draw_k} が履歴に存在しません。")

        # Truncate all draws > K and mutate draw K's outcome
        trunc_df = df[df["draw_no"] <= draw_k].copy().reset_index(drop=True)
        last_idx = len(trunc_df) - 1
        orig_nums = set(int(trunc_df.loc[last_idx, c]) for c in MAIN_NUM_COLS)
        alt_nums = [n for n in range(1, 44) if n not in orig_nums][:7]
        for j, col in enumerate(MAIN_NUM_COLS):
            trunc_df.loc[last_idx, col] = alt_nums[j]
        trunc_df.loc[last_idx, "bonus"] = alt_nums[6]

        if set_ball_mode != "actual":
            orig_sb = str(trunc_df.loc[last_idx, "set_ball"])
            alt_sb = next(sb for sb in VALID_SET_BALLS if sb != orig_sb)
            trunc_df.loc[last_idx, "set_ball"] = alt_sb

        trunc_features = generate_features(
            trunc_df, set_ball_mode=set_ball_mode, target_set_ball=target_set_ball
        )
        trunc_slice = (
            trunc_features[trunc_features["draw_no"] == draw_k]
            .sort_values("number")
            .reset_index(drop=True)
        )

        for col in FEATURE_COLS:
            pd.testing.assert_series_equal(
                full_slice[col],
                trunc_slice[col],
                check_exact=False,
                rtol=0.0,
                atol=1e-15,
                obj=f"draw_no={draw_k} feature '{col}'",
            )

    return {
        "passed": True,
        "checked_draws": test_draw_nos,
        "set_ball_mode": set_ball_mode,
        "checked_features": list(FEATURE_COLS),
    }


if __name__ == "__main__":
    from src.scraper import DEFAULT_OUTPUT_PATH, load_history_csv

    history = load_history_csv(DEFAULT_OUTPUT_PATH)
    feats_df = generate_features(history, set_ball_mode="prev")
    next_df = build_next_draw_features(history)
    leak_report = verify_no_data_leakage(history, set_ball_mode="prev")

    print(f"履歴件数: {len(history)} 回")
    print(f"生成特徴量シェイプ: {feats_df.shape} (期待値: {len(history) * 43} 行)")
    print(f"データリーク検証: {'PASS' if leak_report['passed'] else 'FAIL'} (検証回: {leak_report['checked_draws']})")
    print("\n--- 次回予測用特徴量 (上位10数字・セット球別乖離降順) ---")
    cols_preview = [
        "draw_no",
        "number",
        "ref_set_ball",
        "set_ball_appearance_rate",
        "set_ball_rate_diff",
        "interval",
        "recent_5_count",
        "recent_10_count",
        "recent_30_count",
        "is_pull",
        "is_slide",
    ]
    print(
        next_df.sort_values("set_ball_rate_diff", ascending=False)[cols_preview]
        .head(10)
        .to_string(index=False)
    )
