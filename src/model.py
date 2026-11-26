"""Phase 2: Leak-free Loto 6 prediction engine (Set-ball + Bayes + LightGBM).

Architecture (strictly zero data leakage; predicts draw N using draws <= N-1 only):
  1. Period Filter (`filter_history_period`):
     - Default: post-new-ball era (`start_draw_no=138`).
     - Supports full history (`start_draw_no=1`), rolling recent window (`recent_n_draws`),
       and strict causal cutoff (`as_of_draw_no=N-1`).
  2. Stage 1 — Set Ball Predictor (`predict_set_ball_probabilities`):
     - Combines empirical survival hazard over set-ball inter-arrival gaps (capturing the
       strong physical rotation cycle in Loto 6 where consecutive repeat is <1% and typical
       gaps cluster around 8-11 draws), Laplace-smoothed 1st-order Markov transitions, and
       recent usage balance.
     - Also supports manual set ball override (`manual_set_ball="A".."J"`).
  3. Stage 2A — Hierarchical Empirical Bayes (`compute_bayesian_conditional_scores`):
     - Estimates P_bayes(y_{N,k}=1 | set_ball_N=s) with Beta-Binomial shrinkage toward the
       global mean (6/43) and number-level baseline, calibrated to sum to 6.0 per set ball.
  4. Stage 2B — Conditional LightGBM (`Loto6PredictionEngine`):
     - Trained strictly on draws <= N-1 (with `min_past_draws=100` warm-up and chronological
       early-stopping split) using Phase 1 leak-free features to learn
       P_lgbm(y_{t,k}=1 | set_ball_t=s, history_{<t}).
  5. Integration & Marginalization:
     - Blends Bayes and LightGBM conditional probabilities per set ball using configurable
       `lgbm_weight`, then marginalizes over Stage 1's set-ball distribution:
           P_final(k) = sum_s P(set_ball_N = s) * P_blend(k | s)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.features import (
    FEATURE_COLS,
    GLOBAL_MEAN_RATE,
    N_MAIN_DRAW,
    N_NUMBERS,
    NUM_MAX,
    NUM_MIN,
    SET_BALL_TO_IDX,
    build_next_draw_features_all_set_balls,
    generate_features,
)
from src.scraper import MAIN_NUM_COLS, VALID_SET_BALLS, validate_history_df

logger = logging.getLogger(__name__)

DEFAULT_START_DRAW_NO = 138  # 新球切り替わり回（第138回〜）
DEFAULT_MIN_PAST_DRAWS = 100
DEFAULT_TOP_K = 12
MODEL_FEATURE_COLS = [*FEATURE_COLS, "number"]


@dataclass
class PredictionConfig:
    """Configuration for the Phase 2 prediction engine."""

    start_draw_no: Optional[int] = DEFAULT_START_DRAW_NO
    recent_n_draws: Optional[int] = None
    min_past_draws: int = DEFAULT_MIN_PAST_DRAWS
    lgbm_weight: float = 0.50
    bayes_prior_strength: float = 30.0
    bayes_num_prior_strength: float = 50.0
    bayes_num_baseline_weight: float = 0.35
    bayes_recency_half_life: Optional[float] = 50.0  # per-set-ball exponential decay half-life (None/<=0 disables)
    manual_set_ball: Optional[str] = None
    top_k: int = DEFAULT_TOP_K
    val_draws: int = 100
    hazard_smoothing: float = 5.0
    markov_smoothing: float = 2.0
    hazard_weight: float = 0.70
    markov_weight: float = 0.20
    balance_weight: float = 0.10
    random_state: int = 42


@dataclass
class PredictionResult:
    """Structured prediction output for draw N (computed strictly from draws <= N-1)."""

    target_draw_no: int
    as_of_draw_no: int
    train_start_draw: int
    train_end_draw: int
    n_history_draws: int
    lgbm_weight: float
    manual_set_ball: Optional[str]
    set_ball_probs: pd.DataFrame
    number_scores: pd.DataFrame
    conditional_scores: pd.DataFrame
    top_numbers: list[int]
    model_metrics: dict[str, float]
    feature_importance: pd.DataFrame


def filter_history_period(
    history_df: pd.DataFrame,
    start_draw_no: Optional[int] = DEFAULT_START_DRAW_NO,
    recent_n_draws: Optional[int] = None,
    as_of_draw_no: Optional[int] = None,
    already_validated: bool = False,
) -> pd.DataFrame:
    """Filter Loto 6 history with strict causal cutoff (`as_of_draw_no`).

    Args:
        history_df: Full or partial Loto 6 history DataFrame.
        start_draw_no: Minimum draw_no to include (default 138 = new balls; 1 or None = all).
        recent_n_draws: If specified, keep only the most recent N draws (after causal cutoff).
        as_of_draw_no: Strict upper bound (`draw_no <= as_of_draw_no`) to prevent data leakage.
        already_validated: Skip re-validating schema/dates when caller already validated.
    """
    df = history_df if already_validated else validate_history_df(history_df)

    # Apply causal cutoff FIRST so future draws can never influence period selection.
    if as_of_draw_no is not None:
        df = df[df["draw_no"] <= int(as_of_draw_no)]

    if start_draw_no is not None:
        df = df[df["draw_no"] >= int(start_draw_no)]

    if recent_n_draws is not None:
        if recent_n_draws <= 0:
            raise ValueError("recent_n_draws は 1 以上を指定してください。")
        df = df.tail(int(recent_n_draws))

    df = df.sort_values("draw_no").reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"フィルタ後の履歴データが空です (start_draw_no={start_draw_no}, "
            f"recent_n_draws={recent_n_draws}, as_of_draw_no={as_of_draw_no})。"
        )
    return df


def predict_set_ball_probabilities(
    history_df: pd.DataFrame,
    manual_set_ball: Optional[str] = None,
    hazard_smoothing: float = 5.0,
    markov_smoothing: float = 2.0,
    hazard_weight: float = 0.70,
    markov_weight: float = 0.20,
    balance_weight: float = 0.10,
    max_gap_bucket: int = 18,
    balance_window: int = 50,
    already_validated: bool = False,
) -> pd.DataFrame:
    """Predict next draw's set-ball probabilities strictly from past history.

    Combines:
      1. Empirical gap-hazard distribution (models physical set-ball rotation cycle).
      2. Laplace-smoothed 1st-order Markov transitions P(s_N | s_{N-1}).
      3. Recent balance adjustment over the last `balance_window` draws.
    """
    df = history_df if already_validated else validate_history_df(history_df)
    n_draws = len(df)
    sbs = df["set_ball"].to_numpy()
    sb_indices = np.array([SET_BALL_TO_IDX[sb] for sb in sbs], dtype=np.int32)
    n_sb = len(VALID_SET_BALLS)

    # Track empirical gap hazard strictly sequentially across history
    opp_counts = np.zeros(max_gap_bucket + 1, dtype=np.float64)
    hit_counts = np.zeros(max_gap_bucket + 1, dtype=np.float64)
    last_seen = np.full(n_sb, -1, dtype=np.int32)

    for t in range(n_draws):
        chosen = sb_indices[t]
        for s in range(n_sb):
            if last_seen[s] >= 0:
                gap = min(t - last_seen[s] - 1, max_gap_bucket)
                opp_counts[gap] += 1.0
                if s == chosen:
                    hit_counts[gap] += 1.0
        last_seen[chosen] = t

    # Current gap for each set ball before the NEXT draw (index = n_draws)
    current_gaps = np.array(
        [n_draws - last_seen[s] - 1 if last_seen[s] >= 0 else n_draws for s in range(n_sb)],
        dtype=np.int32,
    )

    # Smoothed empirical hazard per gap bucket
    prior_p = 1.0 / n_sb
    hazard_by_gap = (hit_counts + hazard_smoothing * prior_p) / (opp_counts + hazard_smoothing)
    raw_hazard = np.array([hazard_by_gap[min(int(g), max_gap_bucket)] for g in current_gaps], dtype=np.float64)
    hazard_prob = raw_hazard / raw_hazard.sum()

    # 1st-order Markov transition probabilities from the latest draw's set ball
    trans_counts = np.zeros((n_sb, n_sb), dtype=np.float64)
    if n_draws >= 2:
        for t in range(1, n_draws):
            trans_counts[sb_indices[t - 1], sb_indices[t]] += 1.0
    last_sb_idx = int(sb_indices[-1])
    markov_row = trans_counts[last_sb_idx] + markov_smoothing * prior_p
    markov_prob = markov_row / markov_row.sum()

    # Recent usage balance (slightly favors under-used set balls in recent window)
    recent_slice = sb_indices[-min(balance_window, n_draws) :]
    recent_counts = np.bincount(recent_slice, minlength=n_sb).astype(np.float64)
    expected_recent = len(recent_slice) / n_sb
    deficit = np.clip(expected_recent - recent_counts, -expected_recent, expected_recent)
    raw_balance = np.exp(0.25 * deficit / max(1.0, np.sqrt(expected_recent)))
    balance_prob = raw_balance / raw_balance.sum()

    # Past draw counts per set ball across filtered history
    total_counts = np.bincount(sb_indices, minlength=n_sb)

    if manual_set_ball is not None:
        chosen_sb = manual_set_ball.strip().upper()
        if chosen_sb not in SET_BALL_TO_IDX:
            raise ValueError(f"manual_set_ball は A〜J を指定してください: {manual_set_ball}")
        final_prob = np.array([1.0 if sb == chosen_sb else 0.0 for sb in VALID_SET_BALLS], dtype=np.float64)
    else:
        w_sum = hazard_weight + markov_weight + balance_weight
        if w_sum <= 0:
            raise ValueError("セット球予測の重み合計は正の値にしてください。")
        combined = (
            hazard_weight * hazard_prob
            + markov_weight * markov_prob
            + balance_weight * balance_prob
        ) / w_sum
        final_prob = combined / combined.sum()

    result = pd.DataFrame(
        {
            "set_ball": list(VALID_SET_BALLS),
            "last_gap": current_gaps,
            "past_draws": total_counts,
            "recent_50_count": recent_counts.astype(int),
            "hazard_prob": hazard_prob,
            "markov_prob": markov_prob,
            "balance_prob": balance_prob,
            "final_prob": final_prob,
        }
    )
    return result.sort_values("final_prob", ascending=False).reset_index(drop=True)


def compute_bayesian_conditional_scores(
    history_df: pd.DataFrame,
    prior_strength: float = 30.0,
    num_prior_strength: float = 50.0,
    num_baseline_weight: float = 0.35,
    recency_half_life: Optional[float] = 50.0,
    already_validated: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute hierarchical Beta-Binomial posterior probabilities P_bayes(k | s).

    When `recency_half_life` is positive, applies exponential recency decay per set ball
    (half-life in draws of that set ball, and `10 * half_life` across overall history) so
    recent physical ball conditions carry greater weight than decades-old draws.

    Returns:
        (mean_df, std_df): Both shaped (43, 10) with index 1..43 and columns A..J.
            Each column of `mean_df` is calibrated so sum_{k=1..43} P(k | s) = 6.0.
    """
    df = history_df if already_validated else validate_history_df(history_df)
    n_draws = len(df)
    n_sb = len(VALID_SET_BALLS)
    use_decay = recency_half_life is not None and float(recency_half_life) > 0.0
    hl = float(recency_half_life) if use_decay else 0.0

    appear_mat = np.zeros((n_draws, N_NUMBERS), dtype=np.float64)
    main_nums = df[MAIN_NUM_COLS].to_numpy(dtype=np.int32)
    appear_mat[np.repeat(np.arange(n_draws), N_MAIN_DRAW), (main_nums - 1).ravel()] = 1.0

    # Level 1: number-level overall posterior rate shrunk toward global mean 6/43
    if use_decay and n_draws > 0:
        ages_all = np.arange(n_draws - 1, -1, -1, dtype=np.float64)
        w_all = np.power(0.5, ages_all / (hl * float(n_sb)))
        num_counts = (appear_mat * w_all[:, None]).sum(axis=0)
        eff_draws = float(w_all.sum())
    else:
        num_counts = appear_mat.sum(axis=0)
        eff_draws = float(n_draws)
    num_overall_post = (num_counts + num_prior_strength * GLOBAL_MEAN_RATE) / (eff_draws + num_prior_strength)
    prior_p_per_num = (1.0 - num_baseline_weight) * GLOBAL_MEAN_RATE + num_baseline_weight * num_overall_post

    # Level 2: set-ball conditional Beta posterior per (number, set_ball)
    mean_mat = np.empty((N_NUMBERS, n_sb), dtype=np.float64)
    std_mat = np.empty((N_NUMBERS, n_sb), dtype=np.float64)

    for s_idx, sb in enumerate(VALID_SET_BALLS):
        mask = (df["set_ball"] == sb).to_numpy()
        sb_sub = appear_mat[mask]
        n_s = len(sb_sub)
        if n_s > 0:
            if use_decay:
                ages_s = np.arange(n_s - 1, -1, -1, dtype=np.float64)
                w_s = np.power(0.5, ages_s / hl)
                sb_counts = (sb_sub * w_s[:, None]).sum(axis=0)
                sb_draws = float(w_s.sum())
            else:
                sb_counts = sb_sub.sum(axis=0)
                sb_draws = float(n_s)
        else:
            sb_counts = np.zeros(N_NUMBERS, dtype=np.float64)
            sb_draws = 0.0

        alpha_post = sb_counts + prior_strength * prior_p_per_num
        beta_post = (sb_draws - sb_counts) + prior_strength * (1.0 - prior_p_per_num)
        total_post = alpha_post + beta_post

        mean_raw = alpha_post / total_post
        # Calibrate column sum to exact expected count of 6 drawn numbers
        scale = float(N_MAIN_DRAW) / float(mean_raw.sum())
        mean_mat[:, s_idx] = np.clip(mean_raw * scale, 1e-6, 1.0 - 1e-6)
        std_mat[:, s_idx] = np.sqrt((alpha_post * beta_post) / ((total_post ** 2) * (total_post + 1.0)))

    index = pd.Index(np.arange(NUM_MIN, NUM_MAX + 1), name="number")
    mean_df = pd.DataFrame(mean_mat, index=index, columns=list(VALID_SET_BALLS))
    std_df = pd.DataFrame(std_mat, index=index, columns=list(VALID_SET_BALLS))
    return mean_df, std_df


def _calibrate_probs_to_six(probs: np.ndarray) -> np.ndarray:
    """Scale a 43-vector of predicted appearance probabilities to sum to 6.0."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    total = float(p.sum())
    if total <= 0:
        return np.full_like(p, GLOBAL_MEAN_RATE)
    return np.clip(p * (float(N_MAIN_DRAW) / total), 1e-6, 1.0 - 1e-6)


class Loto6PredictionEngine:
    """Two-stage leak-free Loto 6 prediction engine (Set-ball + Bayes + LightGBM)."""

    def __init__(self, config: Optional[PredictionConfig] = None) -> None:
        self.config = config or PredictionConfig()
        self._filtered_history: Optional[pd.DataFrame] = None
        self._as_of_draw_no: Optional[int] = None
        self._bayes_mean_df: Optional[pd.DataFrame] = None
        self._bayes_std_df: Optional[pd.DataFrame] = None
        self._lgbm_model: Optional[lgb.LGBMClassifier] = None
        self._lgbm_cond_df: Optional[pd.DataFrame] = None
        self._emp_sb_rates: Optional[np.ndarray] = None
        self._emp_sb_rec50_rates: Optional[np.ndarray] = None
        self._next_base_feats: Optional[pd.DataFrame] = None
        self._model_metrics: dict[str, float] = {}
        self._feature_importance: pd.DataFrame = pd.DataFrame(columns=["feature", "importance"])

    def fit(
        self,
        history_df: pd.DataFrame,
        as_of_draw_no: Optional[int] = None,
    ) -> "Loto6PredictionEngine":
        """Fit Bayesian & LightGBM models strictly on past draws (<= as_of_draw_no)."""
        cfg = self.config
        if not (0.0 <= cfg.lgbm_weight <= 1.0):
            raise ValueError("lgbm_weight は 0.0〜1.0 の範囲で指定してください。")

        filtered = filter_history_period(
            history_df,
            start_draw_no=cfg.start_draw_no,
            recent_n_draws=cfg.recent_n_draws,
            as_of_draw_no=as_of_draw_no,
        )
        self._filtered_history = filtered
        self._as_of_draw_no = int(filtered["draw_no"].max())

        # Stage 2A: Hierarchical Empirical Bayes (strictly from filtered history)
        self._bayes_mean_df, self._bayes_std_df = compute_bayesian_conditional_scores(
            filtered,
            prior_strength=cfg.bayes_prior_strength,
            num_prior_strength=cfg.bayes_num_prior_strength,
            num_baseline_weight=cfg.bayes_num_baseline_weight,
            recency_half_life=cfg.bayes_recency_half_life,
            already_validated=True,
        )

        # Stage 2B: Conditional LightGBM P(y_{t,k}=1 | set_ball_t=s, history_{<t})
        # Every training draw t <= as_of_draw_no uses ONLY features from draws < t.
        n_draws = len(filtered)
        effective_min_past = min(cfg.min_past_draws, max(10, n_draws - 40))
        if effective_min_past < cfg.min_past_draws:
            logger.warning(
                "履歴件数 (%d 回) に対して min_past_draws=%d が大きいため %d に自動調整しました。",
                n_draws,
                cfg.min_past_draws,
                effective_min_past,
            )

        feat_df = generate_features(
            filtered,
            set_ball_mode="actual",
            min_past_draws=effective_min_past,
        )
        unique_draws = feat_df["draw_no"].unique()
        if len(unique_draws) < 20:
            raise ValueError(
                f"LightGBM 学習に十分な開催回数がありません（有効回数: {len(unique_draws)} 回）。"
                "start_draw_no や min_past_draws を見直してください。"
            )

        # Strictly chronological train / validation split (never shuffle!)
        val_n = min(cfg.val_draws, max(10, len(unique_draws) // 5))
        split_draw = int(unique_draws[-val_n])
        train_mask = feat_df["draw_no"] < split_draw
        val_mask = feat_df["draw_no"] >= split_draw

        X_train = feat_df.loc[train_mask, MODEL_FEATURE_COLS]
        y_train = feat_df.loc[train_mask, "target"].to_numpy(dtype=np.int32)
        X_val = feat_df.loc[val_mask, MODEL_FEATURE_COLS]
        y_val = feat_df.loc[val_mask, "target"].to_numpy(dtype=np.int32)

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=7,
            max_depth=3,
            min_child_samples=120,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=cfg.random_state,
            n_jobs=1,
            verbose=-1,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
            )

        val_pred = np.clip(model.predict_proba(X_val)[:, 1], 1e-6, 1.0 - 1e-6)
        uniform_pred = np.full_like(val_pred, GLOBAL_MEAN_RATE)
        val_ll = float(log_loss(y_val, val_pred))
        base_ll = float(log_loss(y_val, uniform_pred))

        # Refit on all available training draws (<= as_of_draw_no) using optimal iteration count
        best_iter = int(getattr(model, "best_iteration_", 0) or 50)
        best_iter = max(15, best_iter)
        final_model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=best_iter,
            learning_rate=0.03,
            num_leaves=7,
            max_depth=3,
            min_child_samples=120,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=cfg.random_state,
            n_jobs=1,
            verbose=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_model.fit(feat_df[MODEL_FEATURE_COLS], feat_df["target"].to_numpy(dtype=np.int32))

        self._lgbm_model = final_model
        self._model_metrics = {
            "val_draws": float(val_n),
            "best_iteration": float(best_iter),
            "val_log_loss": val_ll,
            "uniform_baseline_log_loss": base_ll,
            "log_loss_diff_vs_uniform": val_ll - base_ll,
        }
        self._feature_importance = (
            pd.DataFrame(
                {
                    "feature": MODEL_FEATURE_COLS,
                    "importance": final_model.feature_importances_.astype(float),
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        # Precompute conditional LightGBM probabilities & empirical set-ball rates
        # for all 10 candidate set balls A..J in one pass using strictly filtered history.
        all_next_feats = build_next_draw_features_all_set_balls(filtered, already_validated=True)
        lgbm_cond_mat = np.empty((N_NUMBERS, len(VALID_SET_BALLS)), dtype=np.float64)
        emp_sb_rates = np.empty((N_NUMBERS, len(VALID_SET_BALLS)), dtype=np.float64)
        emp_sb_rec50_rates = np.empty((N_NUMBERS, len(VALID_SET_BALLS)), dtype=np.float64)
        next_base_feats = all_next_feats[VALID_SET_BALLS[0]]
        for s_idx, sb in enumerate(VALID_SET_BALLS):
            next_feat_s = all_next_feats[sb]
            raw_probs = final_model.predict_proba(next_feat_s[MODEL_FEATURE_COLS])[:, 1]
            lgbm_cond_mat[:, s_idx] = _calibrate_probs_to_six(raw_probs)
            rates = next_feat_s["set_ball_appearance_rate"].to_numpy(dtype=np.float64)
            emp_sb_rates[:, s_idx] = np.where(np.isnan(rates), GLOBAL_MEAN_RATE, rates)
            rec50 = next_feat_s["set_ball_recent_50_rate"].to_numpy(dtype=np.float64)
            emp_sb_rec50_rates[:, s_idx] = np.where(np.isnan(rec50), GLOBAL_MEAN_RATE, rec50)

        self._lgbm_cond_df = pd.DataFrame(
            lgbm_cond_mat,
            index=pd.Index(np.arange(NUM_MIN, NUM_MAX + 1), name="number"),
            columns=list(VALID_SET_BALLS),
        )
        self._emp_sb_rates = emp_sb_rates
        self._emp_sb_rec50_rates = emp_sb_rec50_rates
        self._next_base_feats = next_base_feats
        return self

    def predict_next(
        self,
        manual_set_ball: Optional[str] = None,
        lgbm_weight: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> PredictionResult:
        """Produce marginalized next-draw scores from the fitted engine.

        Allows instant re-scoring when changing `manual_set_ball`, `lgbm_weight`, or
        `top_k` (e.g., from Streamlit controls) without retraining LightGBM.
        """
        if (
            self._filtered_history is None
            or self._bayes_mean_df is None
            or self._lgbm_cond_df is None
            or self._as_of_draw_no is None
            or self._emp_sb_rates is None
            or self._emp_sb_rec50_rates is None
            or self._next_base_feats is None
        ):
            raise RuntimeError("predict_next() の前に fit() を実行してください。")

        cfg = self.config
        use_sb = manual_set_ball if manual_set_ball is not None else cfg.manual_set_ball
        w_lgbm = cfg.lgbm_weight if lgbm_weight is None else float(lgbm_weight)
        k_top = cfg.top_k if top_k is None else int(top_k)

        if not (0.0 <= w_lgbm <= 1.0):
            raise ValueError("lgbm_weight は 0.0〜1.0 の範囲で指定してください。")
        if not (1 <= k_top <= N_NUMBERS):
            raise ValueError("top_k は 1〜43 の範囲で指定してください。")

        filtered = self._filtered_history

        # Stage 1: Set ball probabilities
        sb_probs_df = predict_set_ball_probabilities(
            filtered,
            manual_set_ball=use_sb,
            hazard_smoothing=cfg.hazard_smoothing,
            markov_smoothing=cfg.markov_smoothing,
            hazard_weight=cfg.hazard_weight,
            markov_weight=cfg.markov_weight,
            balance_weight=cfg.balance_weight,
        )
        sb_prob_vec = (
            sb_probs_df.set_index("set_ball").loc[list(VALID_SET_BALLS), "final_prob"].to_numpy(dtype=np.float64)
        )

        # Stage 2: Blend conditional Bayes & LightGBM scores per set ball
        cond_blend = (1.0 - w_lgbm) * self._bayes_mean_df + w_lgbm * self._lgbm_cond_df

        # Marginalize over Stage-1 set ball distribution: sum_s P(s) * P(k | s)
        bayes_marg = self._bayes_mean_df.to_numpy(dtype=np.float64) @ sb_prob_vec
        lgbm_marg = self._lgbm_cond_df.to_numpy(dtype=np.float64) @ sb_prob_vec
        final_marg = _calibrate_probs_to_six(cond_blend.to_numpy(dtype=np.float64) @ sb_prob_vec)

        next_base_feats = self._next_base_feats
        expected_sb_rate = self._emp_sb_rates @ sb_prob_vec
        expected_sb_rec50_rate = self._emp_sb_rec50_rates @ sb_prob_vec

        scores_df = pd.DataFrame(
            {
                "number": np.arange(NUM_MIN, NUM_MAX + 1, dtype=int),
                "final_score": final_marg,
                "bayes_score": bayes_marg,
                "lgbm_score": lgbm_marg,
                "score_lift": final_marg / GLOBAL_MEAN_RATE,
                "expected_set_ball_rate": expected_sb_rate,
                "expected_set_ball_rec50_rate": expected_sb_rec50_rate,
                "overall_rate": next_base_feats["overall_appearance_rate"].to_numpy(dtype=np.float64),
                "interval": next_base_feats["interval"].to_numpy(dtype=int),
                "recent_5_count": next_base_feats["recent_5_count"].to_numpy(dtype=int),
                "recent_10_count": next_base_feats["recent_10_count"].to_numpy(dtype=int),
                "recent_24_count": next_base_feats["recent_24_count"].to_numpy(dtype=int),
                "recent_30_count": next_base_feats["recent_30_count"].to_numpy(dtype=int),
                "is_pull": next_base_feats["is_pull"].to_numpy(dtype=int),
                "is_lag2_repeat": next_base_feats["is_lag2_repeat"].to_numpy(dtype=int),
                "is_slide": next_base_feats["is_slide"].to_numpy(dtype=int),
                "is_prev_consec_pull": next_base_feats["is_prev_consec_pull"].to_numpy(dtype=int),
                "is_prev_consec_slide": next_base_feats["is_prev_consec_slide"].to_numpy(dtype=int),
                "last_digit_gap": next_base_feats["last_digit_gap"].to_numpy(dtype=int),
                "prev_decade_count": next_base_feats["prev_decade_count"].to_numpy(dtype=int),
            }
        )
        scores_df = scores_df.sort_values(
            ["final_score", "bayes_score", "number"], ascending=[False, False, True]
        ).reset_index(drop=True)
        scores_df.insert(0, "rank", np.arange(1, N_NUMBERS + 1, dtype=int))

        top_numbers = scores_df.head(k_top)["number"].astype(int).tolist()

        return PredictionResult(
            target_draw_no=self._as_of_draw_no + 1,
            as_of_draw_no=self._as_of_draw_no,
            train_start_draw=int(filtered["draw_no"].min()),
            train_end_draw=int(filtered["draw_no"].max()),
            n_history_draws=len(filtered),
            lgbm_weight=w_lgbm,
            manual_set_ball=use_sb,
            set_ball_probs=sb_probs_df,
            number_scores=scores_df,
            conditional_scores=cond_blend,
            top_numbers=top_numbers,
            model_metrics=dict(self._model_metrics),
            feature_importance=self._feature_importance.copy(),
        )

    def fit_predict_next(
        self,
        history_df: pd.DataFrame,
        as_of_draw_no: Optional[int] = None,
    ) -> PredictionResult:
        """Convenience method to fit strictly up to `as_of_draw_no` and predict next draw."""
        return self.fit(history_df, as_of_draw_no=as_of_draw_no).predict_next()


def verify_model_no_data_leakage(
    history_df: pd.DataFrame,
    test_draw_no: int,
    config: Optional[PredictionConfig] = None,
) -> dict[str, object]:
    """Verify that predicting `test_draw_no` never reads `test_draw_no` or later draws.

    Runs `fit_predict_next` twice with `as_of_draw_no = test_draw_no - 1`:
      1. On the unmodified full history DataFrame.
      2. On a DataFrame where `test_draw_no` and all future draws are deleted, AND
         also on a DataFrame where `test_draw_no`'s numbers & set_ball are corrupted.
    Asserts exact equality on both Stage-1 set ball probabilities and Stage-2 number scores.
    """
    df = validate_history_df(history_df)
    cutoff = int(test_draw_no) - 1
    cfg = config or PredictionConfig()

    res_full = Loto6PredictionEngine(cfg).fit_predict_next(df, as_of_draw_no=cutoff)

    # Test 1: Truncated (future draws deleted completely)
    df_trunc = df[df["draw_no"] <= cutoff].copy().reset_index(drop=True)
    res_trunc = Loto6PredictionEngine(cfg).fit_predict_next(df_trunc, as_of_draw_no=cutoff)

    # Test 2: Corrupted draw `test_draw_no` kept in DataFrame
    df_corrupt = df.copy()
    mask_target = df_corrupt["draw_no"] == int(test_draw_no)
    if mask_target.any():
        idx = df_corrupt.index[mask_target][0]
        orig = set(int(df_corrupt.loc[idx, c]) for c in MAIN_NUM_COLS)
        alt = [n for n in range(1, 44) if n not in orig][:7]
        for j, c in enumerate(MAIN_NUM_COLS):
            df_corrupt.loc[idx, c] = alt[j]
        df_corrupt.loc[idx, "bonus"] = alt[6]
        orig_sb = str(df_corrupt.loc[idx, "set_ball"])
        df_corrupt.loc[idx, "set_ball"] = next(sb for sb in VALID_SET_BALLS if sb != orig_sb)
    res_corrupt = Loto6PredictionEngine(cfg).fit_predict_next(df_corrupt, as_of_draw_no=cutoff)

    for label, other in [("truncated", res_trunc), ("corrupted", res_corrupt)]:
        pd.testing.assert_frame_equal(
            res_full.set_ball_probs.sort_values("set_ball").reset_index(drop=True),
            other.set_ball_probs.sort_values("set_ball").reset_index(drop=True),
            rtol=0.0,
            atol=1e-12,
            obj=f"set_ball_probs ({label})",
        )
        pd.testing.assert_frame_equal(
            res_full.number_scores.sort_values("number").reset_index(drop=True),
            other.number_scores.sort_values("number").reset_index(drop=True),
            rtol=0.0,
            atol=1e-12,
            obj=f"number_scores ({label})",
        )

    return {
        "passed": True,
        "test_draw_no": int(test_draw_no),
        "as_of_draw_no": cutoff,
        "top_numbers": res_full.top_numbers,
    }


if __name__ == "__main__":
    from src.scraper import DEFAULT_OUTPUT_PATH, load_history_csv

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    history = load_history_csv(DEFAULT_OUTPUT_PATH)

    engine = Loto6PredictionEngine(PredictionConfig())
    pred = engine.fit_predict_next(history)

    print(
        f"予測対象: 第{pred.target_draw_no}回 "
        f"(学習期間: 第{pred.train_start_draw}回〜第{pred.train_end_draw}回 / {pred.n_history_draws}回分)"
    )
    print(f"モデル検証指標: {pred.model_metrics}")
    print("\n--- Stage 1: 次回セット球予測確率 ---")
    print(
        pred.set_ball_probs[["set_ball", "last_gap", "hazard_prob", "markov_prob", "final_prob"]]
        .head(5)
        .to_string(index=False)
    )
    print(f"\n--- Stage 2: 上位 {len(pred.top_numbers)} 推奨数字 (lgbm_weight={pred.lgbm_weight}) ---")
    print(f"上位推奨数字: {pred.top_numbers}")
    print(
        pred.number_scores[
            [
                "rank",
                "number",
                "final_score",
                "bayes_score",
                "lgbm_score",
                "score_lift",
                "interval",
                "is_pull",
                "is_slide",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )
