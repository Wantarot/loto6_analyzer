"""Phase 5: Streamlit Dashboard for Loto 6 Physical Bias, Probability Structure & Wheeling Optimizer."""

from __future__ import annotations

import hashlib
import inspect
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_STRETCH_KW: dict[str, object] = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.button).parameters
    else {"use_container_width": True}
)

from src.auto_run import (
    LATEST_PREDICTION_PATH,
    LATEST_REPORT_PATH,
    PREDICTION_LOG_PATH,
    load_latest_prediction,
    load_prediction_log,
    run_auto_pipeline,
)
from src.backtest import (
    STRATEGY_LABELS,
    THEORETICAL_AVG_MATCHES,
    BacktestConfig,
    BacktestResult,
    run_walk_forward_backtest,
)
from src.features import (
    GLOBAL_MEAN_RATE,
    compute_set_ball_summary,
    verify_no_data_leakage,
)
from src.model import (
    DEFAULT_MIN_PAST_DRAWS,
    DEFAULT_START_DRAW_NO,
    Loto6PredictionEngine,
    PredictionConfig,
    filter_history_period,
    verify_model_no_data_leakage,
)
from src.optimizer import (
    OptimizerConfig,
    optimize_tickets,
)
from src.scraper import (
    DEFAULT_OUTPUT_PATH,
    MAIN_NUM_COLS,
    VALID_SET_BALLS,
    generate_dummy_data,
    import_manual_csv,
    load_history_csv,
    update_history_csv,
)

# ---------------------------------------------------------------------------
# Page Config & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ロト6 物理偏り・確率構造・買い目最適化ツール",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.loto-ball {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    font-weight: 700;
    font-size: 15px;
    margin: 2px 4px 2px 0;
    color: #ffffff;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}
.ball-main { background: linear-gradient(135deg, #2563eb, #1d4ed8); }
.ball-bonus { background: linear-gradient(135deg, #f59e0b, #d97706); }
.ball-pool { background: linear-gradient(135deg, #059669, #047857); }
.ball-ticket { background: linear-gradient(135deg, #0f766e, #115e59); }
.ticket-card {
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 10px;
    background: rgba(128, 128, 128, 0.05);
}
.ticket-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    font-size: 13px;
    opacity: 0.85;
}
.sb-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 13px;
    background: rgba(37, 99, 235, 0.16);
    border: 1px solid rgba(37, 99, 235, 0.45);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers & Cached Pipelines
# ---------------------------------------------------------------------------
def _format_yen(amount: int | float) -> str:
    val = int(round(amount))
    if val >= 100_000_000:
        return f"{val / 100_000_000:.2f}億円 ({val:,}円)"
    if val >= 10_000:
        return f"{val / 10_000:.1f}万円 ({val:,}円)"
    return f"{val:,}円"


def _render_balls_html(numbers: list[int], css_class: str = "ball-main", bonus: Optional[int] = None) -> str:
    parts = [f'<span class="loto-ball {css_class}">{int(n):02d}</span>' for n in numbers]
    if bonus is not None:
        parts.append(f'<span style="margin:0 4px;font-weight:700;opacity:0.7;">+</span>')
        parts.append(f'<span class="loto-ball ball-bonus" title="ボーナス数字">{int(bonus):02d}</span>')
    return "".join(parts)


def _compute_file_signature(path: Path) -> str:
    if not path.exists():
        return "missing"
    stat = path.stat()
    raw = f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


@st.cache_data(show_spinner=False)
def _load_history_cached(csv_path_str: str, file_sig: str) -> pd.DataFrame:
    _ = file_sig  # cache invalidation key
    return load_history_csv(Path(csv_path_str))


@st.cache_resource(show_spinner=False)
def _fit_engine_cached(
    csv_path_str: str,
    file_sig: str,
    start_draw_no: Optional[int],
    recent_n_draws: Optional[int],
    min_past_draws: int,
    bayes_prior_strength: float,
) -> Loto6PredictionEngine:
    """Fit Bayesian + LightGBM engine once per (dataset, period, model hyperparams).

    Changes to manual_set_ball, lgbm_weight, pool_size, n_tickets, or crowd penalty
    re-score instantaneously without retraining.
    """
    history_df = _load_history_cached(csv_path_str, file_sig)
    cfg = PredictionConfig(
        start_draw_no=start_draw_no,
        recent_n_draws=recent_n_draws,
        min_past_draws=min_past_draws,
        bayes_prior_strength=bayes_prior_strength,
    )
    return Loto6PredictionEngine(cfg).fit(history_df)


def _clear_all_caches() -> None:
    _load_history_cached.clear()
    _fit_engine_cached.clear()


# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
def render_sidebar() -> dict[str, object]:
    st.sidebar.title("🎯 ロト6 解析コントローラー")

    # 1. Data Management
    st.sidebar.subheader("1. データ取得・管理")
    col_dl1, col_dl2 = st.sidebar.columns(2)
    with col_dl1:
        if st.button("🔄 最新取得", **_STRETCH_KW, help="KYO's LOTO6＋ロト6生活から最新データを取得"):
            with st.spinner("最新データをオンライン取得中..."):
                try:
                    df_new = update_history_csv(DEFAULT_OUTPUT_PATH, fallback_to_dummy=False)
                    _clear_all_caches()
                    st.sidebar.success(f"更新完了: 第1〜{int(df_new['draw_no'].max())}回 ({len(df_new)}件)")
                    st.rerun()
                except Exception as exc:
                    st.sidebar.error(f"取得失敗: {exc}")
    with col_dl2:
        if st.button("🧪 ダミー生成", **_STRETCH_KW, help="テスト用500回分の疑似データを生成"):
            with st.spinner("ダミーデータを生成中..."):
                generate_dummy_data(n_draws=500, seed=42, output_path=DEFAULT_OUTPUT_PATH)
                _clear_all_caches()
                st.sidebar.warning("ダミーデータ (500回分) に切替えました。「最新取得」で実データへ戻せます。")
                st.rerun()

    with st.sidebar.expander("📂 手動CSVインポート", expanded=False):
        uploaded = st.file_uploader(
            "ロト6履歴CSV（UTF-8 / Shift-JIS対応）",
            type=["csv"],
            help="必須列: 開催回, 日付, 第1〜第6数字, ボーナス, セット球, キャリーオーバー",
        )
        if uploaded is not None:
            if st.button("このCSVを取り込む", **_STRETCH_KW):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = Path(tmp.name)
                    df_imp = import_manual_csv(tmp_path, output_path=DEFAULT_OUTPUT_PATH)
                    tmp_path.unlink(missing_ok=True)
                    _clear_all_caches()
                    st.success(f"インポート完了: {len(df_imp)}件")
                    st.rerun()
                except Exception as exc:
                    st.error(f"CSV読込エラー: {exc}")

    st.sidebar.divider()

    # 2. Analysis Period
    st.sidebar.subheader("2. 解析対象期間")
    period_choice = st.sidebar.radio(
        "学習・解析に使用する期間",
        options=["新球切り替わり後（第138回〜）", "全期間（第1回〜）", "直近N回"],
        index=0,
    )
    if period_choice == "新球切り替わり後（第138回〜）":
        start_draw_no: Optional[int] = DEFAULT_START_DRAW_NO
        recent_n_draws: Optional[int] = None
    elif period_choice == "全期間（第1回〜）":
        start_draw_no = 1
        recent_n_draws = None
    else:
        start_draw_no = 1
        recent_n_draws = int(
            st.sidebar.slider("直近の抽選回数", min_value=200, max_value=1800, value=500, step=50)
        )

    st.sidebar.divider()

    # 3. Prediction Engine Controls
    st.sidebar.subheader("3. 予測エンジン設定 (Phase 2)")
    sb_choice = st.sidebar.selectbox(
        "次回セット球の指定",
        options=["自動予測（確率重み付け周辺化）", *list(VALID_SET_BALLS)],
        index=0,
        help="自動予測ではStage 1のセット球確率分布で10球を重み付け統合します。",
    )
    manual_set_ball = None if sb_choice.startswith("自動予測") else sb_choice

    lgbm_weight = float(
        st.sidebar.slider(
            "LightGBM ブレンド比率（0=ベイズ単独, 1=LGBM単独）",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
        )
    )

    with st.sidebar.expander("⚙️ モデル詳細設定", expanded=False):
        min_past_draws = int(
            st.slider(
                "ウォームアップ期間 (min_past_draws)",
                min_value=50,
                max_value=300,
                value=DEFAULT_MIN_PAST_DRAWS,
                step=10,
            )
        )
        bayes_prior_strength = float(
            st.slider(
                "ベイズ事前強度（大きいほど6/43へ強く縮小）",
                min_value=5.0,
                max_value=100.0,
                value=30.0,
                step=5.0,
            )
        )

    st.sidebar.divider()

    # 4. Ticket Optimization Controls (4-5th Prize Hit Rate First)
    st.sidebar.subheader("4. 買い目最適化設定 (Phase 3)")
    pool_size = int(
        st.sidebar.slider(
            "上位プール数字数（推奨 10〜15）",
            min_value=8,
            max_value=15,
            value=12,
            step=1,
        )
    )
    n_tickets = int(
        st.sidebar.slider(
            "生成口数（1口200円）",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
        )
    )

    use_crowd_penalty = bool(
        st.sidebar.toggle(
            "大衆心理ペナルティをON（期待値重視）",
            value=False,
            help="デフォルトOFF（4〜5等的中の純粋な最大化を優先）。ONにすると誕生日偏重や連番・等差数列などを避けます。",
        )
    )
    crowd_penalty_weight = 0.25
    if use_crowd_penalty:
        crowd_penalty_weight = float(
            st.sidebar.slider("ペナルティ強度", min_value=0.05, max_value=0.80, value=0.25, step=0.05)
        )

    with st.sidebar.expander("🎛️ 構造フィルタ（緩やか設定）", expanded=False):
        use_sum_filter = st.checkbox("合計値フィルタを有効化", value=True)
        sum_range = (
            st.slider("合計値の範囲", min_value=21, max_value=243, value=(80, 185), step=5)
            if use_sum_filter
            else None
        )
        use_odd_filter = st.checkbox("奇数個数フィルタを有効化", value=True)
        odd_count_range = (
            st.slider("奇数個数の範囲", min_value=0, max_value=6, value=(1, 5), step=1)
            if use_odd_filter
            else None
        )
        max_consecutive_run = int(
            st.slider("許容する最大連番長", min_value=2, max_value=6, value=4, step=1)
        )

    return {
        "period_choice": period_choice,
        "start_draw_no": start_draw_no,
        "recent_n_draws": recent_n_draws,
        "manual_set_ball": manual_set_ball,
        "lgbm_weight": lgbm_weight,
        "min_past_draws": min_past_draws,
        "bayes_prior_strength": bayes_prior_strength,
        "pool_size": pool_size,
        "n_tickets": n_tickets,
        "use_crowd_penalty": use_crowd_penalty,
        "crowd_penalty_weight": crowd_penalty_weight,
        "sum_range": sum_range,
        "odd_count_range": odd_count_range,
        "max_consecutive_run": max_consecutive_run,
    }


# ---------------------------------------------------------------------------
# Tab 1: Next Draw Prediction & Optimized Wheeling Tickets
# ---------------------------------------------------------------------------
def render_tab_prediction(pred, opt, cfg: dict[str, object]) -> None:
    st.subheader(f"🔮 第{pred.target_draw_no}回 予測スコア＆最適買い目")
    st.caption(
        f"学習・集計範囲：第{pred.train_start_draw}回 〜 第{pred.train_end_draw}回（{pred.n_history_draws:,}回分・完全リーク防止）"
        f" ／ LightGBM重み: {pred.lgbm_weight:.2f} ／ "
        f"セット球モード: {'手動固定 (' + pred.manual_set_ball + ')' if pred.manual_set_ball else '自動確率周辺化'} ／ "
        f"大衆心理ペナルティ: {'ON' if cfg['use_crowd_penalty'] else 'OFF（的中率優先）'}"
    )

    # --- Section A: Optimized Tickets (Hero Section) ---
    st.markdown("### 🎫 最適化買い目（カバレッジ・ホイーリング）")
    m = opt.coverage_metrics

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "プール4個的中 → 5等以上カバー",
        f"{m['cov_4_hits_ge_5th'] * 100:.1f}%",
        help="上位プール内に当たりが4個入ったとき、作成した買い目で5等以上を拾い切れる厳密確率",
    )
    k2.metric(
        "プール3個的中 → 5等以上カバー",
        f"{m['cov_3_hits_ge_5th'] * 100:.1f}%",
        help="上位プール内に当たりが3個入ったとき、作成した買い目で5等を拾い切れる厳密確率",
    )
    k3.metric(
        "プール4個的中 → 4等以上カバー",
        f"{m['cov_4_hits_ge_4th'] * 100:.1f}%",
        help="上位プール内に当たりが4個入ったとき、4等以上を直接的中させる確率",
    )
    k4.metric(
        "買い目間の平均重複数字",
        f"{m['avg_pairwise_overlap']:.2f} 個",
        delta=f"最大 {int(m['max_pairwise_overlap'])} 個",
        delta_color="off",
    )
    k5.metric(
        "購入金額合計",
        f"{len(opt.tickets) * 200:,} 円",
        delta=f"{len(opt.tickets)}口 × 200円",
        delta_color="off",
    )

    if m.get("filters_relaxed", 0.0) > 0:
        st.info("指定の構造フィルタを満たす組み合わせが不足したため、フィルタを自動緩和して最適買い目を生成しました。")

    col_cards, col_pool = st.columns([1.45, 1.0])

    with col_cards:
        for row in opt.tickets_df.itertuples(index=False):
            nums = [int(row.num1), int(row.num2), int(row.num3), int(row.num4), int(row.num5), int(row.num6)]
            balls_html = _render_balls_html(nums, css_class="ball-ticket")
            st.markdown(
                f"""
                <div class="ticket-card">
                  <div class="ticket-header">
                    <strong>買い目 #{int(row.ticket_id)}</strong>
                    <span>合計: <b>{int(row.sum_value)}</b> ｜ 奇:偶 = <b>{int(row.odd_count)}:{int(row.even_count)}</b> ｜ スコア和: <b>{row.score_sum:.4f}</b> ｜ 大衆度: {row.crowd_penalty:.2f}</span>
                  </div>
                  <div>{balls_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        copy_text = "\n".join(
            f"口{int(r.ticket_id)}: {r.numbers_str} (合計={int(r.sum_value)}, 奇偶={int(r.odd_count)}:{int(r.even_count)})"
            for r in opt.tickets_df.itertuples(index=False)
        )
        with st.expander("📋 コピー用テキスト ＆ CSVダウンロード"):
            st.code(copy_text, language="text")
            csv_bytes = opt.tickets_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 買い目CSVをダウンロード",
                data=csv_bytes,
                file_name=f"loto6_draw{pred.target_draw_no}_tickets.csv",
                mime="text/csv",
                **_STRETCH_KW,
            )

    with col_pool:
        st.markdown(f"**採用上位プール（{len(opt.pool_numbers)}数字）**")
        st.markdown(_render_balls_html(opt.pool_numbers, css_class="ball-pool"), unsafe_allow_html=True)
        st.caption("※ 左から予測スコア順位が高い数字です。")

        usage_df = pd.DataFrame(
            {
                "number": [f"{int(n):02d}" for n in opt.number_usage.index],
                "usage_count": opt.number_usage.values.astype(int),
            }
        )
        fig_usage = px.bar(
            usage_df,
            x="number",
            y="usage_count",
            text="usage_count",
            title="プール各数字の使用回数（ホイーリング配置バランス）",
            labels={"number": "推奨数字", "usage_count": "採用口数"},
            color="usage_count",
            color_continuous_scale="Teal",
        )
        fig_usage.update_layout(height=280, margin=dict(l=10, r=10, t=42, b=10), coloraxis_showscale=False)
        fig_usage.update_traces(textposition="outside")
        st.plotly_chart(fig_usage, **_STRETCH_KW)

    st.divider()

    # --- Section B: Stage 1 Set Ball Prediction ---
    st.markdown("### 🎱 Stage 1：次回セット球予測（A〜J）")
    col_sb_chart, col_sb_tbl = st.columns([1.25, 1.0])

    sb_df = pred.set_ball_probs.copy()
    sb_plot_df = sb_df.sort_values("set_ball")
    top3_sbs = set(sb_df["set_ball"].head(3).tolist())
    sb_plot_df["tier"] = sb_plot_df["set_ball"].apply(lambda s: "上位3候補" if s in top3_sbs else "その他")

    with col_sb_chart:
        fig_sb = px.bar(
            sb_plot_df,
            x="set_ball",
            y="final_prob",
            color="tier",
            color_discrete_map={"上位3候補": "#2563eb", "その他": "#94a3b8"},
            text=sb_plot_df["final_prob"].map(lambda v: f"{v * 100:.1f}%"),
            title="セット球別 次回使用予測確率（破線=一様10%）",
            labels={"set_ball": "セット球", "final_prob": "予測確率", "tier": "区分"},
        )
        fig_sb.add_hline(y=0.10, line_dash="dash", line_color="#ef4444", annotation_text="理論一様 10%")
        fig_sb.update_yaxes(tickformat=".0%")
        fig_sb.update_traces(textposition="outside")
        fig_sb.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_sb, **_STRETCH_KW)

    with col_sb_tbl:
        sb_disp = sb_df[["set_ball", "last_gap", "hazard_prob", "markov_prob", "final_prob", "past_draws"]].copy()
        sb_disp["hazard_prob"] = (sb_disp["hazard_prob"] * 100).round(1)
        sb_disp["markov_prob"] = (sb_disp["markov_prob"] * 100).round(1)
        sb_disp["final_prob"] = (sb_disp["final_prob"] * 100).round(1)
        sb_disp.columns = ["セット球", "未出現間隔", "ハザード(%)", "マルコフ(%)", "統合確率(%)", "過去使用回"]
        st.dataframe(sb_disp, **_STRETCH_KW, hide_index=True, height=340)

    st.divider()

    # --- Section C: Stage 2 All 43 Numbers Score Ranking ---
    st.markdown("### 📊 Stage 2：全43数字 出現スコアランキング（ベイズ＋LightGBM統合）")
    scores_df = pred.number_scores.copy()
    pool_set = set(opt.pool_numbers)
    scores_by_num = scores_df.sort_values("number").copy()
    scores_by_num["区分"] = scores_by_num["number"].apply(lambda n: f"上位プール ({len(pool_set)}個)" if n in pool_set else "その他")

    fig_scores = px.bar(
        scores_by_num,
        x="number",
        y="final_score",
        color="区分",
        color_discrete_map={f"上位プール ({len(pool_set)}個)": "#059669", "その他": "#94a3b8"},
        hover_data={
            "rank": True,
            "final_score": ":.4f",
            "bayes_score": ":.4f",
            "lgbm_score": ":.4f",
            "score_lift": ":.3f",
            "interval": True,
            "is_pull": True,
            "is_slide": True,
        },
        title="全43数字の次回出現予測スコア（破線=理論平均 6/43 ≈ 0.1395）",
        labels={"number": "数字 (1〜43)", "final_score": "統合スコア"},
    )
    fig_scores.add_hline(
        y=GLOBAL_MEAN_RATE,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="理論平均 0.1395",
    )
    fig_scores.update_xaxes(dtick=1)
    fig_scores.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig_scores, **_STRETCH_KW)

    disp_scores = scores_df[
        [
            "rank",
            "number",
            "final_score",
            "bayes_score",
            "lgbm_score",
            "score_lift",
            "expected_set_ball_rate",
            "interval",
            "recent_5_count",
            "recent_10_count",
            "recent_30_count",
            "is_pull",
            "is_slide",
        ]
    ].copy()
    disp_scores.columns = [
        "順位",
        "数字",
        "統合スコア",
        "ベイズ",
        "LGBM",
        "理論比(倍)",
        "期待セット球率",
        "未出現間隔",
        "直近5回",
        "直近10回",
        "直近30回",
        "引っ張り",
        "スライド",
    ]
    st.dataframe(
        disp_scores.style.format(
            {
                "統合スコア": "{:.4f}",
                "ベイズ": "{:.4f}",
                "LGBM": "{:.4f}",
                "理論比(倍)": "{:.3f}",
                "期待セット球率": "{:.3%}",
            }
        ),
        **_STRETCH_KW,
        hide_index=True,
        height=380,
    )

    with st.expander("🔍 LightGBM 学習指標・特徴量重要度"):
        c_m1, c_m2, c_m3 = st.columns(3)
        mm = pred.model_metrics
        c_m1.metric("時系列検証 Log-Loss", f"{mm.get('val_log_loss', 0.0):.6f}")
        c_m2.metric("一様ベースライン比", f"{mm.get('log_loss_diff_vs_uniform', 0.0):+.6f}")
        c_m3.metric("最適木数 (early stopping)", f"{int(mm.get('best_iteration', 0))} 本")
        fig_fi = px.bar(
            pred.feature_importance.sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            title="LightGBM 特徴量重要度（分割回数）",
            labels={"importance": "重要度", "feature": "特徴量"},
        )
        fig_fi.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_fi, **_STRETCH_KW)


# ---------------------------------------------------------------------------
# Tab 2: Physical Bias & Probabilistic Structure EDA
# ---------------------------------------------------------------------------
def render_tab_eda(filtered_df: pd.DataFrame, pred, cfg: dict[str, object]) -> None:
    st.subheader("🔬 物理的偏り・確率構造の分析（EDA）")
    st.caption(
        f"分析対象：第{int(filtered_df['draw_no'].min())}回 〜 第{int(filtered_df['draw_no'].max())}回（{len(filtered_df):,}回分）"
    )

    # 1. Set-ball x Number Deviation Heatmap
    st.markdown("### 1. セット球別（A〜J）× 数字（1〜43）理論平均乖離ヒートマップ")
    sb_summary = compute_set_ball_summary(filtered_df)
    pivot_diff = sb_summary.pivot(index="set_ball", columns="number", values="rate_diff_from_global") * 100.0
    pivot_rate = sb_summary.pivot(index="set_ball", columns="number", values="appearance_rate") * 100.0
    pivot_cnt = sb_summary.pivot(index="set_ball", columns="number", values="appearances")

    customdata = np.stack([pivot_rate.to_numpy(), pivot_cnt.to_numpy()], axis=-1)
    max_abs = float(np.nanmax(np.abs(pivot_diff.to_numpy())))
    z_lim = max(4.0, round(max_abs, 1))

    fig_hm = go.Figure(
        data=go.Heatmap(
            z=pivot_diff.to_numpy(),
            x=pivot_diff.columns.tolist(),
            y=pivot_diff.index.tolist(),
            customdata=customdata,
            colorscale="RdBu_r",
            zmid=0.0,
            zmin=-z_lim,
            zmax=z_lim,
            colorbar=dict(title="乖離 (%pt)"),
            hovertemplate=(
                "セット球: <b>%{y}</b> / 数字: <b>%{x:02d}</b><br>"
                "出現率: %{customdata[0]:.2f}% (理論値 13.95%)<br>"
                "乖離: %{z:+.2f} %pt<br>"
                "出現回数: %{customdata[1]} 回<extra></extra>"
            ),
        )
    )
    fig_hm.update_xaxes(dtick=1, title="数字 (1〜43)")
    fig_hm.update_yaxes(title="セット球", autorange="reversed")
    fig_hm.update_layout(
        title="セット球ごとの数字出現率 乖離マップ（赤=出やすい / 青=出にくい）",
        height=400,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    st.plotly_chart(fig_hm, **_STRETCH_KW)

    # 2. Set-ball gap distribution & latest interval / pull / slide
    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown("### 2. セット球の使用間隔分布（物理ローテーション構造）")
        sbs = filtered_df["set_ball"].to_numpy()
        last_seen: dict[str, int] = {}
        gaps = []
        for idx, sb in enumerate(sbs):
            if sb in last_seen:
                gaps.append(idx - last_seen[sb] - 1)
            last_seen[sb] = idx
        gap_s = pd.Series(gaps, name="gap")
        fig_gap = px.histogram(
            gap_s,
            x="gap",
            nbins=int(gap_s.max()) + 1 if len(gap_s) > 0 else 15,
            title=f"同一セット球の再使用間隔（中央値: {gap_s.median():.0f}回 / 連続出現率: {(gap_s == 0).mean() * 100:.2f}%）",
            labels={"gap": "再使用までの間隔（回）", "count": "回数"},
            color_discrete_sequence=["#2563eb"],
        )
        fig_gap.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
        st.plotly_chart(fig_gap, **_STRETCH_KW)

    with col_e2:
        st.markdown("### 3. 最新インターバル ＆ 引っ張り・スライド候補")
        sc = pred.number_scores.sort_values("number").copy()

        def _status_label(row) -> str:
            if row["is_pull"] == 1:
                return "前回出現（引っ張り候補）"
            if row["is_slide"] == 1:
                return "前回±1（スライド候補）"
            return "通常"

        sc["状態"] = sc.apply(_status_label, axis=1)
        fig_int = px.bar(
            sc,
            x="number",
            y="interval",
            color="状態",
            color_discrete_map={
                "前回出現（引っ張り候補）": "#dc2626",
                "前回±1（スライド候補）": "#f59e0b",
                "通常": "#64748b",
            },
            title="全43数字の連続未出現回数（インターバル）",
            labels={"number": "数字 (1〜43)", "interval": "連続未出現回数"},
        )
        fig_int.update_xaxes(dtick=2)
        fig_int.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_int, **_STRETCH_KW)

    # 3. Sum distribution & Odd/Even distribution
    col_e3, col_e4 = st.columns(2)
    main_arr = filtered_df[MAIN_NUM_COLS].to_numpy(dtype=int)
    sums = main_arr.sum(axis=1)
    odd_counts = (main_arr % 2 == 1).sum(axis=1)

    with col_e3:
        st.markdown("### 4. 本数字合計値の分布")
        fig_sum = px.histogram(
            pd.DataFrame({"sum_value": sums}),
            x="sum_value",
            nbins=35,
            title=f"本数字合計値のヒストグラム（平均: {sums.mean():.1f}, 標準偏差: {sums.std():.1f}）",
            labels={"sum_value": "本数字の合計値", "count": "回数"},
            color_discrete_sequence=["#059669"],
        )
        sum_range = cfg.get("sum_range")
        if sum_range is not None:
            fig_sum.add_vrect(
                x0=sum_range[0],
                x1=sum_range[1],
                fillcolor="rgba(37, 99, 235, 0.12)",
                line_width=1,
                line_dash="dash",
                annotation_text=f"現在フィルタ ({sum_range[0]}〜{sum_range[1]})",
                annotation_position="top left",
            )
            in_ratio = float(((sums >= sum_range[0]) & (sums <= sum_range[1])).mean() * 100.0)
            st.caption(f"現在の合計値フィルタ範囲（{sum_range[0]}〜{sum_range[1]}）は過去実抽選の **{in_ratio:.1f}%** をカバーしています。")
        fig_sum.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
        st.plotly_chart(fig_sum, **_STRETCH_KW)

    with col_e4:
        st.markdown("### 5. 奇数：偶数 バランス分布")
        oe_counts = pd.Series(odd_counts).value_counts().sort_index()
        oe_df = pd.DataFrame(
            {
                "奇:偶": [f"{k}:{6 - k}" for k in oe_counts.index],
                "回数": oe_counts.values,
                "割合(%)": (oe_counts.values / len(filtered_df) * 100.0).round(1),
            }
        )
        fig_oe = px.bar(
            oe_df,
            x="奇:偶",
            y="割合(%)",
            text=oe_df["割合(%)"].map(lambda v: f"{v:.1f}%"),
            title="本数字 奇数：偶数 出現構成比",
            color="割合(%)",
            color_continuous_scale="Blues",
        )
        fig_oe.update_traces(textposition="outside")
        fig_oe.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_oe, **_STRETCH_KW)


# ---------------------------------------------------------------------------
# Tab 3: Walk-Forward Backtesting (Phase 4)
# ---------------------------------------------------------------------------
def render_tab_backtest(history_df: pd.DataFrame, cfg: dict[str, object]) -> None:
    st.subheader("📈 ウォークフォワード検証（完全リーク防止バックテスト）")
    st.caption(
        "各検証回 N について第 N-1 回以前のデータのみで学習・予測・ホイーリング最適化を行い、"
        "「予測の力」と「ホイーリングの力」を4戦略で同一期間比較します。"
    )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
    with c1:
        n_eval_draws = int(st.slider("検証回数（直近）", min_value=30, max_value=250, value=100, step=10))
    with c2:
        retrain_every = int(st.slider("LGBM再学習間隔（回）", min_value=1, max_value=50, value=10, step=1))
    with c3:
        n_random_trials = int(st.slider("ランダムMC試行数", min_value=10, max_value=100, value=50, step=10))
    with c4:
        st.write("")
        run_bt = st.button("🚀 ウォークフォワード検証を実行", type="primary", **_STRETCH_KW)

    if run_bt:
        bt_cfg = BacktestConfig(
            n_eval_draws=n_eval_draws,
            retrain_every=retrain_every,
            start_draw_no=cfg["start_draw_no"],
            recent_n_draws=cfg["recent_n_draws"],
            min_past_draws=int(cfg["min_past_draws"]),
            lgbm_weight=float(cfg["lgbm_weight"]),
            bayes_prior_strength=float(cfg["bayes_prior_strength"]),
            pool_size=int(cfg["pool_size"]),
            n_tickets=int(cfg["n_tickets"]),
            use_crowd_penalty=bool(cfg["use_crowd_penalty"]),
            crowd_penalty_weight=float(cfg["crowd_penalty_weight"]),
            sum_range=cfg["sum_range"],
            odd_count_range=cfg["odd_count_range"],
            max_consecutive_run=int(cfg["max_consecutive_run"]),
            n_random_trials=n_random_trials,
        )
        progress_bar = st.progress(0, text="ウォークフォワード検証を準備中...")

        def _on_progress(step: int, total: int, draw_no: int) -> None:
            progress_bar.progress(
                int(step / total * 100),
                text=f"検証中: 第{draw_no}回 ({step}/{total}回完了)",
            )

        try:
            bt_result = run_walk_forward_backtest(history_df, config=bt_cfg, progress_callback=_on_progress)
            progress_bar.empty()
            st.session_state["last_backtest_result"] = bt_result
            st.success(
                f"検証完了: 第{bt_result.eval_start_draw}回〜第{bt_result.eval_end_draw}回"
                f"（{bt_result.n_eval_draws}回 / 所要 {bt_result.elapsed_seconds:.1f}秒）"
            )
        except Exception as exc:
            progress_bar.empty()
            st.error(f"バックテスト実行エラー: {exc}")

    bt_res: Optional[BacktestResult] = st.session_state.get("last_backtest_result")
    if bt_res is None:
        st.info("上の「🚀 ウォークフォワード検証を実行」ボタンを押すと、サイドバーの現在設定で検証を開始します（直近100回・再学習間隔10回で約30〜45秒）。")
    else:
        _render_backtest_results(bt_res)

    st.divider()
    with st.expander("🛡️ データリーク防止 セルフチェック"):
        st.write("第K回以降のデータを削除・改ざんしても、第K回の特徴量およびモデル予測スコアが1ビットも変化しないことを検証します。")
        if st.button("リーク防止テストを実行"):
            with st.spinner("特徴量＆モデルのリーク検証を実行中..."):
                latest_draw = int(history_df["draw_no"].max())
                feat_rep = verify_no_data_leakage(history_df, set_ball_mode="prev")
                model_rep = verify_model_no_data_leakage(
                    history_df,
                    test_draw_no=latest_draw,
                    config=PredictionConfig(
                        start_draw_no=cfg["start_draw_no"],
                        recent_n_draws=cfg["recent_n_draws"],
                        min_past_draws=int(cfg["min_past_draws"]),
                    ),
                )
            st.success(
                f"✅ PASS — 特徴量検証（検証回: {feat_rep['checked_draws']}）＆ "
                f"モデル検証（第{model_rep['test_draw_no']}回を第{model_rep['as_of_draw_no']}回までのデータで検証）ともにデータリーク 0 件！"
            )


def _render_backtest_results(bt_res: BacktestResult) -> None:
    summ = bt_res.summary_df
    prop = summ[summ["strategy"] == "blend_wheel"].iloc[0]
    rand = summ[summ["strategy"] == "random_mc"].iloc[0]
    naive = summ[summ["strategy"] == "blend_naive"].iloc[0]
    sb = bt_res.set_ball_eval
    tb = bt_res.theoretical_benchmarks

    st.caption(
        f"検証結果：第{bt_res.eval_start_draw}回〜第{bt_res.eval_end_draw}回（{bt_res.n_eval_draws}回 × {bt_res.config.n_tickets}口 / "
        f"プール{bt_res.config.pool_size}数字 / LGBM再学習 {bt_res.metadata.get('retrain_count')}回）"
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        "1口平均一致数字数（提案戦略）",
        f"{prop['avg_matches_per_ticket']:.3f} 個",
        delta=f"理論値 {THEORETICAL_AVG_MATCHES:.3f} 比 {prop['matches_diff_vs_theory']:+.3f}",
    )
    m2.metric(
        "回別 5等以上的中率",
        f"{prop['draw_hit_rate_ge_5th_pct']:.1f}%",
        delta=f"単純上位比 {prop['draw_hit_rate_ge_5th_pct'] - naive['draw_hit_rate_ge_5th_pct']:+.1f}pt",
    )
    m3.metric(
        "上位プール平均包含数",
        f"{prop['avg_pool_hits']:.3f} 個",
        delta=f"理論値 {tb['avg_pool_hits']:.3f} 比 {prop['avg_pool_hits'] - tb['avg_pool_hits']:+.3f}",
    )
    m4.metric(
        "実賞金ベース ROI",
        f"{prop['roi_pct']:.1f}%",
        delta=f"ランダム比 {prop['roi_pct'] - rand['roi_pct']:+.1f}pt",
    )
    m5.metric(
        "Stage 1 セット球OOS精度",
        f"Top-1: {sb['top1_acc_pct']:.1f}%",
        delta=f"Top-3: {sb['top3_acc_pct']:.1f}% (ランダム30%)",
        delta_color="off",
    )

    # Summary Table
    disp_cols = [
        "strategy_label",
        "avg_matches_per_ticket",
        "matches_diff_vs_theory",
        "avg_pool_hits",
        "pool_ge3_capture_pct",
        "pool_ge4_capture_pct",
        "prize4_count",
        "prize5_count",
        "draw_hit_rate_ge_5th_pct",
        "total_payout",
        "roi_pct",
    ]
    disp = summ[disp_cols].copy()
    disp.columns = [
        "戦略",
        "1口平均一致",
        "理論値差",
        "プール包含数",
        "3個包含→5等捕獲%",
        "4個包含→5等捕獲%",
        "4等口数",
        "5等口数",
        "回別5等以上率%",
        "総払戻(円)",
        "ROI(%)",
    ]
    st.dataframe(
        disp.style.format(
            {
                "1口平均一致": "{:.3f}",
                "理論値差": "{:+.3f}",
                "プール包含数": "{:.3f}",
                "3個包含→5等捕獲%": "{:.1f}%",
                "4個包含→5等捕獲%": "{:.1f}%",
                "4等口数": "{:.1f}",
                "5等口数": "{:.1f}",
                "回別5等以上率%": "{:.1f}%",
                "総払戻(円)": "{:,.0f}",
                "ROI(%)": "{:.1f}%",
            },
            na_rep="—",
        ),
        **_STRETCH_KW,
        hide_index=True,
    )

    # Charts Row 1: Avg Matches & Hit Rate
    col_c1, col_c2 = st.columns(2)
    color_map = {
        STRATEGY_LABELS["blend_wheel"]: "#2563eb",
        STRATEGY_LABELS["bayes_wheel"]: "#059669",
        STRATEGY_LABELS["blend_naive"]: "#f59e0b",
        STRATEGY_LABELS["random_mc"]: "#94a3b8",
    }

    with col_c1:
        fig_match = px.bar(
            summ,
            x="strategy_label",
            y="avg_matches_per_ticket",
            color="strategy_label",
            color_discrete_map=color_map,
            text=summ["avg_matches_per_ticket"].map(lambda v: f"{v:.3f}"),
            title="1口あたり平均一致数字数（破線=ランダム理論値 0.8372）",
            labels={"strategy_label": "戦略", "avg_matches_per_ticket": "平均一致数字数"},
        )
        fig_match.add_hline(
            y=THEORETICAL_AVG_MATCHES,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"理論値 {THEORETICAL_AVG_MATCHES:.4f}",
        )
        fig_match.update_yaxes(range=[0.7, max(1.0, float(summ["avg_matches_per_ticket"].max()) * 1.12)])
        fig_match.update_traces(textposition="outside")
        fig_match.update_layout(height=360, showlegend=False, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_match, **_STRETCH_KW)

    with col_c2:
        fig_hit = px.bar(
            summ,
            x="strategy_label",
            y="draw_hit_rate_ge_5th_pct",
            color="strategy_label",
            color_discrete_map=color_map,
            text=summ["draw_hit_rate_ge_5th_pct"].map(lambda v: f"{v:.1f}%"),
            title=f"回別 5等以上的中率（破線={bt_res.config.n_tickets}口ランダム理論値 {tb['draw_ge_5th_prob_pct']:.1f}%）",
            labels={"strategy_label": "戦略", "draw_hit_rate_ge_5th_pct": "的中率 (%)"},
        )
        fig_hit.add_hline(
            y=tb["draw_ge_5th_prob_pct"],
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"理論値 {tb['draw_ge_5th_prob_pct']:.1f}%",
        )
        fig_hit.update_traces(textposition="outside")
        fig_hit.update_layout(height=360, showlegend=False, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_hit, **_STRETCH_KW)

    # Charts Row 2: Cumulative Profit & Cumulative ROI
    col_c3, col_c4 = st.columns(2)
    pddf = bt_res.per_draw_df

    with col_c3:
        fig_prof = px.line(
            pddf,
            x="draw_no",
            y="cum_profit",
            color="strategy_label",
            color_discrete_map=color_map,
            title="累積純収支の推移（円）",
            labels={"draw_no": "開催回", "cum_profit": "累積収支 (円)", "strategy_label": "戦略"},
        )
        fig_prof.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_prof.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_prof, **_STRETCH_KW)

    with col_c4:
        fig_roi = px.line(
            pddf,
            x="draw_no",
            y="cum_roi_pct",
            color="strategy_label",
            color_discrete_map=color_map,
            title="累積回収率 (ROI%) の推移",
            labels={"draw_no": "開催回", "cum_roi_pct": "累積 ROI (%)", "strategy_label": "戦略"},
        )
        fig_roi.add_hline(y=100, line_dash="dot", line_color="#ef4444", annotation_text="損益分岐 100%")
        fig_roi.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_roi, **_STRETCH_KW)

    with st.expander("📄 開催回別バックテスト詳細ログ"):
        strat_filter = st.selectbox(
            "表示する戦略",
            options=list(STRATEGY_LABELS.values()),
            index=0,
        )
        log_df = pddf[pddf["strategy_label"] == strat_filter].sort_values("draw_no", ascending=False)
        st.dataframe(log_df, **_STRETCH_KW, hide_index=True, height=340)
        st.download_button(
            "📥 バックテスト全ログCSVダウンロード",
            data=pddf.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"loto6_backtest_{bt_res.eval_start_draw}_{bt_res.eval_end_draw}.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Tab 4: Historical Data Management & Integrity
# ---------------------------------------------------------------------------
def render_tab_history(history_df: pd.DataFrame) -> None:
    st.subheader("🗂️ 過去データ管理・整合性チェック")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("総収録回数", f"{len(history_df):,} 回")
    c2.metric("期間", f"第{int(history_df['draw_no'].min())}〜{int(history_df['draw_no'].max())}回")
    c3.metric("欠損セル数", f"{int(history_df.isna().sum().sum())} 件")
    c4.metric("キャリーオーバー発生率", f"{(history_df['carryover'] > 0).mean() * 100:.1f}%")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        recent_co = history_df.tail(150).copy()
        recent_co["carryover_oku"] = recent_co["carryover"] / 100_000_000.0
        fig_co = px.area(
            recent_co,
            x="draw_no",
            y="carryover_oku",
            title="直近150回のキャリーオーバー額推移（億円）",
            labels={"draw_no": "開催回", "carryover_oku": "キャリーオーバー (億円)"},
            color_discrete_sequence=["#f59e0b"],
        )
        fig_co.update_layout(height=300, margin=dict(l=10, r=10, t=42, b=10))
        st.plotly_chart(fig_co, **_STRETCH_KW)

    with col_h2:
        sb_cnt = history_df["set_ball"].value_counts().reindex(list(VALID_SET_BALLS), fill_value=0)
        fig_sbc = px.bar(
            x=sb_cnt.index,
            y=sb_cnt.values,
            text=sb_cnt.values,
            title="全収録期間のセット球別 登場回数",
            labels={"x": "セット球", "y": "登場回数"},
            color_discrete_sequence=["#2563eb"],
        )
        fig_sbc.update_traces(textposition="outside")
        fig_sbc.update_layout(height=300, margin=dict(l=10, r=10, t=42, b=10))
        st.plotly_chart(fig_sbc, **_STRETCH_KW)

    st.markdown("### 📋 抽選履歴一覧")
    f1, f2 = st.columns([1, 2])
    with f1:
        sb_sel = st.multiselect("セット球フィルタ", options=list(VALID_SET_BALLS), default=[])
    with f2:
        n_show = int(st.slider("表示件数（最新から）", min_value=20, max_value=min(500, len(history_df)), value=50, step=10))

    view_df = history_df.sort_values("draw_no", ascending=False)
    if sb_sel:
        view_df = view_df[view_df["set_ball"].isin(sb_sel)]
    st.dataframe(view_df.head(n_show), **_STRETCH_KW, hide_index=True, height=400)

    st.download_button(
        "📥 全履歴CSVをダウンロード",
        data=history_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="loto6_history.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Tab 4: Live Automated Performance Log (prediction_log.csv)
# ---------------------------------------------------------------------------
def render_tab_live_performance(cfg: dict[str, object]) -> None:
    st.subheader("🏆 自動運用 成績ログ（リアル照合履歴）")
    st.caption(
        "GitHub Actions または `python -m src.auto_run` が保存した次回予測（`data/latest_prediction.json`）と、"
        "実際の抽選結果を自動照合した実運用ログ（`data/prediction_log.csv`）を表示します。"
    )

    col_act1, col_act2 = st.columns([1.4, 2.6])
    with col_act1:
        if st.button("🤖 自動更新・照合・レポート生成を実行", type="primary", **_STRETCH_KW):
            with st.spinner("最新データ取得・前回結果照合・次回レポート生成を実行中..."):
                try:
                    res = run_auto_pipeline(
                        notify=False,
                        start_draw_no=int(cfg["start_draw_no"] or DEFAULT_START_DRAW_NO),
                        min_past_draws=int(cfg["min_past_draws"]),
                        lgbm_weight=float(cfg["lgbm_weight"]),
                        pool_size=int(cfg["pool_size"]),
                        n_tickets=int(cfg["n_tickets"]),
                        use_crowd_penalty=bool(cfg["use_crowd_penalty"]),
                    )
                    _clear_all_caches()
                    eval_msg = "前回予測の照合結果を追記しました。" if res["evaluated"] else "照合待ちの開催回はまだ未抽選でした。"
                    st.success(f"完了（最新: 第{res['latest_draw_no']}回 → 次回予測: 第{res['target_draw_no']}回） / {eval_msg}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"自動バッチ実行エラー: {exc}")
    with col_act2:
        st.caption(
            "※ 画面からの実行時は Discord / Gmail 通知は送信しません（通知テストは CLI の `python -m src.auto_run` または GitHub Actions から実行してください）。"
        )

    log_df = load_prediction_log(PREDICTION_LOG_PATH)

    if log_df.empty:
        st.info(
            "まだ照合済みの運用成績ログ（`data/prediction_log.csv`）がありません。"
            "次回予測を `data/latest_prediction.json` に保存した状態で次の抽選結果が公開されると、自動で照合・蓄積されます。"
        )
    else:
        n_draws = len(log_df)
        tot_cost = int(log_df["total_cost"].sum())
        tot_payout = int(log_df["total_payout"].sum())
        tot_profit = tot_payout - tot_cost
        roi_pct = (tot_payout / tot_cost * 100.0) if tot_cost > 0 else 0.0
        avg_matches = float(log_df["avg_matches_per_ticket"].mean())
        hit_draws = int((log_df["total_payout"] > 0).sum())
        hit_rate_pct = (hit_draws / n_draws) * 100.0
        p4_tot = int(log_df["prize4_count"].sum())
        p5_tot = int(log_df["prize5_count"].sum())
        p123_tot = int(log_df["prize1_count"].sum() + log_df["prize2_count"].sum() + log_df["prize3_count"].sum())
        sb_top1_pct = float(log_df["set_ball_top1_hit"].mean() * 100.0)
        sb_top3_pct = float(log_df["set_ball_top3_hit"].mean() * 100.0)

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("照合済み回数", f"{n_draws} 回")
        k2.metric("累積投資額", f"{tot_cost:,} 円")
        k3.metric("累積払戻額", f"{tot_payout:,} 円", delta=f"収支 {tot_profit:+,} 円")
        k4.metric("通算回収率 (ROI)", f"{roi_pct:.1f}%")
        k5.metric(
            "1口平均一致数",
            f"{avg_matches:.3f} 個",
            delta=f"理論値比 {avg_matches - THEORETICAL_AVG_MATCHES:+.3f}",
        )
        k6.metric(
            "回別 的中率 (5等以上)",
            f"{hit_rate_pct:.1f}%",
            delta=f"1-3等:{p123_tot} / 4等:{p4_tot} / 5等:{p5_tot}",
            delta_color="off",
        )
        st.caption(
            f"Stage 1 セット球予測 実運用精度：Top-1 **{sb_top1_pct:.1f}%**（ランダム10%） ／ Top-3 **{sb_top3_pct:.1f}%**（ランダム30%）"
        )

        plot_df = log_df.copy()
        plot_df["cum_cost"] = plot_df["total_cost"].cumsum()
        plot_df["cum_payout"] = plot_df["total_payout"].cumsum()
        plot_df["cum_profit"] = plot_df["cum_payout"] - plot_df["cum_cost"]
        plot_df["cum_roi_pct"] = (plot_df["cum_payout"] / plot_df["cum_cost"]) * 100.0

        c_l1, c_l2 = st.columns(2)
        with c_l1:
            fig_live_pl = go.Figure()
            fig_live_pl.add_trace(
                go.Scatter(
                    x=plot_df["draw_no"],
                    y=plot_df["cum_cost"],
                    mode="lines+markers",
                    name="累積投資 (円)",
                    line=dict(color="#64748b", dash="dot"),
                )
            )
            fig_live_pl.add_trace(
                go.Scatter(
                    x=plot_df["draw_no"],
                    y=plot_df["cum_payout"],
                    mode="lines+markers",
                    name="累積払戻 (円)",
                    line=dict(color="#059669", width=2.5),
                )
            )
            fig_live_pl.add_trace(
                go.Scatter(
                    x=plot_df["draw_no"],
                    y=plot_df["cum_profit"],
                    mode="lines+markers",
                    name="累積純収支 (円)",
                    line=dict(color="#2563eb", width=2.5),
                )
            )
            fig_live_pl.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_live_pl.update_layout(
                title="累積投資・払戻・純収支の推移",
                xaxis_title="開催回",
                yaxis_title="金額 (円)",
                height=340,
                margin=dict(l=10, r=10, t=45, b=10),
                legend=dict(orientation="h", y=-0.22),
            )
            st.plotly_chart(fig_live_pl, **_STRETCH_KW)

        with c_l2:
            fig_live_m = go.Figure()
            fig_live_m.add_trace(
                go.Bar(
                    x=plot_df["draw_no"],
                    y=plot_df["pool_hits"],
                    name="上位プール包含数",
                    marker_color="rgba(5, 150, 105, 0.35)",
                )
            )
            fig_live_m.add_trace(
                go.Scatter(
                    x=plot_df["draw_no"],
                    y=plot_df["avg_matches_per_ticket"],
                    mode="lines+markers",
                    name="1口平均一致数",
                    line=dict(color="#2563eb", width=2.5),
                )
            )
            fig_live_m.add_hline(
                y=THEORETICAL_AVG_MATCHES,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"理論値 {THEORETICAL_AVG_MATCHES:.3f}",
            )
            fig_live_m.update_layout(
                title="開催回別：1口平均一致数 ＆ プール包含数",
                xaxis_title="開催回",
                yaxis_title="一致個数",
                height=340,
                margin=dict(l=10, r=10, t=45, b=10),
                legend=dict(orientation="h", y=-0.22),
            )
            st.plotly_chart(fig_live_m, **_STRETCH_KW)

        st.markdown("### 📋 的中・照合履歴一覧")
        disp_cols = [
            "draw_no",
            "draw_date",
            "actual_numbers",
            "actual_bonus",
            "actual_set_ball",
            "pred_set_ball_top3",
            "pool_hits",
            "best_ticket_matches",
            "avg_matches_per_ticket",
            "best_prize",
            "total_cost",
            "total_payout",
            "profit",
        ]
        disp = log_df.sort_values("draw_no", ascending=False)[disp_cols].copy()
        disp.columns = [
            "開催回",
            "抽選日",
            "当選本数字",
            "B",
            "実セット球",
            "予測Top3",
            "プール包含",
            "最大一致",
            "1口平均一致",
            "最高等級",
            "投資(円)",
            "払戻(円)",
            "収支(円)",
        ]
        st.dataframe(
            disp.style.format({"1口平均一致": "{:.3f}", "投資(円)": "{:,}", "払戻(円)": "{:,}", "収支(円)": "{:+,}"}),
            **_STRETCH_KW,
            hide_index=True,
            height=340,
        )
        st.download_button(
            "📥 運用成績ログCSVをダウンロード",
            data=log_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="prediction_log.csv",
            mime="text/csv",
        )

    st.divider()

    # Show currently saved prediction awaiting next draw
    saved_pred = load_latest_prediction(LATEST_PREDICTION_PATH)
    if saved_pred is not None:
        st.markdown(f"### ⏳ 照合待ちの保存済み予測（第{saved_pred['target_draw_no']}回向け）")
        st.caption(
            f"保存日時: {saved_pred.get('generated_at_jst', '-')} ／ 基準データ: 第{saved_pred.get('as_of_draw_no', '-')}回まで"
        )
        sb3_str = " / ".join(
            f"{x['set_ball']} ({x['prob'] * 100:.1f}%)" for x in saved_pred.get("set_ball_top3", [])
        )
        st.markdown(f"- **予測セット球 Top3**: `{sb3_str}`")
        st.markdown(_render_balls_html(saved_pred.get("pool_numbers", []), css_class="ball-pool"), unsafe_allow_html=True)

        saved_tickets_df = pd.DataFrame(saved_pred.get("tickets", []))
        if not saved_tickets_df.empty:
            st.dataframe(
                saved_tickets_df[["ticket_id", "numbers_str", "sum_value", "odd_count", "even_count", "score_sum"]].rename(
                    columns={
                        "ticket_id": "口",
                        "numbers_str": "保存済み買い目",
                        "sum_value": "合計値",
                        "odd_count": "奇数",
                        "even_count": "偶数",
                        "score_sum": "スコア和",
                    }
                ),
                **_STRETCH_KW,
                hide_index=True,
            )

    if LATEST_REPORT_PATH.exists():
        with st.expander("📝 最新自動レポート（data/latest_report.md）を表示"):
            st.markdown(LATEST_REPORT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Main App Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = render_sidebar()

    st.title("🎯 ロト6 物理偏り・確率構造・最適買い目アナライザー")

    if not DEFAULT_OUTPUT_PATH.exists():
        st.warning(f"過去データファイルが見つかりません: `{DEFAULT_OUTPUT_PATH}`")
        st.info("左サイドバーの「🔄 最新取得」「🧪 ダミー生成」または「📂 手動CSVインポート」からデータを用意してください。")
        return

    file_sig = _compute_file_signature(DEFAULT_OUTPUT_PATH)
    try:
        history_df = _load_history_cached(str(DEFAULT_OUTPUT_PATH), file_sig)
    except Exception as exc:
        st.error(f"履歴データの読み込みに失敗しました: {exc}")
        return

    # Top Latest Draw Banner
    latest = history_df.iloc[-1]
    latest_nums = [int(latest[c]) for c in MAIN_NUM_COLS]
    b1, b2, b3, b4 = st.columns([1.0, 2.2, 0.9, 1.3])
    b1.metric("最新開催回", f"第{int(latest['draw_no'])}回", delta=str(latest["draw_date"]), delta_color="off")
    with b2:
        st.markdown("**最新当選番号（＋ボーナス）**")
        st.markdown(_render_balls_html(latest_nums, css_class="ball-main", bonus=int(latest["bonus"])), unsafe_allow_html=True)
    with b3:
        st.markdown("**前回セット球**")
        st.markdown(f'<div style="margin-top:6px;"><span class="sb-badge">セット球 {latest["set_ball"]}</span></div>', unsafe_allow_html=True)
    b4.metric("次回キャリーオーバー", _format_yen(int(latest["carryover"])))

    st.divider()

    # Fit / Retrieve Cached Prediction Engine & Compute Next-Draw Prediction + Optimal Tickets
    try:
        with st.spinner("予測エンジン（階層ベイズ＋LightGBM）を準備中..."):
            engine = _fit_engine_cached(
                csv_path_str=str(DEFAULT_OUTPUT_PATH),
                file_sig=file_sig,
                start_draw_no=cfg["start_draw_no"],
                recent_n_draws=cfg["recent_n_draws"],
                min_past_draws=int(cfg["min_past_draws"]),
                bayes_prior_strength=float(cfg["bayes_prior_strength"]),
            )
            pred = engine.predict_next(
                manual_set_ball=cfg["manual_set_ball"],
                lgbm_weight=float(cfg["lgbm_weight"]),
                top_k=int(cfg["pool_size"]),
            )
            opt = optimize_tickets(
                pred.number_scores,
                OptimizerConfig(
                    pool_size=int(cfg["pool_size"]),
                    n_tickets=int(cfg["n_tickets"]),
                    use_crowd_penalty=bool(cfg["use_crowd_penalty"]),
                    crowd_penalty_weight=float(cfg["crowd_penalty_weight"]),
                    sum_range=cfg["sum_range"],
                    odd_count_range=cfg["odd_count_range"],
                    max_consecutive_run=int(cfg["max_consecutive_run"]),
                ),
            )
            filtered_df = filter_history_period(
                history_df,
                start_draw_no=cfg["start_draw_no"],
                recent_n_draws=cfg["recent_n_draws"],
                already_validated=True,
            )
    except Exception as exc:
        st.error(f"予測または最適化の実行中にエラーが発生しました: {exc}")
        return

    # 5 Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🔮 次回予測＆最適買い目",
            "🔬 物理偏り・確率構造 (EDA)",
            "📈 ウォークフォワード検証",
            "🏆 運用成績ログ",
            "🗂️ 過去データ管理",
        ]
    )

    with tab1:
        render_tab_prediction(pred, opt, cfg)
    with tab2:
        render_tab_eda(filtered_df, pred, cfg)
    with tab3:
        render_tab_backtest(history_df, cfg)
    with tab4:
        render_tab_live_performance(cfg)
    with tab5:
        render_tab_history(history_df)


if __name__ == "__main__":
    main()
