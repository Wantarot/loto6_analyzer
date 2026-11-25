"""Automated Loto 6 pipeline for GitHub Actions & local scheduled execution.

Workflow executed by `python -m src.auto_run`:
  1. Fetch latest draw results & update `data/loto6_history.csv` (with retries).
  2. If `data/latest_prediction.json` exists and its `target_draw_no` is now present
     in `data/loto6_history.csv`, evaluate the saved tickets/pool/set-ball predictions
     against official results and append to `data/prediction_log.csv` (idempotent).
  3. Fit the leak-free Phase 2 prediction engine on all available history and generate
     next-draw predictions (Top-3 set balls, Top-12 pool, 5 wheeled tickets).
  4. Save machine-readable `data/latest_prediction.json` and human-readable Markdown
     `data/latest_report.md`.
  5. Send notification via Discord Webhook (`DISCORD_WEBHOOK_URL`) and/or Gmail SMTP
     (`GMAIL_USER`, `GMAIL_APP_PASSWORD`, optional `GMAIL_TO`) if configured; silently
     skip if unset.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import smtplib
import time
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

from src.backtest import (
    THEORETICAL_AVG_MATCHES,
    _extract_draw_payouts,
    evaluate_tickets_against_draw,
)
from src.model import (
    DEFAULT_MIN_PAST_DRAWS,
    DEFAULT_START_DRAW_NO,
    Loto6PredictionEngine,
    PredictionConfig,
)
from src.optimizer import OptimizerConfig, optimize_tickets
from src.scraper import (
    DEFAULT_OUTPUT_PATH,
    MAIN_NUM_COLS,
    PROJECT_ROOT,
    load_history_csv,
    update_history_csv,
)

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

DATA_DIR = PROJECT_ROOT / "data"
LATEST_PREDICTION_PATH = DATA_DIR / "latest_prediction.json"
LATEST_REPORT_PATH = DATA_DIR / "latest_report.md"
PREDICTION_LOG_PATH = DATA_DIR / "prediction_log.csv"

PREDICTION_LOG_COLS = [
    "draw_no",
    "draw_date",
    "evaluated_at_jst",
    "actual_numbers",
    "actual_bonus",
    "actual_set_ball",
    "pred_set_ball_top1",
    "pred_set_ball_top3",
    "set_ball_top1_hit",
    "set_ball_top3_hit",
    "pool_size",
    "pool_numbers",
    "pool_hits",
    "n_tickets",
    "tickets_str",
    "best_ticket_matches",
    "avg_matches_per_ticket",
    "prize1_count",
    "prize2_count",
    "prize3_count",
    "prize4_count",
    "prize5_count",
    "total_cost",
    "total_payout",
    "profit",
    "best_prize",
]


def _now_jst_iso() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST")


def _format_yen(amount: int | float) -> str:
    val = int(round(amount))
    if abs(val) >= 100_000_000:
        return f"{val / 100_000_000:.2f}億円 ({val:,}円)"
    if abs(val) >= 10_000:
        return f"{val / 10_000:.1f}万円 ({val:,}円)"
    return f"{val:,}円"


def _best_prize_label(ev: dict[str, float]) -> str:
    for grade in range(1, 6):
        if int(round(ev[f"prize{grade}"])) > 0:
            return f"{grade}等"
    return "ハズレ"


def load_prediction_log(log_path: Path = PREDICTION_LOG_PATH) -> pd.DataFrame:
    """Load `data/prediction_log.csv` if present, otherwise return empty DataFrame."""
    if not log_path.exists():
        return pd.DataFrame(columns=PREDICTION_LOG_COLS)
    df = pd.read_csv(log_path, encoding="utf-8")
    if df.empty:
        return pd.DataFrame(columns=PREDICTION_LOG_COLS)
    for col in PREDICTION_LOG_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df["draw_no"] = df["draw_no"].astype(int)
    return df[PREDICTION_LOG_COLS].sort_values("draw_no").reset_index(drop=True)


def load_latest_prediction(pred_path: Path = LATEST_PREDICTION_PATH) -> Optional[dict[str, Any]]:
    """Load `data/latest_prediction.json` if present."""
    if not pred_path.exists():
        return None
    try:
        with pred_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("前回予測JSONの読み込みに失敗しました (%s): %s", pred_path, exc)
        return None


def fetch_latest_with_retry(
    history_path: Path = DEFAULT_OUTPUT_PATH,
    retries: int = 3,
    retry_delay_sec: float = 5.0,
    skip_scrape: bool = False,
) -> pd.DataFrame:
    """Update `data/loto6_history.csv` with retry logic, or load local CSV if skipped."""
    if skip_scrape:
        logger.info("--skip-scrape 指定のためローカル履歴を読み込みます: %s", history_path)
        return load_history_csv(history_path)

    last_err: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            return update_history_csv(output_path=history_path, fallback_to_dummy=False)
        except Exception as exc:
            last_err = exc
            logger.warning("最新データ取得失敗 (%d/%d): %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(retry_delay_sec)

    if history_path.exists():
        logger.error("オンライン更新に失敗したため既存CSV (%s) で続行します: %s", history_path, last_err)
        return load_history_csv(history_path)

    raise RuntimeError(f"データ取得に失敗し、既存CSVも存在しません: {last_err}") from last_err


def evaluate_previous_prediction(
    history_df: pd.DataFrame,
    pred_path: Path = LATEST_PREDICTION_PATH,
    log_path: Path = PREDICTION_LOG_PATH,
) -> Optional[dict[str, Any]]:
    """Evaluate `latest_prediction.json` against `history_df` if the target draw has occurred.

    Idempotent: if `target_draw_no` is already recorded in `prediction_log.csv`, does not
    duplicate it and returns None.
    """
    prev_pred = load_latest_prediction(pred_path)
    if prev_pred is None:
        logger.info("前回の予測ファイル (%s) が未作成のため結果照合をスキップします。", pred_path)
        return None

    target_draw_no = int(prev_pred["target_draw_no"])
    match_rows = history_df[history_df["draw_no"] == target_draw_no]
    if match_rows.empty:
        logger.info(
            "前回予測の対象回（第%d回）の抽選結果がまだ履歴に含まれていないため照合をスキップします（最新: 第%d回）。",
            target_draw_no,
            int(history_df["draw_no"].max()),
        )
        return None

    existing_log = load_prediction_log(log_path)
    if not existing_log.empty and target_draw_no in set(existing_log["draw_no"].astype(int)):
        logger.info("第%d回の照合結果は既に %s に記録済みのためスキップします。", target_draw_no, log_path)
        return None

    row = match_rows.iloc[0]
    actual_nums_sorted = sorted(int(row[c]) for c in MAIN_NUM_COLS)
    actual_main = set(actual_nums_sorted)
    actual_bonus = int(row["bonus"])
    actual_sb = str(row["set_ball"])
    actual_date = str(row["draw_date"])

    # Determine previous draw's carryover for realistic 1st-prize fallback
    prev_rows = history_df[history_df["draw_no"] == target_draw_no - 1]
    prev_co = int(prev_rows["carryover"].iloc[0]) if not prev_rows.empty else 0
    payouts = _extract_draw_payouts(row, prev_carryover=prev_co)

    saved_tickets_raw = prev_pred.get("tickets", [])
    tickets = [tuple(sorted(int(x) for x in t["numbers"])) for t in saved_tickets_raw]
    pool_numbers = [int(x) for x in prev_pred.get("pool_numbers", [])]
    sb_top3_list = [str(item["set_ball"]) for item in prev_pred.get("set_ball_top3", [])[:3]]
    sb_top1 = sb_top3_list[0] if sb_top3_list else ""

    ev = evaluate_tickets_against_draw(tickets, actual_main, actual_bonus, payouts)
    pool_hits = len(actual_main.intersection(pool_numbers))
    total_cost = int(round(ev["cost"]))
    total_payout = int(round(ev["payout"]))
    profit = total_payout - total_cost
    best_prize = _best_prize_label(ev)

    # Per-ticket detailed breakdown for the Markdown report
    ticket_details = []
    for idx, t in enumerate(tickets, start=1):
        t_set = set(t)
        hit_nums = sorted(t_set.intersection(actual_main))
        m = len(hit_nums)
        has_bonus = actual_bonus in t_set
        grade = 0
        if m == 6:
            grade = 1
        elif m == 5:
            grade = 2 if has_bonus else 3
        elif m == 4:
            grade = 4
        elif m == 3:
            grade = 5
        ticket_details.append(
            {
                "ticket_id": idx,
                "numbers_str": "-".join(f"{x:02d}" for x in t),
                "matches": m,
                "hit_numbers": hit_nums,
                "bonus_hit": has_bonus,
                "prize": f"{grade}等" if grade > 0 else "ハズレ",
                "payout": payouts[grade] if grade > 0 else 0,
            }
        )

    record = {
        "draw_no": target_draw_no,
        "draw_date": actual_date,
        "evaluated_at_jst": _now_jst_iso(),
        "actual_numbers": "-".join(f"{x:02d}" for x in actual_nums_sorted),
        "actual_bonus": actual_bonus,
        "actual_set_ball": actual_sb,
        "pred_set_ball_top1": sb_top1,
        "pred_set_ball_top3": ",".join(sb_top3_list),
        "set_ball_top1_hit": int(actual_sb == sb_top1),
        "set_ball_top3_hit": int(actual_sb in sb_top3_list),
        "pool_size": len(pool_numbers),
        "pool_numbers": ",".join(f"{x:02d}" for x in pool_numbers),
        "pool_hits": int(pool_hits),
        "n_tickets": len(tickets),
        "tickets_str": " | ".join("-".join(f"{x:02d}" for x in t) for t in tickets),
        "best_ticket_matches": int(round(ev["max_matches"])),
        "avg_matches_per_ticket": round(float(ev["avg_matches"]), 4),
        "prize1_count": int(round(ev["prize1"])),
        "prize2_count": int(round(ev["prize2"])),
        "prize3_count": int(round(ev["prize3"])),
        "prize4_count": int(round(ev["prize4"])),
        "prize5_count": int(round(ev["prize5"])),
        "total_cost": total_cost,
        "total_payout": total_payout,
        "profit": profit,
        "best_prize": best_prize,
    }

    updated_log = pd.concat([existing_log, pd.DataFrame([record])], ignore_index=True)
    updated_log = updated_log.drop_duplicates(subset=["draw_no"], keep="last").sort_values("draw_no").reset_index(drop=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    updated_log[PREDICTION_LOG_COLS].to_csv(log_path, index=False, encoding="utf-8")
    logger.info(
        "第%d回の結果照合を追記しました: 最高=%s / 払戻=%s円 / プール包含=%d/%d個 / 1口平均一致=%.3f個",
        target_draw_no,
        best_prize,
        f"{total_payout:,}",
        pool_hits,
        len(pool_numbers),
        ev["avg_matches"],
    )

    return {
        "record": record,
        "ticket_details": ticket_details,
        "pool_hit_numbers": sorted(actual_main.intersection(pool_numbers)),
    }


def generate_next_prediction_payload(
    history_df: pd.DataFrame,
    start_draw_no: int = DEFAULT_START_DRAW_NO,
    min_past_draws: int = DEFAULT_MIN_PAST_DRAWS,
    lgbm_weight: float = 0.50,
    pool_size: int = 12,
    n_tickets: int = 5,
    use_crowd_penalty: bool = False,
) -> dict[str, Any]:
    """Run Phase 2 + Phase 3 strictly on `history_df` and build JSON-serializable payload."""
    pred_cfg = PredictionConfig(
        start_draw_no=start_draw_no,
        min_past_draws=min_past_draws,
        lgbm_weight=lgbm_weight,
        top_k=pool_size,
    )
    opt_cfg = OptimizerConfig(
        pool_size=pool_size,
        n_tickets=n_tickets,
        use_crowd_penalty=use_crowd_penalty,
    )

    pred = Loto6PredictionEngine(pred_cfg).fit_predict_next(history_df)
    opt = optimize_tickets(pred.number_scores, opt_cfg)

    latest_row = history_df.iloc[-1]
    latest_nums = [int(latest_row[c]) for c in MAIN_NUM_COLS]

    sb_top3 = []
    for idx, r in enumerate(pred.set_ball_probs.head(3).itertuples(index=False), start=1):
        sb_top3.append(
            {
                "rank": idx,
                "set_ball": str(r.set_ball),
                "prob": round(float(r.final_prob), 4),
                "last_gap": int(r.last_gap),
                "hazard_prob": round(float(r.hazard_prob), 4),
                "markov_prob": round(float(r.markov_prob), 4),
            }
        )

    top_detail = []
    for r in pred.number_scores.head(pool_size).itertuples(index=False):
        top_detail.append(
            {
                "rank": int(r.rank),
                "number": int(r.number),
                "final_score": round(float(r.final_score), 5),
                "bayes_score": round(float(r.bayes_score), 5),
                "lgbm_score": round(float(r.lgbm_score), 5),
                "score_lift": round(float(r.score_lift), 4),
                "interval": int(r.interval),
                "is_pull": int(r.is_pull),
                "is_slide": int(r.is_slide),
            }
        )

    tickets_payload = []
    for r in opt.tickets_df.itertuples(index=False):
        nums = [int(r.num1), int(r.num2), int(r.num3), int(r.num4), int(r.num5), int(r.num6)]
        tickets_payload.append(
            {
                "ticket_id": int(r.ticket_id),
                "numbers": nums,
                "numbers_str": str(r.numbers_str),
                "sum_value": int(r.sum_value),
                "odd_count": int(r.odd_count),
                "even_count": int(r.even_count),
                "score_sum": round(float(r.score_sum), 5),
                "crowd_penalty": round(float(r.crowd_penalty), 3),
            }
        )

    cov = opt.coverage_metrics
    return {
        "schema_version": 1,
        "generated_at_jst": _now_jst_iso(),
        "target_draw_no": int(pred.target_draw_no),
        "as_of_draw_no": int(pred.as_of_draw_no),
        "as_of_draw_date": str(latest_row["draw_date"]),
        "latest_result": {
            "draw_no": int(latest_row["draw_no"]),
            "draw_date": str(latest_row["draw_date"]),
            "numbers": latest_nums,
            "numbers_str": "-".join(f"{x:02d}" for x in latest_nums),
            "bonus": int(latest_row["bonus"]),
            "set_ball": str(latest_row["set_ball"]),
            "carryover": int(latest_row["carryover"]),
        },
        "config": {
            "start_draw_no": int(start_draw_no),
            "min_past_draws": int(min_past_draws),
            "lgbm_weight": float(lgbm_weight),
            "pool_size": int(pool_size),
            "n_tickets": int(n_tickets),
            "use_crowd_penalty": bool(use_crowd_penalty),
        },
        "set_ball_top3": sb_top3,
        "pool_numbers": [int(x) for x in opt.pool_numbers],
        "top_numbers_detail": top_detail,
        "tickets": tickets_payload,
        "coverage_metrics": {
            "cov_3_hits_ge_5th": round(float(cov["cov_3_hits_ge_5th"]), 4),
            "cov_4_hits_ge_5th": round(float(cov["cov_4_hits_ge_5th"]), 4),
            "cov_4_hits_ge_4th": round(float(cov["cov_4_hits_ge_4th"]), 4),
            "cov_5_hits_ge_4th": round(float(cov["cov_5_hits_ge_4th"]), 4),
            "avg_pairwise_overlap": round(float(cov["avg_pairwise_overlap"]), 2),
        },
    }


def render_markdown_report(
    payload: dict[str, Any],
    eval_info: Optional[dict[str, Any]],
    log_df: pd.DataFrame,
) -> str:
    """Format a clean, readable Markdown report suitable for GitHub, Discord, and Email."""
    latest = payload["latest_result"]
    target_draw = payload["target_draw_no"]
    cov = payload["coverage_metrics"]
    cfg = payload["config"]

    lines: list[str] = [
        f"# 🎯 ロト6 自動分析・予測レポート（第{target_draw}回向け）",
        f"- 作成日時: `{payload['generated_at_jst']}`",
        f"- 最新抽選結果: **第{latest['draw_no']}回 ({latest['draw_date']})**",
        f"  - 当選番号: `{latest['numbers_str']}` （ボーナス: `{latest['bonus']:02d}`） / セット球: **{latest['set_ball']}**",
        f"  - 次回キャリーオーバー: **{_format_yen(latest['carryover'])}**",
        "",
    ]

    # Evaluation section if previous prediction was just graded
    if eval_info is not None:
        rec = eval_info["record"]
        lines.extend(
            [
                f"## ✅ 前回予測の自動照合結果（第{rec['draw_no']}回）",
                f"- 結果サマリー: 最高 **{rec['best_prize']}** ｜ 投資 `{rec['total_cost']:,}円` → 払戻 **`{rec['total_payout']:,}円`** （収支 `{rec['profit']:+,}円`）",
                f"- セット球予測: 実際 **{rec['actual_set_ball']}** ／ 予測Top3 `{rec['pred_set_ball_top3']}` （Top1: {'🎯的中' if rec['set_ball_top1_hit'] else '外れ'} / Top3: {'🎯的中' if rec['set_ball_top3_hit'] else '外れ'}）",
                f"- プール包含: **{rec['pool_hits']} / {rec['pool_size']}個** （的中数字: `{', '.join(f'{x:02d}' for x in eval_info['pool_hit_numbers']) or 'なし'}`）",
                f"- 1口平均一致: **{rec['avg_matches_per_ticket']:.3f}個** （理論値 {THEORETICAL_AVG_MATCHES:.3f}個）",
                "",
                "| 口 | 予測買い目 | 一致数 | 判定 | 払戻金 |",
                "|---:|---|---:|---|---:|",
            ]
        )
        for td in eval_info["ticket_details"]:
            hit_str = ", ".join(f"{x:02d}" for x in td["hit_numbers"])
            match_label = f"{td['matches']}個" + (f" ({hit_str})" if hit_str else "")
            lines.append(
                f"| #{td['ticket_id']} | `{td['numbers_str']}` | {match_label} | **{td['prize']}** | {td['payout']:,}円 |"
            )
        lines.append("")

    # Cumulative live performance summary if prediction_log.csv has rows
    if not log_df.empty:
        n_runs = len(log_df)
        tot_cost = int(log_df["total_cost"].sum())
        tot_pay = int(log_df["total_payout"].sum())
        tot_prof = tot_pay - tot_cost
        roi = (tot_pay / tot_cost * 100.0) if tot_cost > 0 else 0.0
        avg_m = float(log_df["avg_matches_per_ticket"].mean())
        p4_tot = int(log_df["prize4_count"].sum())
        p5_tot = int(log_df["prize5_count"].sum())
        sb1_rate = float(log_df["set_ball_top1_hit"].mean() * 100.0)
        sb3_rate = float(log_df["set_ball_top3_hit"].mean() * 100.0)
        lines.extend(
            [
                f"## 📊 通算運用成績（全{n_runs}回照合済み）",
                f"- 累積投資: `{tot_cost:,}円` ／ 累積払戻: **`{tot_pay:,}円`** （純収支 `{tot_prof:+,}円` / ROI **{roi:.1f}%**）",
                f"- 1口平均一致: **{avg_m:.3f}個** （理論値比 `{avg_m - THEORETICAL_AVG_MATCHES:+.3f}`） ／ 4等 `{p4_tot}口`・5等 `{p5_tot}口`",
                f"- セット球予測通算精度: Top1 **{sb1_rate:.1f}%** ／ Top3 **{sb3_rate:.1f}%**",
                "",
            ]
        )

    # Next draw prediction section
    lines.extend(
        [
            f"## 🔮 次回予測：第{target_draw}回",
            "",
            "### 🎱 Stage 1：予測セット球 上位3候補",
        ]
    )
    for sb in payload["set_ball_top3"]:
        lines.append(
            f"- **{sb['rank']}位: セット球 {sb['set_ball']}** — 予測確率 **{sb['prob'] * 100:.1f}%**（未出現間隔: {sb['last_gap']}回）"
        )

    pool_str = " - ".join(f"{n:02d}" for n in payload["pool_numbers"])
    lines.extend(
        [
            "",
            f"### 🌟 Stage 2：推奨上位 {cfg['pool_size']} 数字（スコア順）",
            f"`{pool_str}`",
            "",
            f"### 🎫 Phase 3：ホイーリング最適買い目（{cfg['n_tickets']}口 / 計 {cfg['n_tickets'] * 200:,}円）",
            f"- プール内4個的中 → 5等以上カバー率: **{cov['cov_4_hits_ge_5th'] * 100:.1f}%** ｜ 3個的中 → 5等カバー率: **{cov['cov_3_hits_ge_5th'] * 100:.1f}%**",
            f"- 買い目間平均重複: `{cov['avg_pairwise_overlap']:.2f}個` ｜ 大衆心理ペナルティ: `{'ON' if cfg['use_crowd_penalty'] else 'OFF（的中率優先）'}`",
            "",
            "| 口 | 推奨買い目 | 合計値 | 奇:偶 | スコア和 |",
            "|---:|---|---:|:---:|---:|",
        ]
    )
    for t in payload["tickets"]:
        lines.append(
            f"| **#{t['ticket_id']}** | `{t['numbers_str']}` | {t['sum_value']} | {t['odd_count']}:{t['even_count']} | {t['score_sum']:.4f} |"
        )

    lines.append("")
    return "\n".join(lines)


def _split_discord_chunks(text: str, limit: int = 1900) -> list[str]:
    """Split text cleanly at newline boundaries to stay safely within Discord's 2000-char limit."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for line in text.splitlines(keepends=True):
        if cur_len + len(line) > limit and cur:
            chunks.append("".join(cur).rstrip())
            cur = [line]
            cur_len = len(line)
        else:
            cur.append(line)
            cur_len += len(line)
    if cur:
        chunks.append("".join(cur).rstrip())
    return chunks


def send_discord_notification(report_text: str, webhook_url: Optional[str] = None) -> bool:
    """Send report to Discord via Webhook if `DISCORD_WEBHOOK_URL` is configured."""
    url = (webhook_url if webhook_url is not None else os.environ.get("DISCORD_WEBHOOK_URL", "")).strip()
    if not url:
        logger.info("DISCORD_WEBHOOK_URL が未設定のため Discord 通知をスキップします。")
        return False

    for chunk in _split_discord_chunks(report_text, limit=1900):
        resp = requests.post(url, json={"content": chunk}, timeout=15)
        resp.raise_for_status()
    logger.info("Discord Webhook への通知を送信しました。")
    return True


def send_gmail_notification(
    subject: str,
    report_text: str,
    gmail_user: Optional[str] = None,
    gmail_app_password: Optional[str] = None,
    gmail_to: Optional[str] = None,
) -> bool:
    """Send report via Gmail SMTP SSL if `GMAIL_USER` & `GMAIL_APP_PASSWORD` are configured."""
    user = (gmail_user if gmail_user is not None else os.environ.get("GMAIL_USER", "")).strip()
    password = (
        gmail_app_password if gmail_app_password is not None else os.environ.get("GMAIL_APP_PASSWORD", "")
    ).strip()
    recipient = (gmail_to if gmail_to is not None else os.environ.get("GMAIL_TO", "")).strip() or user

    if not user or not password:
        logger.info("GMAIL_USER / GMAIL_APP_PASSWORD が未設定のため Gmail 通知をスキップします。")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = recipient
    msg.set_content(report_text)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
        server.login(user, password)
        server.send_message(msg)
    logger.info("Gmail 通知を送信しました (宛先: %s)。", recipient)
    return True


def run_auto_pipeline(
    history_path: Path = DEFAULT_OUTPUT_PATH,
    pred_path: Path = LATEST_PREDICTION_PATH,
    report_path: Path = LATEST_REPORT_PATH,
    log_path: Path = PREDICTION_LOG_PATH,
    skip_scrape: bool = False,
    notify: bool = True,
    strict_notify: bool = False,
    start_draw_no: int = DEFAULT_START_DRAW_NO,
    min_past_draws: int = DEFAULT_MIN_PAST_DRAWS,
    lgbm_weight: float = 0.50,
    pool_size: int = 12,
    n_tickets: int = 5,
    use_crowd_penalty: bool = False,
) -> dict[str, Any]:
    """Execute the full automated pipeline and return summary metadata."""
    # 1. Fetch latest data
    history_df = fetch_latest_with_retry(history_path=history_path, skip_scrape=skip_scrape)

    # 2. Evaluate previous prediction if its target draw is now available
    eval_info = evaluate_previous_prediction(history_df, pred_path=pred_path, log_path=log_path)
    log_df = load_prediction_log(log_path)

    # 3. Generate next-draw prediction
    payload = generate_next_prediction_payload(
        history_df=history_df,
        start_draw_no=start_draw_no,
        min_past_draws=min_past_draws,
        lgbm_weight=lgbm_weight,
        pool_size=pool_size,
        n_tickets=n_tickets,
        use_crowd_penalty=use_crowd_penalty,
    )

    # 4. Persist JSON & Markdown report
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("次回予測JSONを保存しました: %s (対象: 第%d回)", pred_path, payload["target_draw_no"])

    report_md = render_markdown_report(payload=payload, eval_info=eval_info, log_df=log_df)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")
    logger.info("予測レポートMarkdownを保存しました: %s", report_path)

    # Write to GitHub Actions Step Summary automatically if running in CI
    gh_summary = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
    if gh_summary:
        try:
            with open(gh_summary, "a", encoding="utf-8") as sf:
                sf.write(report_md + "\n")
        except Exception as exc:
            logger.warning("GITHUB_STEP_SUMMARY 書き込み失敗: %s", exc)

    # 5. Optional notifications (Discord / Gmail)
    notified = {"discord": False, "gmail": False}
    if notify:
        subject = f"[ロト6自動予測] 第{payload['target_draw_no']}回 推奨買い目＆結果レポート"
        try:
            notified["discord"] = send_discord_notification(report_md)
        except Exception as exc:
            logger.error("Discord 通知の送信に失敗しました: %s", exc)
            if strict_notify:
                raise

        try:
            notified["gmail"] = send_gmail_notification(subject=subject, report_text=report_md)
        except Exception as exc:
            logger.error("Gmail 通知の送信に失敗しました: %s", exc)
            if strict_notify:
                raise

    return {
        "latest_draw_no": int(history_df["draw_no"].max()),
        "target_draw_no": int(payload["target_draw_no"]),
        "evaluated": eval_info is not None,
        "eval_info": eval_info,
        "payload": payload,
        "report_md": report_md,
        "notified": notified,
        "log_rows": len(log_df),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="ロト6 全自動更新・結果照合・次回予測・通知バッチ")
    parser.add_argument("--skip-scrape", action="store_true", help="オンライン取得をスキップし既存CSVを使用")
    parser.add_argument("--no-notify", action="store_true", help="Discord/Gmail通知を無効化")
    parser.add_argument("--strict-notify", action="store_true", help="通知失敗時に非0終了する")
    parser.add_argument("--start-draw-no", type=int, default=DEFAULT_START_DRAW_NO, help="学習開始回 (デフォルト: 138)")
    parser.add_argument("--min-past-draws", type=int, default=DEFAULT_MIN_PAST_DRAWS, help="ウォームアップ回数 (デフォルト: 100)")
    parser.add_argument("--lgbm-weight", type=float, default=0.50, help="LightGBMブレンド比率 (デフォルト: 0.50)")
    parser.add_argument("--pool-size", type=int, default=12, help="上位プール数字数 (デフォルト: 12)")
    parser.add_argument("--n-tickets", type=int, default=5, help="生成口数 (デフォルト: 5)")
    parser.add_argument("--crowd-penalty", action="store_true", help="大衆心理ペナルティをONにする")
    args = parser.parse_args()

    res = run_auto_pipeline(
        skip_scrape=args.skip_scrape,
        notify=not args.no_notify,
        strict_notify=args.strict_notify,
        start_draw_no=args.start_draw_no,
        min_past_draws=args.min_past_draws,
        lgbm_weight=args.lgbm_weight,
        pool_size=args.pool_size,
        n_tickets=args.n_tickets,
        use_crowd_penalty=args.crowd_penalty,
    )
    print(res["report_md"])


if __name__ == "__main__":
    main()
