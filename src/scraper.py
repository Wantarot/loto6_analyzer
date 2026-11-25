"""Loto 6 historical draw data acquisition, manual CSV import, and dummy generator.

Primary online data sources (2-source merge with cross-validation):
  1. KYO's LOTO6 CSV (https://loto6.thekyo.jp/data/loto6.csv)
     - Draw number, draw date, 6 main numbers, bonus number, prize counts/amounts, carryover.
  2. Sougaku / Loto6 Seikatsu (http://sougaku.com/loto6/data/list1/)
     - Draw number, 6 main numbers, bonus number, set ball (A-J) for all historical draws.

Fallback modes:
  - Manual CSV import (`import_manual_csv`) supporting Japanese/English headers & UTF-8/CP932.
  - Synthetic dummy dataset generator (`generate_dummy_data`) for offline testing.
"""

from __future__ import annotations

import argparse
import io
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

KYO_CSV_URL = "https://loto6.thekyo.jp/data/loto6.csv"
SOUGAKU_URL = "http://sougaku.com/loto6/data/list1/"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}

VALID_SET_BALLS = tuple("ABCDEFGHIJ")
MAIN_NUM_COLS = ["num1", "num2", "num3", "num4", "num5", "num6"]
REQUIRED_COLS = [
    "draw_no",
    "draw_date",
    *MAIN_NUM_COLS,
    "bonus",
    "set_ball",
    "carryover",
]
PRIZE_COLS = [
    "prize1_count",
    "prize2_count",
    "prize3_count",
    "prize4_count",
    "prize5_count",
    "prize1_amount",
    "prize2_amount",
    "prize3_amount",
    "prize4_amount",
    "prize5_amount",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "loto6_history.csv"

# Column alias map for manual CSV imports (Japanese & English variations)
MANUAL_COLUMN_ALIASES = {
    "開催回": "draw_no",
    "回号": "draw_no",
    "抽選回": "draw_no",
    "抽せん回": "draw_no",
    "回": "draw_no",
    "日付": "draw_date",
    "抽選日": "draw_date",
    "抽せん日": "draw_date",
    "date": "draw_date",
    "第1数字": "num1",
    "第2数字": "num2",
    "第3数字": "num3",
    "第4数字": "num4",
    "第5数字": "num5",
    "第6数字": "num6",
    "本数字1": "num1",
    "本数字2": "num2",
    "本数字3": "num3",
    "本数字4": "num4",
    "本数字5": "num5",
    "本数字6": "num6",
    "BONUS数字": "bonus",
    "ボーナス数字": "bonus",
    "ボーナス": "bonus",
    "B数字": "bonus",
    "セット球": "set_ball",
    "セット": "set_ball",
    "ｾｯﾄ": "set_ball",
    "キャリーオーバー": "carryover",
    "キャリーオーバー額": "carryover",
    "CO": "carryover",
    "1等口数": "prize1_count",
    "2等口数": "prize2_count",
    "3等口数": "prize3_count",
    "4等口数": "prize4_count",
    "5等口数": "prize5_count",
    "1等賞金": "prize1_amount",
    "2等賞金": "prize2_amount",
    "3等賞金": "prize3_amount",
    "4等賞金": "prize4_amount",
    "5等賞金": "prize5_amount",
}


def _clean_int_series(series: pd.Series) -> pd.Series:
    """Convert strings possibly containing commas/yen symbols/spaces to integers."""
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("円", "", regex=False)
        .str.replace("第", "", regex=False)
        .str.replace("回", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="raise").astype(int)


def validate_history_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a Loto 6 history DataFrame to canonical schema.

    Raises:
        ValueError: If required columns are missing or values violate Loto 6 rules.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"必須カラムが不足しています: {missing}")

    out = df.copy()
    out["draw_no"] = _clean_int_series(out["draw_no"])
    out["draw_date"] = pd.to_datetime(out["draw_date"], errors="raise").dt.strftime("%Y-%m-%d")

    for col in [*MAIN_NUM_COLS, "bonus", "carryover"]:
        out[col] = _clean_int_series(out[col])

    for col in PRIZE_COLS:
        if col in out.columns:
            out[col] = _clean_int_series(out[col])

    out["set_ball"] = out["set_ball"].astype(str).str.strip().str.upper()

    # Sort main numbers ascending within each row to guarantee consistent order
    main_vals = np.sort(out[MAIN_NUM_COLS].to_numpy(dtype=int), axis=1)
    out[MAIN_NUM_COLS] = main_vals

    # Integrity checks
    if (out["draw_no"] <= 0).any():
        raise ValueError("draw_no に 0 以下の値が含まれています。")
    if out["draw_no"].duplicated().any():
        dup_count = int(out["draw_no"].duplicated().sum())
        logger.warning("draw_no に重複が %d 件あります。最新行を優先して重複排除します。", dup_count)
        out = out.drop_duplicates(subset=["draw_no"], keep="last")

    if (main_vals < 1).any() or (main_vals > 43).any():
        raise ValueError("本数字に 1〜43 範囲外の値が含まれています。")
    if (np.diff(main_vals, axis=1) <= 0).any():
        raise ValueError("同一開催回内に重複した本数字が含まれています。")

    if (out["bonus"] < 1).any() or (out["bonus"] > 43).any():
        raise ValueError("ボーナス数字に 1〜43 範囲外の値が含まれています。")
    bonus_in_main = (main_vals == out["bonus"].to_numpy()[:, None]).any(axis=1)
    if bonus_in_main.any():
        bad_draws = out.loc[bonus_in_main, "draw_no"].tolist()[:5]
        raise ValueError(f"ボーナス数字が本数字と重複している回があります: {bad_draws}")

    invalid_sb = ~out["set_ball"].isin(VALID_SET_BALLS)
    if invalid_sb.any():
        bad_vals = out.loc[invalid_sb, "set_ball"].unique().tolist()
        raise ValueError(f"セット球に A〜J 以外の値が含まれています: {bad_vals}")

    if (out["carryover"] < 0).any():
        raise ValueError("carryover に負の値が含まれています。")

    out = out.sort_values("draw_no").reset_index(drop=True)

    ordered_cols = [*REQUIRED_COLS, *[c for c in PRIZE_COLS if c in out.columns]]
    return out[ordered_cols]


def fetch_kyo_csv(url: str = KYO_CSV_URL, timeout: int = 20) -> pd.DataFrame:
    """Fetch full Loto 6 draw results & carryover from KYO's LOTO6 CSV."""
    logger.info("KYO's LOTO6 CSV を取得中: %s", url)
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    resp.raise_for_status()

    text: Optional[str] = None
    for enc in ("cp932", "shift_jis", "utf-8-sig", "utf-8"):
        try:
            text = resp.content.decode(enc)
            if "開催回" in text.splitlines()[0]:
                break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("KYO's LOTO6 CSV の文字コード判別に失敗しました。")

    raw = pd.read_csv(io.StringIO(text))
    raw.columns = [str(c).strip() for c in raw.columns]
    df = raw.rename(columns=MANUAL_COLUMN_ALIASES)

    expected = ["draw_no", "draw_date", *MAIN_NUM_COLS, "bonus", "carryover"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"KYO's LOTO6 CSV の想定カラムが見つかりません: {missing}")

    df["draw_no"] = _clean_int_series(df["draw_no"])
    df["draw_date"] = pd.to_datetime(df["draw_date"]).dt.strftime("%Y-%m-%d")
    for col in [*MAIN_NUM_COLS, "bonus", "carryover", *[c for c in PRIZE_COLS if c in df.columns]]:
        df[col] = _clean_int_series(df[col])

    keep_cols = [c for c in ["draw_no", "draw_date", *MAIN_NUM_COLS, "bonus", "carryover", *PRIZE_COLS] if c in df.columns]
    logger.info("KYO's LOTO6 CSV 取得完了: %d 件 (第%d回〜第%d回)", len(df), df["draw_no"].min(), df["draw_no"].max())
    return df[keep_cols].sort_values("draw_no").reset_index(drop=True)


def fetch_sougaku_set_balls(url: str = SOUGAKU_URL, timeout: int = 20) -> pd.DataFrame:
    """Fetch full Loto 6 set ball history (A-J) from sougaku.com."""
    logger.info("ロト6生活（セット球一覧）を取得中: %s", url)
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    resp.raise_for_status()
    html = resp.content.decode("utf-8", errors="replace")

    soup = BeautifulSoup(html, "html.parser")
    draw_pattern = re.compile(r"^第(\d+)回$")

    records = []
    for tr in soup.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 9:
            continue
        m = draw_pattern.match(cells[0])
        if not m:
            continue

        draw_no = int(m.group(1))
        set_ball = cells[8].strip().upper()
        if set_ball not in VALID_SET_BALLS:
            logger.warning("第%d回のセット球が不正な値です (%s) — スキップします。", draw_no, set_ball)
            continue

        try:
            nums = sorted(int(cells[i]) for i in range(1, 7))
            bonus = int(cells[7])
        except ValueError:
            logger.warning("第%d回の数字パースに失敗しました — スキップします。", draw_no)
            continue

        records.append(
            {
                "draw_no": draw_no,
                "sb_num1": nums[0],
                "sb_num2": nums[1],
                "sb_num3": nums[2],
                "sb_num4": nums[3],
                "sb_num5": nums[4],
                "sb_num6": nums[5],
                "sb_bonus": bonus,
                "set_ball": set_ball,
            }
        )

    if not records:
        raise ValueError("ロト6生活のHTMLから抽選データ行を抽出できませんでした（HTML構造変更の可能性）。")

    df = pd.DataFrame(records).drop_duplicates(subset=["draw_no"], keep="last")
    df = df.sort_values("draw_no").reset_index(drop=True)
    logger.info("ロト6生活 取得完了: %d 件 (第%d回〜第%d回)", len(df), df["draw_no"].min(), df["draw_no"].max())
    return df


def fetch_loto6_data(
    kyo_url: str = KYO_CSV_URL,
    sougaku_url: str = SOUGAKU_URL,
    timeout: int = 20,
) -> pd.DataFrame:
    """Fetch and cross-validate complete Loto 6 history from both sources."""
    kyo_df = fetch_kyo_csv(url=kyo_url, timeout=timeout)
    sb_df = fetch_sougaku_set_balls(url=sougaku_url, timeout=timeout)

    merged = pd.merge(kyo_df, sb_df, on="draw_no", how="inner")

    kyo_only = set(kyo_df["draw_no"]) - set(sb_df["draw_no"])
    sb_only = set(sb_df["draw_no"]) - set(kyo_df["draw_no"])
    if kyo_only:
        logger.warning("KYO's LOTO6 のみに存在する開催回（セット球未反映のため除外）: %s", sorted(kyo_only))
    if sb_only:
        logger.warning("ロト6生活 のみに存在する開催回（賞金/CO未反映のため除外）: %s", sorted(sb_only))

    # Cross-validate winning numbers between the two independent sources
    kyo_nums = np.sort(merged[MAIN_NUM_COLS].to_numpy(dtype=int), axis=1)
    sb_nums = np.sort(
        merged[["sb_num1", "sb_num2", "sb_num3", "sb_num4", "sb_num5", "sb_num6"]].to_numpy(dtype=int),
        axis=1,
    )
    mismatch_main = (kyo_nums != sb_nums).any(axis=1)
    mismatch_bonus = merged["bonus"].to_numpy(dtype=int) != merged["sb_bonus"].to_numpy(dtype=int)
    if mismatch_main.any() or mismatch_bonus.any():
        bad_draws = merged.loc[mismatch_main | mismatch_bonus, "draw_no"].tolist()
        logger.warning(
            "2ソース間で当選数字の不一致が %d 件検出されました（KYO's LOTO6 側を採用）: %s",
            len(bad_draws),
            bad_draws[:10],
        )

    merged = merged.drop(columns=["sb_num1", "sb_num2", "sb_num3", "sb_num4", "sb_num5", "sb_num6", "sb_bonus"])
    return validate_history_df(merged)


def import_manual_csv(
    csv_path: str | Path,
    output_path: Optional[str | Path] = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Import a manually prepared CSV file and normalize it to canonical schema.

    Supports UTF-8 (with/without BOM) and CP932/Shift-JIS, as well as Japanese or
    English column names.
    """
    src_path = Path(csv_path)
    if not src_path.exists():
        raise FileNotFoundError(f"手動CSVファイルが見つかりません: {src_path}")

    raw: Optional[pd.DataFrame] = None
    last_err: Optional[Exception] = None
    for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            raw = pd.read_csv(src_path, encoding=enc)
            break
        except UnicodeDecodeError as exc:
            last_err = exc
    if raw is None:
        raise ValueError(f"CSVの読み込みに失敗しました（文字コード非対応）: {last_err}")

    raw.columns = [str(c).strip() for c in raw.columns]
    df = raw.rename(columns=MANUAL_COLUMN_ALIASES)

    validated = validate_history_df(df)
    if output_path is not None:
        save_history_csv(validated, output_path)
    return validated


def generate_dummy_data(
    n_draws: int = 500,
    seed: int = 42,
    inject_bias: bool = True,
    start_date: str = "2021-01-04",
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Generate realistic synthetic Loto 6 history data for offline testing.

    Args:
        n_draws: Number of draws to simulate.
        seed: Random seed for reproducibility.
        inject_bias: If True, adds subtle physical set-ball weights so downstream
            bias detection and feature pipelines can be tested on known ground truth.
        start_date: First draw date (ISO format, Monday/Thursday cadence).
        output_path: Optional path to save the generated CSV.
    """
    if n_draws <= 0:
        raise ValueError("n_draws は 1 以上を指定してください。")

    rng = np.random.default_rng(seed)
    numbers = np.arange(1, 44)

    # Subtle set-ball weight vectors (normalized probabilities over 1..43)
    sb_probs: dict[str, np.ndarray] = {}
    for sb in VALID_SET_BALLS:
        weights = np.ones(43, dtype=float)
        if inject_bias:
            if sb == "A":
                weights[[2, 16, 28]] = 1.45  # numbers 3, 17, 29
            elif sb == "F":
                weights[[7, 21, 37]] = 1.45  # numbers 8, 22, 38
            elif sb == "J":
                weights[[11, 24, 41]] = 1.40  # numbers 12, 25, 42
        sb_probs[sb] = weights / weights.sum()

    # Twice-weekly cadence: Mon -> Thu (+3 days) -> Mon (+4 days)
    cur_date = pd.Timestamp(start_date)
    dates = []
    for idx in range(n_draws):
        dates.append(cur_date.strftime("%Y-%m-%d"))
        cur_date += pd.Timedelta(days=3 if idx % 2 == 0 else 4)

    records = []
    carryover = 0
    for idx in range(n_draws):
        draw_no = idx + 1
        sb = VALID_SET_BALLS[int(rng.integers(0, len(VALID_SET_BALLS)))]
        picked = rng.choice(numbers, size=7, replace=False, p=sb_probs[sb])
        main = np.sort(picked[:6])
        bonus = int(picked[6])

        # Simulate realistic 1st-prize winners & carryover
        p1_count = int(rng.choice([0, 1, 2, 3], p=[0.35, 0.35, 0.20, 0.10]))
        if p1_count == 0:
            p1_amount = 0
            carryover = min(600_000_000, carryover + int(rng.integers(80_000_000, 260_000_000)))
        else:
            p1_amount = min(600_000_000, (200_000_000 + carryover) // p1_count)
            carryover = 0

        records.append(
            {
                "draw_no": draw_no,
                "draw_date": dates[idx],
                "num1": int(main[0]),
                "num2": int(main[1]),
                "num3": int(main[2]),
                "num4": int(main[3]),
                "num5": int(main[4]),
                "num6": int(main[5]),
                "bonus": bonus,
                "set_ball": sb,
                "carryover": carryover,
                "prize1_count": p1_count,
                "prize2_count": int(rng.integers(2, 15)),
                "prize3_count": int(rng.integers(120, 320)),
                "prize4_count": int(rng.integers(7000, 14000)),
                "prize5_count": int(rng.integers(120000, 210000)),
                "prize1_amount": p1_amount,
                "prize2_amount": int(rng.integers(8_000_000, 25_000_000)),
                "prize3_amount": int(rng.integers(250_000, 600_000)),
                "prize4_amount": int(rng.integers(5_500, 12_000)),
                "prize5_amount": 1000,
            }
        )

    df = validate_history_df(pd.DataFrame(records))
    if output_path is not None:
        save_history_csv(df, output_path)
    return df


def save_history_csv(df: pd.DataFrame, output_path: str | Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Save validated Loto 6 history DataFrame to CSV (UTF-8)."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("過去データを保存しました: %s (%d 件)", out_path, len(df))
    return out_path


def load_history_csv(csv_path: str | Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    """Load and validate Loto 6 history CSV."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"履歴CSVが見つかりません: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    return validate_history_df(df)


def update_history_csv(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    fallback_to_dummy: bool = False,
    dummy_draws: int = 500,
    timeout: int = 20,
) -> pd.DataFrame:
    """Fetch latest Loto 6 data and save to CSV, with optional dummy fallback."""
    try:
        df = fetch_loto6_data(timeout=timeout)
        save_history_csv(df, output_path)
        return df
    except Exception as exc:
        logger.error("オンライン取得に失敗しました: %s", exc)
        if fallback_to_dummy:
            logger.warning("フォールバックとしてダミーデータ (%d 回分) を生成します。", dummy_draws)
            return generate_dummy_data(n_draws=dummy_draws, output_path=output_path)
        raise RuntimeError(
            "オンラインデータ取得に失敗しました。フォールバックとして "
            "`--mode manual --manual-csv <path>` または `--mode dummy` を利用してください。"
        ) from exc


def main() -> None:
    """CLI entry point for Loto 6 data acquisition."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="ロト6過去データ取得・インポート・ダミー生成ツール")
    parser.add_argument(
        "--mode",
        choices=["scrape", "manual", "dummy"],
        default="scrape",
        help="データ取得モード (scrape: 自動取得, manual: 手動CSVインポート, dummy: ダミー生成)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"保存先CSVパス (デフォルト: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--manual-csv",
        type=Path,
        default=None,
        help="手動インポート元のCSVパス (--mode manual 時に必須)",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=500,
        help="ダミーデータの生成回数 (--mode dummy またはフォールバック時)",
    )
    parser.add_argument(
        "--fallback-dummy",
        action="store_true",
        help="自動取得失敗時に自動でダミーデータを生成する",
    )
    args = parser.parse_args()

    if args.mode == "scrape":
        df = update_history_csv(
            output_path=args.output,
            fallback_to_dummy=args.fallback_dummy,
            dummy_draws=args.n_draws,
        )
    elif args.mode == "manual":
        if args.manual_csv is None:
            parser.error("--mode manual を使う場合は --manual-csv を指定してください。")
        df = import_manual_csv(csv_path=args.manual_csv, output_path=args.output)
    else:
        df = generate_dummy_data(n_draws=args.n_draws, output_path=args.output)

    print(f"完了: {len(df)} 回分のデータを {args.output} に保存しました。")
    print(f"期間: 第{df['draw_no'].min()}回 ({df['draw_date'].iloc[0]}) 〜 第{df['draw_no'].max()}回 ({df['draw_date'].iloc[-1]})")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
