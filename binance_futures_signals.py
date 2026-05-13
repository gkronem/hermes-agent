"""
Multi-timeframe futures structure scanner for USDT perpetuals.

Scans Binance / Bybit / OKX perpetual pairs and builds a directional read
using EMA 9, SMA 20, and a lightweight Smart Money Concepts approximation:
- break of structure (BOS)
- bullish / bearish swing structure
- premium / discount
- simple liquidity sweep detection

Usage:
    python binance_futures_signals.py
    python binance_futures_signals.py --exchange bybit
    python binance_futures_signals.py --top 50
    python binance_futures_signals.py --symbols BTCUSDT,SOLUSDT
    python binance_futures_signals.py --demo
"""

import sys
import time
import random
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

EXCHANGE = "auto"
TOP_N = 0
KLINE_LIMIT = 250
TIMEFRAMES = ("4h", "1h", "15m", "1m")
EMA_LENGTH = 9
SMA_LENGTH = 20
SWING_LOOKBACK = 2

ENDPOINTS = {
    "binance": "https://fapi.binance.com/fapi/v1",
    "bybit": "https://api.bybit.com/v5",
    "okx": "https://www.okx.com/api/v5",
}


@dataclass
class MarketMeta:
    symbol: str
    price: float
    change_24h: float
    volume_usdt: float
    venue_id: str


@dataclass
class TimeframeSignal:
    timeframe: str
    price: float
    ema9: float
    sma20: float
    ma_stack: str
    price_vs_ma: str
    ema_slope: str
    structure: str
    swing_high: float
    swing_low: float
    pd_location: str
    liquidity: str
    score: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class CoinSignal:
    symbol: str
    price: float
    change_24h: float
    volume_usdt: float
    funding_rate: float
    direction: str
    score: float
    bias: str
    timeframe_signals: Dict[str, TimeframeSignal]
    reasons: List[str] = field(default_factory=list)


def _ema_series(values: List[float], period: int) -> List[float]:
    k = 2 / (period + 1)
    ema = [values[0]]
    for value in values[1:]:
        ema.append(value * k + ema[-1] * (1 - k))
    return ema


def _sma(values: List[float], period: int) -> float:
    return sum(values[-period:]) / period


def _to_bybit_interval(iv: str) -> str:
    mapping = {
        "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
        "1d": "D", "1w": "W", "1M": "M",
    }
    return mapping.get(iv.lower(), "60")


def _to_okx_interval(iv: str) -> str:
    mapping = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H",
        "1d": "1D", "1w": "1W",
    }
    return mapping.get(iv.lower(), "1H")


def _swing_points(highs: List[float], lows: List[float], lookback: int = SWING_LOOKBACK) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    swing_highs: List[Tuple[int, float]] = []
    swing_lows: List[Tuple[int, float]] = []
    for i in range(lookback, len(highs) - lookback):
        high = highs[i]
        low = lows[i]
        if all(high > highs[j] for j in range(i - lookback, i)) and all(high >= highs[j] for j in range(i + 1, i + 1 + lookback)):
            swing_highs.append((i, high))
        if all(low < lows[j] for j in range(i - lookback, i)) and all(low <= lows[j] for j in range(i + 1, i + 1 + lookback)):
            swing_lows.append((i, low))
    return swing_highs, swing_lows


def _structure(closes: List[float], highs: List[float], lows: List[float]) -> Tuple[str, float, float, str]:
    swing_highs, swing_lows = _swing_points(highs, lows)
    recent_highs = swing_highs[-6:]
    recent_lows = swing_lows[-6:]

    last_swing_high = recent_highs[-1][1] if recent_highs else max(highs[-20:])
    last_swing_low = recent_lows[-1][1] if recent_lows else min(lows[-20:])
    close = closes[-1]

    if close > last_swing_high:
        state = "BOS up"
    elif close < last_swing_low:
        state = "BOS down"
    else:
        hh = len(recent_highs) >= 2 and recent_highs[-1][1] > recent_highs[-2][1]
        hl = len(recent_lows) >= 2 and recent_lows[-1][1] > recent_lows[-2][1]
        lh = len(recent_highs) >= 2 and recent_highs[-1][1] < recent_highs[-2][1]
        ll = len(recent_lows) >= 2 and recent_lows[-1][1] < recent_lows[-2][1]
        if hh and hl:
            state = "bullish structure"
        elif lh and ll:
            state = "bearish structure"
        else:
            state = "range / transition"

    liquidity = "none"
    if len(recent_highs) >= 2 and highs[-1] > recent_highs[-1][1] and close < recent_highs[-1][1]:
        liquidity = "buy-side sweep"
    elif len(recent_lows) >= 2 and lows[-1] < recent_lows[-1][1] and close > recent_lows[-1][1]:
        liquidity = "sell-side sweep"

    return state, last_swing_high, last_swing_low, liquidity


def _analyze_timeframe(timeframe: str, candles: List[Dict[str, float]]) -> TimeframeSignal:
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    ema_series = _ema_series(closes, EMA_LENGTH)
    ema9 = ema_series[-1]
    prev_ema9 = ema_series[-2]
    sma20 = _sma(closes, SMA_LENGTH)
    price = closes[-1]

    ma_stack = "bullish" if ema9 > sma20 else "bearish" if ema9 < sma20 else "flat"
    if price > ema9 and price > sma20:
        price_vs_ma = "above both"
    elif price < ema9 and price < sma20:
        price_vs_ma = "below both"
    else:
        price_vs_ma = "between MAs"

    ema_slope = "up" if ema9 > prev_ema9 else "down"
    structure, swing_high, swing_low, liquidity = _structure(closes, highs, lows)
    midpoint = (swing_high + swing_low) / 2
    pd_location = "premium" if price > midpoint else "discount"

    score = 0.0
    reasons: List[str] = []

    if ma_stack == "bullish":
        score += 1.0
        reasons.append("EMA 9 above SMA 20")
    elif ma_stack == "bearish":
        score -= 1.0
        reasons.append("EMA 9 below SMA 20")

    if price_vs_ma == "above both":
        score += 0.75
        reasons.append("price above EMA 9 and SMA 20")
    elif price_vs_ma == "below both":
        score -= 0.75
        reasons.append("price below EMA 9 and SMA 20")

    if ema_slope == "up":
        score += 0.5
        reasons.append("EMA 9 slope rising")
    else:
        score -= 0.5
        reasons.append("EMA 9 slope falling")

    if structure == "BOS up":
        score += 1.5
        reasons.append("break of structure up")
    elif structure == "BOS down":
        score -= 1.5
        reasons.append("break of structure down")
    elif structure == "bullish structure":
        score += 1.0
        reasons.append("higher-high / higher-low sequence")
    elif structure == "bearish structure":
        score -= 1.0
        reasons.append("lower-high / lower-low sequence")
    else:
        reasons.append("range or transition")

    if liquidity == "sell-side sweep":
        score += 0.5
        reasons.append("sell-side liquidity sweep")
    elif liquidity == "buy-side sweep":
        score -= 0.5
        reasons.append("buy-side liquidity sweep")

    if pd_location == "discount":
        reasons.append("trading in discount of active swing")
    else:
        reasons.append("trading in premium of active swing")

    return TimeframeSignal(
        timeframe=timeframe,
        price=price,
        ema9=ema9,
        sma20=sma20,
        ma_stack=ma_stack,
        price_vs_ma=price_vs_ma,
        ema_slope=ema_slope,
        structure=structure,
        swing_high=swing_high,
        swing_low=swing_low,
        pd_location=pd_location,
        liquidity=liquidity,
        score=score,
        reasons=reasons,
    )


def _composite_bias(tf_signals: Dict[str, TimeframeSignal], funding_rate: float) -> Tuple[str, float, List[str]]:
    weights = {"4h": 4.0, "1h": 3.0, "15m": 2.0, "1m": 1.0}
    total = sum(tf_signals[tf].score * weights[tf] for tf in TIMEFRAMES)
    reasons: List[str] = []

    higher = [tf_signals["4h"], tf_signals["1h"]]
    lower = [tf_signals["15m"], tf_signals["1m"]]

    if all(tf.score > 0 for tf in higher):
        reasons.append("higher timeframes aligned bullish")
        total += 1.0
    elif all(tf.score < 0 for tf in higher):
        reasons.append("higher timeframes aligned bearish")
        total -= 1.0
    else:
        reasons.append("higher timeframes mixed")

    if all(tf.score > 0 for tf in lower):
        reasons.append("execution timeframes aligned bullish")
        total += 0.5
    elif all(tf.score < 0 for tf in lower):
        reasons.append("execution timeframes aligned bearish")
        total -= 0.5
    else:
        reasons.append("execution timeframes mixed")

    if funding_rate > 0.001:
        total -= 0.5
        reasons.append(f"funding elevated at {funding_rate * 100:+.4f}%")
    elif funding_rate < -0.001:
        total += 0.5
        reasons.append(f"funding negative at {funding_rate * 100:+.4f}%")
    else:
        reasons.append(f"funding neutral at {funding_rate * 100:+.4f}%")

    if total >= 6:
        return "LONG", total, reasons
    if total <= -6:
        return "SHORT", total, reasons
    return "NEUTRAL", total, reasons


def fmt_vol(v: float) -> str:
    if v >= 1e9:
        return f"${v / 1e9:.2f}B"
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:.0f}"


def _request_json(url: str, params: Optional[dict] = None, timeout: int = 10) -> dict:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def binance_ping() -> bool:
    try:
        return requests.get(f"{ENDPOINTS['binance']}/ping", timeout=5).status_code == 200
    except Exception:
        return False


def binance_get_tickers() -> List[MarketMeta]:
    data = _request_json(f"{ENDPOINTS['binance']}/ticker/24hr")
    items = [x for x in data if x["symbol"].endswith("USDT") and float(x["quoteVolume"]) > 0]
    items.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [
        MarketMeta(
            symbol=x["symbol"],
            price=float(x["lastPrice"]),
            change_24h=float(x["priceChangePercent"]),
            volume_usdt=float(x["quoteVolume"]),
            venue_id=x["symbol"],
        )
        for x in items
    ]


def binance_get_klines(symbol: str, interval: str, limit: int) -> List[Dict[str, float]]:
    data = _request_json(
        f"{ENDPOINTS['binance']}/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
    )
    return [
        {
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
        }
        for k in data
    ]


def binance_get_funding(symbol: str) -> float:
    try:
        data = _request_json(
            f"{ENDPOINTS['binance']}/fundingRate",
            params={"symbol": symbol, "limit": 1},
            timeout=5,
        )
        return float(data[-1]["fundingRate"]) if data else 0.0
    except Exception:
        return 0.0


def bybit_ping() -> bool:
    try:
        return requests.get(f"{ENDPOINTS['bybit']}/market/time", timeout=5).status_code == 200
    except Exception:
        return False


def bybit_get_tickers() -> List[MarketMeta]:
    data = _request_json(f"{ENDPOINTS['bybit']}/market/tickers", params={"category": "linear"})
    items = data.get("result", {}).get("list", [])
    items = [x for x in items if x["symbol"].endswith("USDT") and float(x.get("turnover24h", 0)) > 0]
    items.sort(key=lambda x: float(x.get("turnover24h", 0)), reverse=True)
    return [
        MarketMeta(
            symbol=x["symbol"],
            price=float(x["lastPrice"]),
            change_24h=float(x.get("price24hPcnt", 0)) * 100,
            volume_usdt=float(x.get("turnover24h", 0)),
            venue_id=x["symbol"],
        )
        for x in items
    ]


def bybit_get_klines(symbol: str, interval: str, limit: int) -> List[Dict[str, float]]:
    data = _request_json(
        f"{ENDPOINTS['bybit']}/market/kline",
        params={
            "category": "linear",
            "symbol": symbol,
            "interval": _to_bybit_interval(interval),
            "limit": limit,
        },
    )
    rows = list(reversed(data.get("result", {}).get("list", [])))
    return [
        {
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
        }
        for k in rows
    ]


def bybit_get_funding(symbol: str) -> float:
    try:
        data = _request_json(
            f"{ENDPOINTS['bybit']}/market/funding/history",
            params={"category": "linear", "symbol": symbol, "limit": 1},
            timeout=5,
        )
        rows = data.get("result", {}).get("list", [])
        return float(rows[0]["fundingRate"]) if rows else 0.0
    except Exception:
        return 0.0


def okx_ping() -> bool:
    try:
        response = requests.get(f"{ENDPOINTS['okx']}/public/time", timeout=5)
        return response.status_code == 200 and response.json().get("code") == "0"
    except Exception:
        return False


def okx_get_tickers() -> List[MarketMeta]:
    data = _request_json(f"{ENDPOINTS['okx']}/market/tickers", params={"instType": "SWAP"})
    items = data.get("data", [])
    items = [x for x in items if x["instId"].endswith("-USDT-SWAP") and float(x.get("volCcy24h", 0)) > 0]
    items.sort(key=lambda x: float(x.get("volCcy24h", 0)), reverse=True)
    metas = []
    for x in items:
        price = float(x["last"])
        open24h = float(x.get("open24h", price) or price)
        change_24h = ((price - open24h) / open24h * 100) if open24h else 0.0
        metas.append(
            MarketMeta(
                symbol=x["instId"].replace("-USDT-SWAP", "USDT"),
                price=price,
                change_24h=change_24h,
                volume_usdt=float(x.get("volCcy24h", 0)),
                venue_id=x["instId"],
            )
        )
    return metas


def okx_get_klines(inst_id: str, interval: str, limit: int) -> List[Dict[str, float]]:
    data = _request_json(
        f"{ENDPOINTS['okx']}/market/candles",
        params={"instId": inst_id, "bar": _to_okx_interval(interval), "limit": limit},
    )
    rows = list(reversed(data.get("data", [])))
    return [
        {
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
        }
        for k in rows
    ]


def okx_get_funding(inst_id: str) -> float:
    try:
        data = _request_json(f"{ENDPOINTS['okx']}/public/funding-rate", params={"instId": inst_id}, timeout=5)
        rows = data.get("data", [])
        return float(rows[0]["fundingRate"]) if rows else 0.0
    except Exception:
        return 0.0


EXCHANGE_ADAPTERS = {
    "binance": (binance_ping, binance_get_tickers, binance_get_klines, binance_get_funding),
    "bybit": (bybit_ping, bybit_get_tickers, bybit_get_klines, bybit_get_funding),
    "okx": (okx_ping, okx_get_tickers, okx_get_klines, okx_get_funding),
}


def detect_exchange() -> str:
    for name in ("binance", "bybit", "okx"):
        ping_fn = EXCHANGE_ADAPTERS[name][0]
        print(f"  Testing {name}...", end=" ", flush=True)
        if ping_fn():
            print("OK")
            return name
        print("blocked")
    raise RuntimeError("No exchange endpoint reachable. Try --exchange bybit, --exchange okx, or --demo.")


def _gen_demo_candles(start: float, drift: float, vol: float, count: int) -> List[Dict[str, float]]:
    random.seed(hash((start, drift, vol, count)) & 0xFFFF)
    closes = [start]
    for _ in range(count - 1):
        closes.append(max(closes[-1] * (1 + drift + random.gauss(0, vol)), 0.0001))
    candles = []
    prev = closes[0]
    for close in closes:
        high = max(prev, close) * (1 + abs(random.gauss(0, vol / 2)))
        low = min(prev, close) * max(0.0001, (1 - abs(random.gauss(0, vol / 2))))
        candles.append({"open": prev, "high": high, "low": low, "close": close})
        prev = close
    return candles


def run_demo(top_n: int) -> None:
    demo = [
        ("BTCUSDT", 67500, 32.4e9, 0.00012, {"4h": (0.0025, 0.007), "1h": (0.0020, 0.006), "15m": (0.0015, 0.005), "1m": (0.0006, 0.003)}),
        ("ETHUSDT", 3520, 18.7e9, 0.00008, {"4h": (0.0018, 0.008), "1h": (0.0012, 0.006), "15m": (0.0008, 0.004), "1m": (0.0003, 0.002)}),
        ("SOLUSDT", 175, 8.2e9, 0.00020, {"4h": (0.0030, 0.012), "1h": (0.0022, 0.010), "15m": (0.0015, 0.008), "1m": (0.0005, 0.004)}),
        ("XRPUSDT", 0.52, 6.3e9, -0.00018, {"4h": (-0.0016, 0.010), "1h": (-0.0010, 0.008), "15m": (-0.0008, 0.006), "1m": (-0.0002, 0.003)}),
    ]
    if top_n > 0:
        demo = demo[:top_n]
    signals = []
    for symbol, price, volume_usdt, funding_rate, tf_config in demo:
        timeframe_signals = {}
        for timeframe in TIMEFRAMES:
            drift, vol = tf_config[timeframe]
            timeframe_signals[timeframe] = _analyze_timeframe(timeframe, _gen_demo_candles(price, drift, vol, KLINE_LIMIT))
        direction, score, reasons = _composite_bias(timeframe_signals, funding_rate)
        signals.append(
            CoinSignal(
                symbol=symbol,
                price=timeframe_signals["1m"].price,
                change_24h=0.0,
                volume_usdt=volume_usdt,
                funding_rate=funding_rate,
                direction=direction,
                score=score,
                bias="demo",
                timeframe_signals=timeframe_signals,
                reasons=reasons,
            )
        )
    print_report(signals, exchange="demo")


def build_signals(exchange: str, top_n: int, symbols: Optional[List[str]]) -> List[CoinSignal]:
    _, get_tickers, get_klines, get_funding = EXCHANGE_ADAPTERS[exchange]
    markets = get_tickers()

    symbol_set = {s.upper() for s in symbols} if symbols else None
    if symbol_set:
        markets = [m for m in markets if m.symbol.upper() in symbol_set]
    if top_n > 0:
        markets = markets[:top_n]

    print(f"Scanning {len(markets)} USDT perpetual pairs on {exchange}...")

    signals: List[CoinSignal] = []
    for idx, market in enumerate(markets, 1):
        print(f"  [{idx:>4}/{len(markets)}] {market.symbol}...", end="\r", flush=True)
        try:
            timeframe_signals = {}
            for timeframe in TIMEFRAMES:
                candles = get_klines(market.venue_id, timeframe, KLINE_LIMIT)
                if len(candles) < max(SMA_LENGTH, EMA_LENGTH) + 5:
                    raise RuntimeError(f"not enough candles for {timeframe}")
                timeframe_signals[timeframe] = _analyze_timeframe(timeframe, candles)
                time.sleep(0.03)

            funding_rate = get_funding(market.venue_id)
            direction, score, reasons = _composite_bias(timeframe_signals, funding_rate)
            signals.append(
                CoinSignal(
                    symbol=market.symbol,
                    price=timeframe_signals["1m"].price,
                    change_24h=market.change_24h,
                    volume_usdt=market.volume_usdt,
                    funding_rate=funding_rate,
                    direction=direction,
                    score=score,
                    bias=timeframe_signals["4h"].structure,
                    timeframe_signals=timeframe_signals,
                    reasons=reasons,
                )
            )
        except Exception as exc:
            print(f"\n  Error on {market.symbol}: {exc}")
    print(" " * 80, end="\r")
    return signals


def print_report(signals: List[CoinSignal], exchange: str) -> None:
    longs = sorted([s for s in signals if s.direction == "LONG"], key=lambda s: s.score, reverse=True)
    shorts = sorted([s for s in signals if s.direction == "SHORT"], key=lambda s: s.score)
    neutral = sorted([s for s in signals if s.direction == "NEUTRAL"], key=lambda s: abs(s.score), reverse=True)

    width = 110
    print("\n" + "=" * width)
    print(f"MULTI-TIMEFRAME FUTURES STRUCTURE SCANNER [{exchange.upper()}]")
    print(f"Pairs: {len(signals)} | Timeframes: {', '.join(TIMEFRAMES)} | EMA {EMA_LENGTH} | SMA {SMA_LENGTH} | SMC-lite")
    print("=" * width + "\n")

    sections = [
        ("LONG", longs),
        ("SHORT", shorts),
        ("NEUTRAL", neutral),
    ]
    for title, items in sections:
        print("-" * width)
        print(f"{title} ({len(items)})")
        print("-" * width)
        if not items:
            print("none\n")
            continue
        for item in items:
            print(
                f"{item.symbol:<14} price={item.price:>12,.6f} 24h={item.change_24h:>+6.2f}% "
                f"vol={fmt_vol(item.volume_usdt):<9} funding={item.funding_rate * 100:+.4f}% score={item.score:+6.2f}"
            )
            print("  composite: " + "; ".join(item.reasons))
            for timeframe in TIMEFRAMES:
                tf = item.timeframe_signals[timeframe]
                print(
                    f"  {timeframe:<3} {tf.structure:<18} | {tf.ma_stack:<7} | {tf.price_vs_ma:<12} | "
                    f"EMA slope {tf.ema_slope:<4} | {tf.pd_location:<8} | liquidity: {tf.liquidity}"
                )
            print()

    print("=" * width)
    print("Use this as a directional map, not a standalone trading system.")
    print("=" * width + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-timeframe scanner for USDT perpetuals")
    parser.add_argument("--exchange", default="auto", choices=["auto", "binance", "bybit", "okx"], help="Exchange to scan")
    parser.add_argument("--demo", action="store_true", help="Run with generated demo data")
    parser.add_argument("--top", type=int, default=TOP_N, help="Limit pair count. 0 means all pairs")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols to scan, for example BTCUSDT,SOLUSDT")
    return parser.parse_args()


def run_live(forced_exchange: str, top_n: int, symbols: Optional[List[str]]) -> None:
    if not _HAS_REQUESTS:
        raise RuntimeError("Install requests: pip install requests")
    exchange = forced_exchange if forced_exchange != "auto" else detect_exchange()
    signals = build_signals(exchange, top_n, symbols)
    print_report(signals, exchange)


def main() -> None:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or None

    if args.demo:
        run_demo(args.top)
        return

    try:
        run_live(args.exchange, args.top, symbols)
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        print("Tip: try --exchange bybit, --exchange okx, or --demo\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
