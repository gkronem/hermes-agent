"""
Binance Futures Signal Scanner
Analisa pares USDT perpetuos e identifica oportunidades de long/short
usando RSI, EMA, MACD e funding rate (apenas APIs públicas).

Uso:
    python binance_futures_signals.py           # dados reais da Binance
    python binance_futures_signals.py --demo    # modo demonstração (sem internet)
"""

import sys
import time
import math
import random
import argparse
from dataclasses import dataclass, field
from typing import Optional

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

FAPI        = "https://fapi.binance.com/fapi/v1"
TOP_N       = 15         # top N pares por volume para analisar (sobrescrito por --top)
KLINE_LIMIT = 100        # velas para cálculo dos indicadores
INTERVAL    = "1h"       # sobrescrito por --interval

RSI_OVERSOLD   = 35
RSI_OVERBOUGHT = 65
EMA_FAST       = 9
EMA_SLOW       = 21


# ── Estrutura de dados ───────────────────────────────────────────────────────

@dataclass
class Signal:
    symbol: str
    price: float
    change_24h: float
    volume_usdt: float
    rsi: Optional[float]
    ema_fast: Optional[float]
    ema_slow: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    funding_rate: Optional[float]
    score: float = 0.0
    direction: str = "NEUTRO"
    reasons: list = field(default_factory=list)


# ── Indicadores técnicos (puro Python, sem numpy) ───────────────────────────

def ema(closes: list, period: int) -> list:
    k = 2 / (period + 1)
    result = [closes[0]]
    for c in closes[1:]:
        result.append(c * k + result[-1] * (1 - k))
    return result


def rsi(closes: list, period: int = 14) -> float:
    if len(closes) < period + 1:
        return float("nan")
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))


def macd_values(closes: list) -> tuple:
    """Retorna (macd_line, signal_line)."""
    if len(closes) < 35:
        return float("nan"), float("nan")
    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    macd_line = [m - s for m, s in zip(ema12, ema26)]
    signal_line = ema(macd_line, 9)
    return macd_line[-1], signal_line[-1]


# ── API Binance ──────────────────────────────────────────────────────────────

def _get(url: str, params: dict = None, timeout: int = 10):
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def get_top_symbols(n: int) -> list:
    tickers = [
        t for t in _get(f"{FAPI}/ticker/24hr")
        if t["symbol"].endswith("USDT") and float(t["quoteVolume"]) > 0
    ]
    tickers.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)
    return tickers[:n]


def get_klines(symbol: str) -> list:
    data = _get(f"{FAPI}/klines", {"symbol": symbol, "interval": INTERVAL, "limit": KLINE_LIMIT})
    return [float(k[4]) for k in data]


def get_funding_rates(symbols: list) -> dict:
    rates = {}
    for sym in symbols:
        try:
            data = _get(f"{FAPI}/fundingRate", {"symbol": sym, "limit": 1}, timeout=5)
            rates[sym] = float(data[-1]["fundingRate"]) if data else 0.0
        except Exception:
            rates[sym] = 0.0
        time.sleep(0.05)
    return rates


# ── Dados de demonstração ────────────────────────────────────────────────────

def _gen_trend_closes(start: float, drift: float, volatility: float, n: int) -> list:
    """Gera série de preços com tendência e ruído."""
    random.seed(hash(str(start)))
    closes = [start]
    for _ in range(n - 1):
        change = closes[-1] * (drift + random.gauss(0, volatility))
        closes.append(max(closes[-1] + change, 0.001))
    return closes


DEMO_PAIRS = [
    # (symbol, price, change24h, vol_B, drift, volatility, funding)
    ("BTCUSDT",   67500.0,  +2.3,  32.4,  +0.003, 0.008,  +0.00012),
    ("ETHUSDT",    3520.0,  +1.8,  18.7,  +0.002, 0.010,  +0.00008),
    ("SOLUSDT",     175.0,  +5.1,   8.2,  +0.005, 0.015,  +0.00020),
    ("BNBUSDT",     580.0,  -0.4,   4.1,  -0.001, 0.009,  -0.00005),
    ("XRPUSDT",       0.52,  -3.2,  6.3,  -0.004, 0.012,  -0.00018),
    ("DOGEUSDT",      0.15,  -5.7,  5.8,  -0.006, 0.018,  -0.00025),
    ("ADAUSDT",       0.48,  +0.9,  3.2,  +0.001, 0.011,  +0.00003),
    ("AVAXUSDT",     38.5,   -2.1,  2.9,  -0.002, 0.013,  -0.00010),
    ("DOTUSDT",       7.8,   -4.3,  2.1,  -0.005, 0.014,  -0.00022),
    ("LINKUSDT",     15.2,   +3.6,  1.8,  +0.004, 0.012,  +0.00015),
    ("LTCUSDT",      82.0,   +1.1,  1.5,  +0.001, 0.009,  +0.00004),
    ("MATICUSDT",     0.72,  -6.2,  1.3,  -0.007, 0.020,  -0.00030),
    ("NEARUSDT",      5.4,   +4.8,  1.1,  +0.005, 0.016,  +0.00018),
    ("ARBUSDT",       1.12,  -1.8,  0.9,  -0.002, 0.015,  -0.00008),
    ("OPUSDT",        2.45,  +2.7,  0.8,  +0.003, 0.014,  +0.00011),
]


def get_demo_data() -> tuple:
    tickers = []
    klines_map = {}
    funding = {}
    for sym, price, chg, vol_b, drift, vol, fr in DEMO_PAIRS:
        tickers.append({
            "symbol": sym,
            "lastPrice": str(price),
            "priceChangePercent": str(chg),
            "quoteVolume": str(vol_b * 1e9),
        })
        klines_map[sym] = _gen_trend_closes(price, drift, vol, KLINE_LIMIT)
        funding[sym] = fr
    return tickers, klines_map, funding


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_signal(sig: Signal) -> Signal:
    score = 0.0
    reasons = []

    if sig.rsi is not None and not math.isnan(sig.rsi):
        if sig.rsi < RSI_OVERSOLD:
            score += 2
            reasons.append(f"RSI {sig.rsi:.1f} → sobrevendido (LONG)")
        elif sig.rsi > RSI_OVERBOUGHT:
            score -= 2
            reasons.append(f"RSI {sig.rsi:.1f} → sobrecomprado (SHORT)")
        else:
            reasons.append(f"RSI {sig.rsi:.1f} → zona neutra")

    if sig.ema_fast and sig.ema_slow:
        if sig.ema_fast > sig.ema_slow:
            score += 1.5
            reasons.append(f"EMA{EMA_FAST} > EMA{EMA_SLOW} → tendência de alta (LONG)")
        else:
            score -= 1.5
            reasons.append(f"EMA{EMA_FAST} < EMA{EMA_SLOW} → tendência de baixa (SHORT)")

    if sig.macd is not None and sig.macd_signal is not None \
       and not math.isnan(sig.macd) and not math.isnan(sig.macd_signal):
        if sig.macd > sig.macd_signal:
            score += 1
            reasons.append("MACD acima do sinal → momentum positivo (LONG)")
        else:
            score -= 1
            reasons.append("MACD abaixo do sinal → momentum negativo (SHORT)")

    if sig.funding_rate is not None:
        fr_pct = sig.funding_rate * 100
        if sig.funding_rate > 0.001:
            score -= 1
            reasons.append(f"Funding {fr_pct:+.4f}% → longs pagando caro (SHORT)")
        elif sig.funding_rate < -0.001:
            score += 1
            reasons.append(f"Funding {fr_pct:+.4f}% → shorts pagando caro (LONG)")
        else:
            reasons.append(f"Funding {fr_pct:+.4f}% → neutro")

    if sig.change_24h > 3:
        score += 0.5
        reasons.append(f"Variação 24h {sig.change_24h:+.1f}% → momentum positivo")
    elif sig.change_24h < -3:
        score -= 0.5
        reasons.append(f"Variação 24h {sig.change_24h:+.1f}% → momentum negativo")

    sig.score = score
    sig.reasons = reasons
    if score >= 2:
        sig.direction = "LONG"
    elif score <= -2:
        sig.direction = "SHORT"
    else:
        sig.direction = "NEUTRO"
    return sig


# ── Relatório ────────────────────────────────────────────────────────────────

def fmt_vol(v: float) -> str:
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    if v >= 1e6:
        return f"${v/1e6:.1f}M"
    return f"${v/1e3:.0f}K"


def print_report(signals: list, demo: bool = False) -> None:
    longs  = sorted([s for s in signals if s.direction == "LONG"],  key=lambda s: s.score, reverse=True)
    shorts = sorted([s for s in signals if s.direction == "SHORT"], key=lambda s: s.score)
    neutro = [s for s in signals if s.direction == "NEUTRO"]

    W = 74
    tag = " [MODO DEMONSTRAÇÃO]" if demo else ""

    print(f"\n{'═'*W}")
    print(f"  BINANCE FUTURES — SCANNER DE SINAIS{tag}")
    print(f"  Timeframe: {INTERVAL} | Pares analisados: {len(signals)}")
    print(f"  Indicadores: RSI-14, EMA{EMA_FAST}/EMA{EMA_SLOW}, MACD, Funding Rate")
    print(f"{'═'*W}\n")

    sections = [
        ("🟢  LONG  — Candidatos a Compra", longs,  True),
        ("🔴  SHORT — Candidatos a Venda",  shorts, True),
        ("⚪  NEUTRO — Sem Sinal Claro",    neutro, False),
    ]

    for title, items, show_details in sections:
        print(f"{'─'*W}")
        print(f"  {title}  ({len(items)} par(es))")
        print(f"{'─'*W}")
        if not items:
            print("  —\n")
            continue
        for s in items:
            fr_str = f"{s.funding_rate*100:+.4f}%" if s.funding_rate is not None else "n/a"
            print(
                f"  {s.symbol:<14}  "
                f"Preço: {s.price:>12,.4f}  "
                f"24h: {s.change_24h:>+6.2f}%  "
                f"Vol: {fmt_vol(s.volume_usdt):<9}  "
                f"Score: {s.score:>+5.1f}"
            )
            if show_details:
                for r in s.reasons:
                    print(f"     • {r}")
            print()

    print(f"{'═'*W}")
    print("  ⚠  Esta análise é apenas informativa. Faça sua própria pesquisa.")
    print("     Sempre utilize stop-loss e gerencie seu risco.")
    print(f"{'═'*W}\n")


# ── Pipeline principal ───────────────────────────────────────────────────────

def run_live() -> None:
    if not _HAS_REQUESTS:
        raise RuntimeError("Instale 'requests': pip install requests")

    print(f"Buscando top {TOP_N} pares USDT por volume...")
    tickers = get_top_symbols(TOP_N)
    symbols = [t["symbol"] for t in tickers]

    print(f"Buscando funding rates para {len(symbols)} pares...")
    funding = get_funding_rates(symbols)

    signals = []
    for i, t in enumerate(tickers, 1):
        sym = t["symbol"]
        print(f"  [{i:>2}/{len(tickers)}] {sym}...", end="\r")
        try:
            closes = get_klines(sym)
            sig = Signal(
                symbol       = sym,
                price        = float(t["lastPrice"]),
                change_24h   = float(t["priceChangePercent"]),
                volume_usdt  = float(t["quoteVolume"]),
                rsi          = rsi(closes),
                ema_fast     = ema(closes, EMA_FAST)[-1],
                ema_slow     = ema(closes, EMA_SLOW)[-1],
                macd         = macd_values(closes)[0],
                macd_signal  = macd_values(closes)[1],
                funding_rate = funding.get(sym),
            )
            signals.append(score_signal(sig))
        except Exception as e:
            print(f"\n  Erro em {sym}: {e}")
        time.sleep(0.1)

    print(" " * 50)
    print_report(signals, demo=False)


def run_demo() -> None:
    print("Gerando análise com dados de demonstração...")
    tickers, klines_map, funding = get_demo_data()

    signals = []
    for t in tickers:
        sym = t["symbol"]
        closes = klines_map[sym]
        m, ms = macd_values(closes)
        sig = Signal(
            symbol       = sym,
            price        = float(t["lastPrice"]),
            change_24h   = float(t["priceChangePercent"]),
            volume_usdt  = float(t["quoteVolume"]),
            rsi          = rsi(closes),
            ema_fast     = ema(closes, EMA_FAST)[-1],
            ema_slow     = ema(closes, EMA_SLOW)[-1],
            macd         = m,
            macd_signal  = ms,
            funding_rate = funding.get(sym),
        )
        signals.append(score_signal(sig))

    print_report(signals, demo=True)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    import binance_futures_signals as _self

    parser = argparse.ArgumentParser(description="Scanner de sinais para futuros da Binance")
    parser.add_argument("--demo",     action="store_true", help="Usar dados de demonstração (sem internet)")
    parser.add_argument("--top",      type=int, default=TOP_N,    help=f"Número de pares a analisar (padrão: {TOP_N})")
    parser.add_argument("--interval", default=INTERVAL,           help="Intervalo das velas (ex: 15m, 1h, 4h, 1d)")
    args = parser.parse_args()

    _self.TOP_N    = args.top
    _self.INTERVAL = args.interval

    if args.demo:
        run_demo()
    else:
        try:
            run_live()
        except Exception as e:
            print(f"\n[ERRO] Não foi possível conectar à Binance: {e}")
            print("Dica: execute com --demo para ver um exemplo da análise.\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
