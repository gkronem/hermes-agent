"""
Futures Signal Scanner — Binance / Bybit / OKX
Analisa pares USDT perpétuos e identifica oportunidades de long/short
usando RSI, EMA, MACD e funding rate (apenas APIs públicas, sem autenticação).

Uso:
    python binance_futures_signals.py                    # detecta exchange automaticamente
    python binance_futures_signals.py --exchange bybit   # força exchange
    python binance_futures_signals.py --demo             # modo demonstração (sem internet)
    python binance_futures_signals.py --top 20 --interval 4h
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

# Configurações (sobrescritas pelos args)
TOP_N       = 15
KLINE_LIMIT = 100
INTERVAL    = "1h"
EXCHANGE    = "auto"   # auto | binance | bybit | okx

RSI_OVERSOLD   = 35
RSI_OVERBOUGHT = 65
EMA_FAST       = 9
EMA_SLOW       = 21

# Endpoints base
ENDPOINTS = {
    "binance": "https://fapi.binance.com/fapi/v1",
    "bybit":   "https://api.bybit.com/v5",
    "okx":     "https://www.okx.com/api/v5",
}


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

def _ema(closes: list, period: int) -> list:
    k = 2 / (period + 1)
    result = [closes[0]]
    for c in closes[1:]:
        result.append(c * k + result[-1] * (1 - k))
    return result


def calc_rsi(closes: list, period: int = 14) -> float:
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


def calc_macd(closes: list) -> tuple:
    if len(closes) < 35:
        return float("nan"), float("nan")
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = [m - s for m, s in zip(ema12, ema26)]
    signal_line = _ema(macd_line, 9)
    return macd_line[-1], signal_line[-1]


# ── Conversão de intervalo entre exchanges ───────────────────────────────────

def _to_bybit_interval(iv: str) -> str:
    mapping = {
        "1m": "1",  "3m": "3",  "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
        "1d": "D",  "1w": "W",  "1M": "M",
    }
    return mapping.get(iv.lower(), "60")


def _to_okx_interval(iv: str) -> str:
    mapping = {
        "1m": "1m",  "3m": "3m",  "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1H",  "2h": "2H",  "4h": "4H", "6h": "6H",  "12h": "12H",
        "1d": "1D",  "1w": "1W",
    }
    return mapping.get(iv.lower(), "1H")


# ── Adaptador Binance ────────────────────────────────────────────────────────

def binance_ping() -> bool:
    try:
        r = requests.get(f"{ENDPOINTS['binance']}/ping", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def binance_get_tickers(n: int) -> list:
    r = requests.get(f"{ENDPOINTS['binance']}/ticker/24hr", timeout=10)
    r.raise_for_status()
    tickers = [t for t in r.json() if t["symbol"].endswith("USDT") and float(t["quoteVolume"]) > 0]
    tickers.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)
    return [
        {"symbol": t["symbol"], "price": float(t["lastPrice"]),
         "change_24h": float(t["priceChangePercent"]), "volume_usdt": float(t["quoteVolume"])}
        for t in tickers[:n]
    ]


def binance_get_klines(symbol: str) -> list:
    r = requests.get(f"{ENDPOINTS['binance']}/klines",
                     params={"symbol": symbol, "interval": INTERVAL, "limit": KLINE_LIMIT}, timeout=10)
    r.raise_for_status()
    return [float(k[4]) for k in r.json()]   # close prices, oldest→newest


def binance_get_funding(symbol: str) -> float:
    try:
        r = requests.get(f"{ENDPOINTS['binance']}/fundingRate",
                         params={"symbol": symbol, "limit": 1}, timeout=5)
        data = r.json()
        return float(data[-1]["fundingRate"]) if data else 0.0
    except Exception:
        return 0.0


# ── Adaptador Bybit ──────────────────────────────────────────────────────────

def bybit_ping() -> bool:
    try:
        r = requests.get(f"{ENDPOINTS['bybit']}/market/time", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def bybit_get_tickers(n: int) -> list:
    r = requests.get(f"{ENDPOINTS['bybit']}/market/tickers",
                     params={"category": "linear"}, timeout=10)
    r.raise_for_status()
    items = r.json().get("result", {}).get("list", [])
    items = [t for t in items if t["symbol"].endswith("USDT") and float(t.get("turnover24h", 0)) > 0]
    items.sort(key=lambda t: float(t.get("turnover24h", 0)), reverse=True)
    result = []
    for t in items[:n]:
        price = float(t["lastPrice"])
        pct = float(t.get("price24hPcnt", 0)) * 100
        vol = float(t.get("turnover24h", 0))
        result.append({"symbol": t["symbol"], "price": price, "change_24h": pct, "volume_usdt": vol})
    return result


def bybit_get_klines(symbol: str) -> list:
    r = requests.get(f"{ENDPOINTS['bybit']}/market/kline",
                     params={"category": "linear", "symbol": symbol,
                             "interval": _to_bybit_interval(INTERVAL), "limit": KLINE_LIMIT}, timeout=10)
    r.raise_for_status()
    data = r.json().get("result", {}).get("list", [])
    # Bybit retorna newest first — inverter para oldest→newest
    closes = [float(k[4]) for k in reversed(data)]
    return closes


def bybit_get_funding(symbol: str) -> float:
    try:
        r = requests.get(f"{ENDPOINTS['bybit']}/market/funding/history",
                         params={"category": "linear", "symbol": symbol, "limit": 1}, timeout=5)
        data = r.json().get("result", {}).get("list", [])
        return float(data[0]["fundingRate"]) if data else 0.0
    except Exception:
        return 0.0


# ── Adaptador OKX ────────────────────────────────────────────────────────────

def okx_ping() -> bool:
    try:
        r = requests.get(f"{ENDPOINTS['okx']}/public/time", timeout=5)
        return r.status_code == 200 and r.json().get("code") == "0"
    except Exception:
        return False


def okx_get_tickers(n: int) -> list:
    r = requests.get(f"{ENDPOINTS['okx']}/market/tickers",
                     params={"instType": "SWAP"}, timeout=10)
    r.raise_for_status()
    items = r.json().get("data", [])
    items = [t for t in items if t["instId"].endswith("-USDT-SWAP") and float(t.get("volCcy24h", 0)) > 0]
    items.sort(key=lambda t: float(t.get("volCcy24h", 0)), reverse=True)
    result = []
    for t in items[:n]:
        price = float(t["last"])
        open24h = float(t.get("open24h", price) or price)
        pct = ((price - open24h) / open24h * 100) if open24h else 0
        vol = float(t.get("volCcy24h", 0))
        sym = t["instId"].replace("-USDT-SWAP", "USDT")
        result.append({"symbol": sym, "price": price, "change_24h": pct, "volume_usdt": vol,
                        "_okx_id": t["instId"]})
    return result


def okx_get_klines(inst_id: str) -> list:
    r = requests.get(f"{ENDPOINTS['okx']}/market/candles",
                     params={"instId": inst_id, "bar": _to_okx_interval(INTERVAL), "limit": KLINE_LIMIT},
                     timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    closes = [float(k[4]) for k in reversed(data)]   # newest first → reverse
    return closes


def okx_get_funding(inst_id: str) -> float:
    try:
        r = requests.get(f"{ENDPOINTS['okx']}/public/funding-rate",
                         params={"instId": inst_id}, timeout=5)
        data = r.json().get("data", [])
        return float(data[0]["fundingRate"]) if data else 0.0
    except Exception:
        return 0.0


# ── Detecção automática de exchange ─────────────────────────────────────────

EXCHANGE_ADAPTERS = {
    "binance": (binance_ping, binance_get_tickers, binance_get_klines, binance_get_funding),
    "bybit":   (bybit_ping,   bybit_get_tickers,   bybit_get_klines,   bybit_get_funding),
    "okx":     (okx_ping,     okx_get_tickers,     okx_get_klines,     okx_get_funding),
}


def detect_exchange() -> str:
    for name in ("binance", "bybit", "okx"):
        ping_fn = EXCHANGE_ADAPTERS[name][0]
        print(f"  Testando {name}...", end=" ", flush=True)
        if ping_fn():
            print("OK")
            return name
        print("bloqueado")
    raise RuntimeError("Nenhuma exchange acessível. Use --demo para modo demonstração.")


# ── Dados de demonstração ────────────────────────────────────────────────────

def _gen_closes(start: float, drift: float, vol: float, n: int) -> list:
    random.seed(hash(str(start)) & 0xFFFF)
    closes = [start]
    for _ in range(n - 1):
        closes.append(max(closes[-1] * (1 + drift + random.gauss(0, vol)), 0.001))
    return closes


DEMO_PAIRS = [
    ("BTCUSDT",   67500, +2.3, 32.4e9, +0.003, 0.008, +0.00012),
    ("ETHUSDT",    3520, +1.8, 18.7e9, +0.002, 0.010, +0.00008),
    ("SOLUSDT",     175, +5.1,  8.2e9, +0.005, 0.015, +0.00020),
    ("BNBUSDT",     580, -0.4,  4.1e9, -0.001, 0.009, -0.00005),
    ("XRPUSDT",    0.52, -3.2,  6.3e9, -0.004, 0.012, -0.00018),
    ("DOGEUSDT",   0.15, -5.7,  5.8e9, -0.006, 0.018, -0.00025),
    ("ADAUSDT",    0.48, +0.9,  3.2e9, +0.001, 0.011, +0.00003),
    ("AVAXUSDT",   38.5, -2.1,  2.9e9, -0.002, 0.013, -0.00010),
    ("DOTUSDT",     7.8, -4.3,  2.1e9, -0.005, 0.014, -0.00022),
    ("LINKUSDT",   15.2, +3.6,  1.8e9, +0.004, 0.012, +0.00015),
    ("LTCUSDT",    82.0, +1.1,  1.5e9, +0.001, 0.009, +0.00004),
    ("MATICUSDT",  0.72, -6.2,  1.3e9, -0.007, 0.020, -0.00030),
    ("NEARUSDT",    5.4, +4.8,  1.1e9, +0.005, 0.016, +0.00018),
    ("ARBUSDT",    1.12, -1.8,  0.9e9, -0.002, 0.015, -0.00008),
    ("OPUSDT",     2.45, +2.7,  0.8e9, +0.003, 0.014, +0.00011),
]


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_signal(sig: Signal) -> Signal:
    score = 0.0
    reasons = []

    if sig.rsi is not None and not math.isnan(sig.rsi):
        if sig.rsi < RSI_OVERSOLD:
            score += 2;  reasons.append(f"RSI {sig.rsi:.1f} → sobrevendido (LONG)")
        elif sig.rsi > RSI_OVERBOUGHT:
            score -= 2;  reasons.append(f"RSI {sig.rsi:.1f} → sobrecomprado (SHORT)")
        else:
            reasons.append(f"RSI {sig.rsi:.1f} → zona neutra")

    if sig.ema_fast and sig.ema_slow:
        if sig.ema_fast > sig.ema_slow:
            score += 1.5;  reasons.append(f"EMA{EMA_FAST} > EMA{EMA_SLOW} → tendência de alta (LONG)")
        else:
            score -= 1.5;  reasons.append(f"EMA{EMA_FAST} < EMA{EMA_SLOW} → tendência de baixa (SHORT)")

    if sig.macd is not None and sig.macd_signal is not None \
       and not math.isnan(sig.macd) and not math.isnan(sig.macd_signal):
        if sig.macd > sig.macd_signal:
            score += 1;  reasons.append("MACD acima do sinal → momentum positivo (LONG)")
        else:
            score -= 1;  reasons.append("MACD abaixo do sinal → momentum negativo (SHORT)")

    if sig.funding_rate is not None:
        fr = sig.funding_rate * 100
        if sig.funding_rate > 0.001:
            score -= 1;  reasons.append(f"Funding {fr:+.4f}% → longs pagando caro (SHORT)")
        elif sig.funding_rate < -0.001:
            score += 1;  reasons.append(f"Funding {fr:+.4f}% → shorts pagando caro (LONG)")
        else:
            reasons.append(f"Funding {fr:+.4f}% → neutro")

    if sig.change_24h > 3:
        score += 0.5;  reasons.append(f"Variação 24h {sig.change_24h:+.1f}% → momentum positivo")
    elif sig.change_24h < -3:
        score -= 0.5;  reasons.append(f"Variação 24h {sig.change_24h:+.1f}% → momentum negativo")

    sig.score = score
    sig.reasons = reasons
    sig.direction = "LONG" if score >= 2 else ("SHORT" if score <= -2 else "NEUTRO")
    return sig


# ── Relatório ────────────────────────────────────────────────────────────────

def fmt_vol(v: float) -> str:
    if v >= 1e9: return f"${v/1e9:.2f}B"
    if v >= 1e6: return f"${v/1e6:.1f}M"
    return f"${v/1e3:.0f}K"


def print_report(signals: list, exchange: str = "", demo: bool = False) -> None:
    longs  = sorted([s for s in signals if s.direction == "LONG"],  key=lambda s: s.score, reverse=True)
    shorts = sorted([s for s in signals if s.direction == "SHORT"], key=lambda s: s.score)
    neutro = [s for s in signals if s.direction == "NEUTRO"]

    W   = 74
    src = f" [{exchange.upper()}]" if exchange else ""
    tag = " [MODO DEMONSTRAÇÃO]" if demo else src

    print(f"\n{'═'*W}")
    print(f"  FUTURES SIGNAL SCANNER{tag}")
    print(f"  Timeframe: {INTERVAL} | Pares: {len(signals)} | RSI-14 / EMA{EMA_FAST}/EMA{EMA_SLOW} / MACD / Funding")
    print(f"{'═'*W}\n")

    for title, items, show_detail in [
        ("🟢  LONG  — Candidatos a Compra", longs, True),
        ("🔴  SHORT — Candidatos a Venda",  shorts, True),
        ("⚪  NEUTRO — Sem Sinal Claro",    neutro, False),
    ]:
        print(f"{'─'*W}")
        print(f"  {title}  ({len(items)} par(es))")
        print(f"{'─'*W}")
        if not items:
            print("  —\n"); continue
        for s in items:
            print(
                f"  {s.symbol:<14}  Preço: {s.price:>12,.4f}  "
                f"24h: {s.change_24h:>+6.2f}%  Vol: {fmt_vol(s.volume_usdt):<9}  Score: {s.score:>+5.1f}"
            )
            if show_detail:
                for r in s.reasons:
                    print(f"     • {r}")
            print()

    print(f"{'═'*W}")
    print("  ⚠  Análise técnica não garante resultados. Use stop-loss.")
    print(f"{'═'*W}\n")


# ── Pipelines ────────────────────────────────────────────────────────────────

def build_signals(exchange: str) -> list:
    _, get_tickers, get_klines, get_funding = EXCHANGE_ADAPTERS[exchange]

    print(f"Buscando top {TOP_N} pares USDT ({exchange})...")
    tickers = get_tickers(TOP_N)

    signals = []
    for i, t in enumerate(tickers, 1):
        sym = t["symbol"]
        okx_id = t.get("_okx_id", sym)   # OKX usa instId diferente
        print(f"  [{i:>2}/{len(tickers)}] {sym}...", end="\r", flush=True)
        try:
            closes = get_klines(okx_id if exchange == "okx" else sym)
            if not closes:
                continue
            macd_v, macd_sig = calc_macd(closes)
            fr = get_funding(okx_id if exchange == "okx" else sym)
            time.sleep(0.08)

            sig = Signal(
                symbol       = sym,
                price        = t["price"],
                change_24h   = t["change_24h"],
                volume_usdt  = t["volume_usdt"],
                rsi          = calc_rsi(closes),
                ema_fast     = _ema(closes, EMA_FAST)[-1],
                ema_slow     = _ema(closes, EMA_SLOW)[-1],
                macd         = macd_v,
                macd_signal  = macd_sig,
                funding_rate = fr,
            )
            signals.append(score_signal(sig))
        except Exception as e:
            print(f"\n  Erro em {sym}: {e}")

    print(" " * 55, end="\r")
    return signals


def run_live(forced_exchange: str = "auto") -> None:
    if not _HAS_REQUESTS:
        raise RuntimeError("Instale 'requests': pip install requests")
    exchange = forced_exchange if forced_exchange != "auto" else detect_exchange()
    signals  = build_signals(exchange)
    print_report(signals, exchange=exchange)


def run_demo() -> None:
    print("Gerando análise com dados de demonstração...")
    signals = []
    for sym, price, chg, vol, drift, vol_p, fr in DEMO_PAIRS:
        closes   = _gen_closes(price, drift, vol_p, KLINE_LIMIT)
        macd_v, macd_sig = calc_macd(closes)
        sig = Signal(
            symbol=sym, price=price, change_24h=chg, volume_usdt=vol,
            rsi=calc_rsi(closes),
            ema_fast=_ema(closes, EMA_FAST)[-1],
            ema_slow=_ema(closes, EMA_SLOW)[-1],
            macd=macd_v, macd_signal=macd_sig,
            funding_rate=fr,
        )
        signals.append(score_signal(sig))
    print_report(signals, demo=True)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    import binance_futures_signals as _m

    parser = argparse.ArgumentParser(description="Scanner de sinais para futuros perpétuos USDT")
    parser.add_argument("--exchange", default="auto",   choices=["auto","binance","bybit","okx"], help="Exchange a usar")
    parser.add_argument("--demo",     action="store_true", help="Usar dados de demonstração")
    parser.add_argument("--top",      type=int, default=TOP_N,    help=f"Número de pares (padrão: {TOP_N})")
    parser.add_argument("--interval", default=INTERVAL,           help="Timeframe: 15m 1h 4h 1d (padrão: 1h)")
    args = parser.parse_args()

    _m.TOP_N    = args.top
    _m.INTERVAL = args.interval

    if args.demo:
        run_demo()
    else:
        try:
            run_live(args.exchange)
        except Exception as e:
            print(f"\n[ERRO] {e}")
            print("Dica: tente --exchange bybit, --exchange okx, ou --demo\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
