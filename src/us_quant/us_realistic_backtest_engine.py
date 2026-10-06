from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ExecutionConfig:
    initial_capital: float = 100000.0
    commission_per_share: float = 0.0
    sec_fee_bps_on_sells: float = 0.0008
    base_slippage_bps: float = 2.0
    half_spread_bps_etf: float = 1.0
    half_spread_bps_equity: float = 2.5
    max_participation_rate: float = 0.05
    min_trade_notional: float = 25.0
    fractional_shares: bool = False
    regular_session_only: bool = True
    allow_short: bool = False
    allow_leverage: bool = False
    account_type: str = "to_be_confirmed"
    broker_rule_uncertainty: bool = True
    broker: str = "generic"
    ibkr_pro_pricing: str = "none"
    ibkr_tiered_exchange_fee_per_share: float = 0.0
    ibkr_tiered_include_clearing_fee: bool = True
    ibkr_tiered_include_pass_through_fee: bool = True
    ibkr_include_regulatory_fees: bool = True
    rebalance_only_on_weight_change: bool = False
    rebalance_tolerance: float = 1e-6
    margin_interest_rate_annual: float = 0.0
    margin_interest_day_count: int = 360
    short_borrow_fee_rate_annual: float = 0.0
    short_borrow_day_count: int = 360
    default_order_type: str = "market"
    limit_offset_bps: float = 5.0
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None
    stop_take_exit_order_type: str = "market"


def ibkr_pro_stock_commission(
    shares: float,
    trade_value: float,
    side: str,
    pricing: str,
    monthly_shares_before_trade: float = 0.0,
    exchange_fee_per_share: float = 0.0,
    include_clearing_fee: bool = True,
    include_pass_through_fee: bool = True,
    include_regulatory_fees: bool = True,
) -> dict[str, float]:
    """Approximate IBKR Pro US stock/ETF commission schedule.

    Uses Interactive Brokers published US stock/ETF schedule:
    - Fixed: USD 0.005/share, USD 1.00 min/order, 1% trade-value cap.
    - Tiered: marginal USD 0.0035 to 0.0005/share, USD 0.35 min/order,
      1% trade-value cap, plus third-party fees.

    Exchange venue fees/rebates are execution-venue dependent; the default is 0
    unless explicitly set in config.
    """
    shares = float(max(shares, 0.0))
    trade_value = float(max(trade_value, 0.0))
    if shares <= 0 or trade_value <= 0 or pricing in {"none", "", None}:
        return {
            "commission": 0.0,
            "regulatory_fee": 0.0,
            "clearing_fee": 0.0,
            "exchange_fee": 0.0,
            "pass_through_fee": 0.0,
            "total": 0.0,
        }

    cap_1pct = trade_value * 0.01
    pricing = pricing.lower()
    if pricing == "fixed":
        base_commission = min(max(shares * 0.005, 1.00), cap_1pct)
        clearing_fee = 0.0
        exchange_fee = 0.0
        pass_through_fee = 0.0
    elif pricing == "tiered":
        remaining = shares
        monthly = monthly_shares_before_trade
        tiers = [
            (300_000, 0.0035),
            (3_000_000, 0.0020),
            (20_000_000, 0.0015),
            (100_000_000, 0.0010),
            (float("inf"), 0.0005),
        ]
        commission_raw = 0.0
        lower = 0.0
        for upper, rate in tiers:
            if monthly >= upper:
                lower = upper
                continue
            capacity = upper - max(monthly, lower)
            take = min(remaining, capacity)
            if take > 0:
                commission_raw += take * rate
                remaining -= take
                monthly += take
            lower = upper
            if remaining <= 0:
                break
        base_commission = min(max(commission_raw, 0.35), cap_1pct)
        clearing_fee = min(shares * 0.00020, trade_value * 0.005) if include_clearing_fee else 0.0
        exchange_fee = shares * exchange_fee_per_share
        pass_through_fee = (
            base_commission * (0.000175 + 0.00056)
            if include_pass_through_fee
            else 0.0
        )
    else:
        raise ValueError(f"Unsupported IBKR pricing mode: {pricing}")

    regulatory_fee = 0.0
    if include_regulatory_fees:
        cat_fee = shares * 0.000003
        regulatory_fee += cat_fee
        if side == "sell":
            sec_fee = trade_value * 0.0000206
            finra_taf = min(shares * 0.000195, 9.79)
            regulatory_fee += sec_fee + finra_taf

    total = base_commission + regulatory_fee + clearing_fee + exchange_fee + pass_through_fee
    return {
        "commission": float(base_commission),
        "regulatory_fee": float(regulatory_fee),
        "clearing_fee": float(clearing_fee),
        "exchange_fee": float(exchange_fee),
        "pass_through_fee": float(pass_through_fee),
        "total": float(total),
    }


def pivot_bars(bars: pd.DataFrame, field: str) -> pd.DataFrame:
    out = bars.pivot(index="date", columns="ticker", values=field).sort_index()
    out.index = pd.to_datetime(out.index)
    return out


def estimate_half_spread_bps(ticker: str, asset_type: str | None = None) -> float:
    if asset_type == "equity":
        return 2.5
    low_spread = {"SPY", "QQQ", "IWM", "DIA", "VOO", "IVV", "VTI", "BIL", "SHY", "IEF", "TLT", "GLD"}
    return 0.5 if ticker in low_spread else 1.5


def simulate_daily_target_weights(
    bars: pd.DataFrame,
    target_weights: pd.DataFrame,
    asset_types: dict[str, str] | None = None,
    config: ExecutionConfig | None = None,
) -> dict[str, Any]:
    """Realistic daily backtest with next-session open execution.

    Signals are assumed to be known after the previous close; execution happens at
    the next regular-session open. This deliberately cannot validate intraday
    stop/take-profit ordering.
    """
    if config is None:
        config = ExecutionConfig()
    asset_types = asset_types or {}
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["date"])
    open_px = pivot_bars(bars, "adj_open")
    close_px = pivot_bars(bars, "adj_close")
    high_field = "adj_high" if "adj_high" in bars.columns else "high"
    low_field = "adj_low" if "adj_low" in bars.columns else "low"
    high_px = pivot_bars(bars, high_field) if high_field in bars.columns else close_px
    low_px = pivot_bars(bars, low_field) if low_field in bars.columns else close_px
    volume = pivot_bars(bars, "volume").fillna(0)
    dates = close_px.index
    previous_close = close_px.shift(1)
    target_weights = target_weights.reindex(dates).ffill().fillna(0.0)
    signal_weights = target_weights.shift(1).fillna(0.0)
    all_tickers = list(close_px.columns)

    cash = float(config.initial_capital)
    shares = pd.Series(0.0, index=all_tickers)
    lots: dict[str, list[dict[str, Any]]] = {ticker: [] for ticker in all_tickers}
    short_lots: dict[str, list[dict[str, Any]]] = {ticker: [] for ticker in all_tickers}
    equity_rows = []
    exposure_rows = []
    financing_rows = []
    order_rows = []
    audit_rows = []
    closed_trade_rows = []
    order_id = 0
    monthly_share_volume: dict[tuple[int, int], float] = {}
    previous_signal = pd.Series(0.0, index=all_tickers)

    for date in dates:
        day_open = open_px.loc[date].dropna()
        day_close = close_px.loc[date].dropna()
        day_high = high_px.loc[date].dropna() if date in high_px.index else day_close
        day_low = low_px.loc[date].dropna() if date in low_px.index else day_close
        tradable = day_open.index.intersection(day_close.index)
        if len(tradable) == 0:
            continue

        open_value = cash + float((shares.reindex(tradable).fillna(0) * day_open).sum())
        raw_desired = signal_weights.loc[date].reindex(tradable).fillna(0)
        desired = raw_desired if config.allow_short else raw_desired.clip(lower=0)
        desired_gross = desired.abs().sum() if config.allow_short else desired.sum()
        if not config.allow_leverage and desired_gross > 1:
            desired = desired / desired_gross

        raw_desired_all = signal_weights.loc[date].reindex(all_tickers).fillna(0)
        desired_all = raw_desired_all if config.allow_short else raw_desired_all.clip(lower=0)
        signal_changed = bool(
            (desired_all - previous_signal).abs().max() > config.rebalance_tolerance
        )
        if config.rebalance_only_on_weight_change and not signal_changed:
            close_value_positions = shares.reindex(day_close.index).fillna(0) * day_close
            margin_interest = 0.0
            short_borrow_fee = 0.0
            if cash < 0 and config.margin_interest_rate_annual > 0:
                margin_interest = abs(cash) * config.margin_interest_rate_annual / config.margin_interest_day_count
                cash -= margin_interest
            short_market_value = float(abs((shares.reindex(day_close.index).fillna(0).clip(upper=0) * day_close).sum()))
            if short_market_value > 0 and config.short_borrow_fee_rate_annual > 0:
                short_borrow_fee = short_market_value * config.short_borrow_fee_rate_annual / config.short_borrow_day_count
                cash -= short_borrow_fee
            close_equity = cash + float(close_value_positions.sum())
            gross_exposure = float(close_value_positions.abs().sum() / close_equity) if close_equity > 0 else 0.0
            equity_rows.append(
                {
                    "date": date,
                    "equity": close_equity,
                    "cash": cash,
                    "margin_interest": margin_interest,
                    "short_borrow_fee": short_borrow_fee,
                    "short_market_value": short_market_value,
                }
            )
            exposure_rows.append({"date": date, "exposure": gross_exposure})
            financing_rows.append(
                {
                    "date": date,
                    "margin_interest": margin_interest,
                    "short_borrow_fee": short_borrow_fee,
                    "short_market_value": short_market_value,
                    "cash": cash,
                }
            )
            continue
        previous_signal = desired_all.copy()

        current_value = shares.reindex(tradable).fillna(0) * day_open
        target_value = desired * open_value
        delta_value = target_value - current_value

        # Sell first, then buy, so long-only cash constraints are conservative.
        for side in ["sell", "buy"]:
            side_deltas = delta_value[delta_value < 0] if side == "sell" else delta_value[delta_value > 0]
            for ticker, delta in side_deltas.items():
                px = float(day_open[ticker])
                if px <= 0 or np.isnan(px):
                    continue
                raw_qty = abs(delta) / px
                qty = raw_qty if config.fractional_shares else np.floor(raw_qty)
                if qty <= 0:
                    continue
                notional = qty * px
                if notional < config.min_trade_notional:
                    continue
                day_volume = float(volume.loc[date].get(ticker, 0.0))
                dollar_volume = max(day_volume * px, 1.0)
                max_notional = dollar_volume * config.max_participation_rate
                liquidity_ok = notional <= max_notional
                order_type = str(config.default_order_type).lower()
                if order_type not in {"market", "limit"}:
                    raise ValueError(f"Unsupported order type: {config.default_order_type}")
                reference_px = float(previous_close.loc[date].get(ticker, np.nan)) if date in previous_close.index else np.nan
                if not np.isfinite(reference_px) or reference_px <= 0:
                    reference_px = px
                limit_price = np.nan
                half_spread_bps = estimate_half_spread_bps(ticker, asset_types.get(ticker))
                if order_type == "limit":
                    if side == "buy":
                        limit_price = reference_px * (1 + config.limit_offset_bps / 10000)
                        fillable = float(day_low.get(ticker, px)) <= limit_price
                    else:
                        limit_price = reference_px * (1 - config.limit_offset_bps / 10000)
                        fillable = float(day_high.get(ticker, px)) >= limit_price
                    if not fillable:
                        order_id += 1
                        order_rows.append(
                            {
                                "order_id": f"O{order_id:08d}",
                                "signal_time": date - pd.tseries.offsets.BDay(1),
                                "submit_time": date,
                                "execution_window": "regular_open",
                                "ticker": ticker,
                                "side": side,
                                "order_type": "limit",
                                "target_qty": qty,
                                "filled_qty": 0.0,
                                "unfilled_qty": qty,
                                "expected_price": px,
                                "actual_fill_price": np.nan,
                                "limit_price": limit_price,
                                "stop_price": np.nan,
                                "take_profit_price": np.nan,
                                "trailing_stop": np.nan,
                                "slippage": np.nan,
                                "spread_cost_bps": half_spread_bps,
                                "commission": 0.0,
                                "regulatory_fee": 0.0,
                                "clearing_fee": 0.0,
                                "exchange_fee": 0.0,
                                "pass_through_fee": 0.0,
                                "total_fees": 0.0,
                                "notional": 0.0,
                                "liquidity_check": bool(liquidity_ok),
                                "halt_check": "proxy_no_halt_feed",
                                "LULD_check": "proxy_no_luld_feed",
                                "fill_status": "not_filled",
                                "block_reason": "limit_not_touched",
                            }
                        )
                        audit_rows.append(
                            {
                                "date": date,
                                "ticker": ticker,
                                "side": side,
                                "participation": 0.0,
                                "dollar_volume": dollar_volume,
                                "max_participation_rate": config.max_participation_rate,
                                "cost_bps": np.nan,
                                "commission": 0.0,
                                "regulatory_fee": 0.0,
                                "clearing_fee": 0.0,
                                "exchange_fee": 0.0,
                                "pass_through_fee": 0.0,
                                "total_fees": 0.0,
                                "fill_status": "not_filled",
                                "block_reason": "limit_not_touched",
                            }
                        )
                        continue
                filled_qty = qty
                fill_status = "filled"
                block_reason = ""
                if not liquidity_ok:
                    filled_qty = max_notional / px if config.fractional_shares else np.floor(max_notional / px)
                    fill_status = "partial_fill" if filled_qty > 0 else "blocked"
                    block_reason = "max_participation_rate"
                if side == "sell" and not config.allow_short:
                    filled_qty = min(filled_qty, shares.get(ticker, 0.0))
                if filled_qty <= 0:
                    continue

                participation = (filled_qty * px) / dollar_volume
                impact_bps = config.base_slippage_bps + max(0.0, participation / config.max_participation_rate) * 2.0
                cost_bps = impact_bps + half_spread_bps
                marketable_fill_price = px * (1 + cost_bps / 10000) if side == "buy" else px * (1 - cost_bps / 10000)
                if order_type == "limit":
                    fill_price = min(marketable_fill_price, limit_price) if side == "buy" else max(marketable_fill_price, limit_price)
                else:
                    fill_price = marketable_fill_price
                month_key = (int(date.year), int(date.month))
                broker_cost = ibkr_pro_stock_commission(
                    shares=filled_qty,
                    trade_value=filled_qty * fill_price,
                    side=side,
                    pricing=config.ibkr_pro_pricing if config.broker == "IBKR_PRO" else "none",
                    monthly_shares_before_trade=monthly_share_volume.get(month_key, 0.0),
                    exchange_fee_per_share=config.ibkr_tiered_exchange_fee_per_share,
                    include_clearing_fee=config.ibkr_tiered_include_clearing_fee,
                    include_pass_through_fee=config.ibkr_tiered_include_pass_through_fee,
                    include_regulatory_fees=config.ibkr_include_regulatory_fees,
                )
                if config.broker == "IBKR_PRO":
                    commission = broker_cost["commission"]
                    regulatory_fee = broker_cost["regulatory_fee"]
                    clearing_fee = broker_cost["clearing_fee"]
                    exchange_fee = broker_cost["exchange_fee"]
                    pass_through_fee = broker_cost["pass_through_fee"]
                else:
                    commission = filled_qty * config.commission_per_share
                    regulatory_fee = (
                        filled_qty * fill_price * config.sec_fee_bps_on_sells / 10000
                        if side == "sell"
                        else 0.0
                    )
                    clearing_fee = 0.0
                    exchange_fee = 0.0
                    pass_through_fee = 0.0
                actual_notional = filled_qty * fill_price

                if side == "buy":
                    affordable_qty = filled_qty
                    total_buy_cost = commission + regulatory_fee + clearing_fee + exchange_fee + pass_through_fee
                    if not config.allow_leverage and actual_notional + total_buy_cost > cash:
                        affordable_qty = max(cash - total_buy_cost, 0) / fill_price
                        if not config.fractional_shares:
                            affordable_qty = np.floor(affordable_qty)
                        fill_status = "partial_fill" if affordable_qty > 0 else "blocked"
                        block_reason = "cash_constraint"
                    filled_qty = affordable_qty
                    actual_notional = filled_qty * fill_price
                    if filled_qty <= 0:
                        continue
                    if affordable_qty != qty and config.broker == "IBKR_PRO":
                        broker_cost = ibkr_pro_stock_commission(
                            shares=filled_qty,
                            trade_value=actual_notional,
                            side=side,
                            pricing=config.ibkr_pro_pricing,
                            monthly_shares_before_trade=monthly_share_volume.get(month_key, 0.0),
                            exchange_fee_per_share=config.ibkr_tiered_exchange_fee_per_share,
                            include_clearing_fee=config.ibkr_tiered_include_clearing_fee,
                            include_pass_through_fee=config.ibkr_tiered_include_pass_through_fee,
                            include_regulatory_fees=config.ibkr_include_regulatory_fees,
                        )
                        commission = broker_cost["commission"]
                        regulatory_fee = broker_cost["regulatory_fee"]
                        clearing_fee = broker_cost["clearing_fee"]
                        exchange_fee = broker_cost["exchange_fee"]
                        pass_through_fee = broker_cost["pass_through_fee"]
                    total_fees = commission + regulatory_fee + clearing_fee + exchange_fee + pass_through_fee
                    cash -= actual_notional + total_fees
                    prior_short_qty = abs(min(float(shares.get(ticker, 0.0)), 0.0))
                    cover_qty = min(filled_qty, prior_short_qty)
                    opening_long_qty = filled_qty - cover_qty
                    remaining_cover = cover_qty
                    while remaining_cover > 0 and short_lots[ticker]:
                        lot = short_lots[ticker][0]
                        close_qty = min(remaining_cover, lot["remaining_qty"])
                        pnl = (lot["entry_price"] - fill_price) * close_qty
                        fee_alloc = total_fees * (close_qty / filled_qty) if filled_qty else 0.0
                        closed_trade_rows.append(
                            {
                                "ticker": ticker,
                                "entry_time": lot["entry_time"],
                                "exit_time": date,
                                "entry_price": lot["entry_price"],
                                "exit_price": fill_price,
                                "qty": close_qty,
                                "pnl": pnl - fee_alloc,
                                "holding_days": int((date - lot["entry_time"]).days),
                                "day_trade_flag": bool(date == lot["entry_time"]),
                                "exit_reason": "short_rebalance_or_target_exit",
                                "position_side": "short",
                            }
                        )
                        lot["remaining_qty"] -= close_qty
                        remaining_cover -= close_qty
                        if lot["remaining_qty"] <= 1e-9:
                            short_lots[ticker].pop(0)
                    shares[ticker] = shares.get(ticker, 0.0) + filled_qty
                    if opening_long_qty > 1e-9:
                        fee_alloc = total_fees * (opening_long_qty / filled_qty) if filled_qty else 0.0
                        lots[ticker].append(
                            {
                                "entry_time": date,
                                "entry_price": fill_price + fee_alloc / opening_long_qty,
                                "qty": opening_long_qty,
                                "remaining_qty": opening_long_qty,
                                "highest_price": fill_price,
                            }
                        )
                else:
                    total_fees = commission + regulatory_fee + clearing_fee + exchange_fee + pass_through_fee
                    cash += actual_notional - total_fees
                    prior_long_qty = max(float(shares.get(ticker, 0.0)), 0.0)
                    close_long_qty = min(filled_qty, prior_long_qty)
                    opening_short_qty = filled_qty - close_long_qty
                    shares[ticker] = shares.get(ticker, 0.0) - filled_qty
                    remaining = close_long_qty
                    while remaining > 0 and lots[ticker]:
                        lot = lots[ticker][0]
                        close_qty = min(remaining, lot["remaining_qty"])
                        pnl = (fill_price - lot["entry_price"]) * close_qty
                        fee_alloc = total_fees * (close_qty / filled_qty) if filled_qty else 0.0
                        closed_trade_rows.append(
                            {
                                "ticker": ticker,
                                "entry_time": lot["entry_time"],
                                "exit_time": date,
                                "entry_price": lot["entry_price"],
                                "exit_price": fill_price,
                                "qty": close_qty,
                                "pnl": pnl - fee_alloc,
                                "holding_days": int((date - lot["entry_time"]).days),
                                "day_trade_flag": bool(date == lot["entry_time"]),
                                "exit_reason": "rebalance_or_target_exit",
                                "position_side": "long",
                            }
                        )
                        lot["remaining_qty"] -= close_qty
                        remaining -= close_qty
                        if lot["remaining_qty"] <= 1e-9:
                            lots[ticker].pop(0)
                    if opening_short_qty > 1e-9:
                        fee_alloc = total_fees * (opening_short_qty / filled_qty) if filled_qty else 0.0
                        short_lots[ticker].append(
                            {
                                "entry_time": date,
                                "entry_price": fill_price - fee_alloc / opening_short_qty,
                                "qty": opening_short_qty,
                                "remaining_qty": opening_short_qty,
                                "lowest_price": fill_price,
                            }
                        )

                order_id += 1
                monthly_share_volume[month_key] = monthly_share_volume.get(month_key, 0.0) + filled_qty
                order_rows.append(
                    {
                        "order_id": f"O{order_id:08d}",
                        "signal_time": date - pd.tseries.offsets.BDay(1),
                        "submit_time": date,
                        "execution_window": "regular_open",
                        "ticker": ticker,
                        "side": side,
                        "order_type": order_type,
                        "target_qty": qty,
                        "filled_qty": filled_qty,
                        "unfilled_qty": max(qty - filled_qty, 0),
                        "expected_price": px,
                        "actual_fill_price": fill_price,
                        "limit_price": limit_price,
                        "stop_price": np.nan,
                        "take_profit_price": np.nan,
                        "trailing_stop": np.nan,
                        "slippage": fill_price - px,
                        "spread_cost_bps": half_spread_bps,
                        "commission": commission,
                        "regulatory_fee": regulatory_fee,
                        "clearing_fee": clearing_fee,
                        "exchange_fee": exchange_fee,
                        "pass_through_fee": pass_through_fee,
                        "total_fees": commission + regulatory_fee + clearing_fee + exchange_fee + pass_through_fee,
                        "notional": actual_notional if side == "buy" else -actual_notional,
                        "position_effect": (
                            "cover_or_buy"
                            if side == "buy" and config.allow_short
                            else "sell_or_short"
                            if side == "sell" and config.allow_short
                            else ""
                        ),
                        "liquidity_check": bool(liquidity_ok),
                        "halt_check": "proxy_no_halt_feed",
                        "LULD_check": "proxy_no_luld_feed",
                        "fill_status": fill_status,
                        "block_reason": block_reason,
                    }
                )
                audit_rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "side": side,
                        "participation": participation,
                        "dollar_volume": dollar_volume,
                        "max_participation_rate": config.max_participation_rate,
                        "cost_bps": cost_bps,
                        "commission": commission,
                        "regulatory_fee": regulatory_fee,
                        "clearing_fee": clearing_fee,
                        "exchange_fee": exchange_fee,
                        "pass_through_fee": pass_through_fee,
                        "total_fees": commission + regulatory_fee + clearing_fee + exchange_fee + pass_through_fee,
                        "fill_status": fill_status,
                        "block_reason": block_reason,
                    }
                )

        if config.stop_loss_pct is not None or config.take_profit_pct is not None or config.trailing_stop_pct is not None:
            for ticker in list(lots.keys()):
                if shares.get(ticker, 0.0) <= 0 or ticker not in tradable:
                    continue
                px_open = float(day_open.get(ticker, np.nan))
                px_high = float(day_high.get(ticker, np.nan))
                px_low = float(day_low.get(ticker, np.nan))
                if not np.isfinite(px_open) or not np.isfinite(px_high) or not np.isfinite(px_low):
                    continue
                lot_index = 0
                while lot_index < len(lots[ticker]):
                    lot = lots[ticker][lot_index]
                    lot["highest_price"] = max(float(lot.get("highest_price", lot["entry_price"])), px_high)
                    fixed_stop_price = (
                        float(lot["entry_price"]) * (1.0 - float(config.stop_loss_pct))
                        if config.stop_loss_pct is not None
                        else np.nan
                    )
                    trailing_stop_price = (
                        float(lot["highest_price"]) * (1.0 - float(config.trailing_stop_pct))
                        if config.trailing_stop_pct is not None
                        else np.nan
                    )
                    stop_candidates = [price for price in [fixed_stop_price, trailing_stop_price] if np.isfinite(price)]
                    stop_price = max(stop_candidates) if stop_candidates else np.nan
                    take_profit_price = (
                        float(lot["entry_price"]) * (1.0 + float(config.take_profit_pct))
                        if config.take_profit_pct is not None
                        else np.nan
                    )
                    stop_hit = np.isfinite(stop_price) and px_low <= stop_price
                    take_hit = np.isfinite(take_profit_price) and px_high >= take_profit_price
                    if not stop_hit and not take_hit:
                        lot_index += 1
                        continue

                    # Daily bars do not reveal intraday ordering; use stop/trailing first when both touch.
                    exit_reason = "take_profit"
                    if stop_hit:
                        exit_reason = (
                            "trailing_stop"
                            if np.isfinite(trailing_stop_price)
                            and (not np.isfinite(fixed_stop_price) or trailing_stop_price >= fixed_stop_price)
                            else "stop_loss"
                        )
                    trigger_price = stop_price if stop_hit else take_profit_price
                    qty = float(lot["remaining_qty"])
                    if qty <= 0:
                        lot_index += 1
                        continue
                    day_volume = float(volume.loc[date].get(ticker, 0.0))
                    dollar_volume = max(day_volume * px_open, 1.0)
                    max_notional = dollar_volume * config.max_participation_rate
                    requested_notional = qty * trigger_price
                    liquidity_ok = requested_notional <= max_notional
                    filled_qty = qty
                    fill_status = "filled"
                    block_reason = exit_reason
                    if not liquidity_ok:
                        filled_qty = max_notional / trigger_price if config.fractional_shares else np.floor(max_notional / trigger_price)
                        filled_qty = min(filled_qty, qty)
                        fill_status = "partial_fill" if filled_qty > 0 else "blocked"
                        block_reason = f"{exit_reason}_max_participation_rate"
                    if filled_qty <= 0:
                        lot_index += 1
                        continue

                    participation = (filled_qty * trigger_price) / dollar_volume
                    half_spread_bps = estimate_half_spread_bps(ticker, asset_types.get(ticker))
                    impact_bps = config.base_slippage_bps + max(0.0, participation / config.max_participation_rate) * 2.0
                    cost_bps = impact_bps + half_spread_bps
                    fill_price = trigger_price * (1 - cost_bps / 10000)
                    month_key = (int(date.year), int(date.month))
                    broker_cost = ibkr_pro_stock_commission(
                        shares=filled_qty,
                        trade_value=filled_qty * fill_price,
                        side="sell",
                        pricing=config.ibkr_pro_pricing if config.broker == "IBKR_PRO" else "none",
                        monthly_shares_before_trade=monthly_share_volume.get(month_key, 0.0),
                        exchange_fee_per_share=config.ibkr_tiered_exchange_fee_per_share,
                        include_clearing_fee=config.ibkr_tiered_include_clearing_fee,
                        include_pass_through_fee=config.ibkr_tiered_include_pass_through_fee,
                        include_regulatory_fees=config.ibkr_include_regulatory_fees,
                    )
                    if config.broker == "IBKR_PRO":
                        commission = broker_cost["commission"]
                        regulatory_fee = broker_cost["regulatory_fee"]
                        clearing_fee = broker_cost["clearing_fee"]
                        exchange_fee = broker_cost["exchange_fee"]
                        pass_through_fee = broker_cost["pass_through_fee"]
                    else:
                        commission = filled_qty * config.commission_per_share
                        regulatory_fee = filled_qty * fill_price * config.sec_fee_bps_on_sells / 10000
                        clearing_fee = 0.0
                        exchange_fee = 0.0
                        pass_through_fee = 0.0
                    total_fees = commission + regulatory_fee + clearing_fee + exchange_fee + pass_through_fee
                    actual_notional = filled_qty * fill_price
                    cash += actual_notional - total_fees
                    shares[ticker] = shares.get(ticker, 0.0) - filled_qty
                    pnl = (fill_price - lot["entry_price"]) * filled_qty
                    closed_trade_rows.append(
                        {
                            "ticker": ticker,
                            "entry_time": lot["entry_time"],
                            "exit_time": date,
                            "entry_price": lot["entry_price"],
                            "exit_price": fill_price,
                            "qty": filled_qty,
                            "pnl": pnl - total_fees,
                            "holding_days": int((date - lot["entry_time"]).days),
                            "day_trade_flag": bool(date == lot["entry_time"]),
                            "exit_reason": exit_reason,
                        }
                    )

                    order_id += 1
                    monthly_share_volume[month_key] = monthly_share_volume.get(month_key, 0.0) + filled_qty
                    order_rows.append(
                        {
                            "order_id": f"O{order_id:08d}",
                            "signal_time": date - pd.tseries.offsets.BDay(1),
                            "submit_time": date,
                            "execution_window": "regular_session_stop_take_proxy",
                            "ticker": ticker,
                            "side": "sell",
                            "order_type": str(config.stop_take_exit_order_type).lower(),
                            "target_qty": qty,
                            "filled_qty": filled_qty,
                            "unfilled_qty": max(qty - filled_qty, 0),
                            "expected_price": trigger_price,
                            "actual_fill_price": fill_price,
                            "limit_price": take_profit_price if exit_reason == "take_profit" else np.nan,
                            "stop_price": stop_price if exit_reason in {"stop_loss", "trailing_stop"} else np.nan,
                            "take_profit_price": take_profit_price if exit_reason == "take_profit" else np.nan,
                            "trailing_stop": trailing_stop_price if np.isfinite(trailing_stop_price) else np.nan,
                            "slippage": fill_price - trigger_price,
                            "spread_cost_bps": half_spread_bps,
                            "commission": commission,
                            "regulatory_fee": regulatory_fee,
                            "clearing_fee": clearing_fee,
                            "exchange_fee": exchange_fee,
                            "pass_through_fee": pass_through_fee,
                            "total_fees": total_fees,
                            "notional": -actual_notional,
                            "liquidity_check": bool(liquidity_ok),
                            "halt_check": "proxy_no_halt_feed",
                            "LULD_check": "proxy_no_luld_feed",
                            "fill_status": fill_status,
                            "block_reason": block_reason,
                        }
                    )
                    audit_rows.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "side": "sell",
                            "participation": participation,
                            "dollar_volume": dollar_volume,
                            "max_participation_rate": config.max_participation_rate,
                            "cost_bps": cost_bps,
                            "commission": commission,
                            "regulatory_fee": regulatory_fee,
                            "clearing_fee": clearing_fee,
                            "exchange_fee": exchange_fee,
                            "pass_through_fee": pass_through_fee,
                            "total_fees": total_fees,
                            "fill_status": fill_status,
                            "block_reason": block_reason,
                        }
                    )
                    lot["remaining_qty"] -= filled_qty
                    if lot["remaining_qty"] <= 1e-9:
                        lots[ticker].pop(lot_index)
                    else:
                        lot_index += 1

        close_value_positions = shares.reindex(day_close.index).fillna(0) * day_close
        margin_interest = 0.0
        short_borrow_fee = 0.0
        if cash < 0 and config.margin_interest_rate_annual > 0:
            margin_interest = abs(cash) * config.margin_interest_rate_annual / config.margin_interest_day_count
            cash -= margin_interest
        short_market_value = float(abs((shares.reindex(day_close.index).fillna(0).clip(upper=0) * day_close).sum()))
        if short_market_value > 0 and config.short_borrow_fee_rate_annual > 0:
            short_borrow_fee = short_market_value * config.short_borrow_fee_rate_annual / config.short_borrow_day_count
            cash -= short_borrow_fee
        close_equity = cash + float(close_value_positions.sum())
        gross_exposure = float(close_value_positions.abs().sum() / close_equity) if close_equity > 0 else 0.0
        equity_rows.append(
            {
                "date": date,
                "equity": close_equity,
                "cash": cash,
                "margin_interest": margin_interest,
                "short_borrow_fee": short_borrow_fee,
                "short_market_value": short_market_value,
            }
        )
        exposure_rows.append({"date": date, "exposure": gross_exposure})
        financing_rows.append(
            {
                "date": date,
                "margin_interest": margin_interest,
                "short_borrow_fee": short_borrow_fee,
                "short_market_value": short_market_value,
                "cash": cash,
            }
        )

    equity = pd.DataFrame(equity_rows).set_index("date")["equity"] if equity_rows else pd.Series(dtype=float)
    cash_curve = pd.DataFrame(equity_rows).set_index("date")["cash"] if equity_rows else pd.Series(dtype=float)
    exposure = pd.DataFrame(exposure_rows).set_index("date")["exposure"] if exposure_rows else pd.Series(dtype=float)
    orders = pd.DataFrame(order_rows)
    audit = pd.DataFrame(audit_rows)
    closed_trades = pd.DataFrame(closed_trade_rows)
    financing = pd.DataFrame(financing_rows).set_index("date") if financing_rows else pd.DataFrame()
    return {
        "equity": equity,
        "cash": cash_curve,
        "exposure": exposure,
        "orders": orders,
        "execution_audit": audit,
        "closed_trades": closed_trades,
        "financing": financing,
    }
