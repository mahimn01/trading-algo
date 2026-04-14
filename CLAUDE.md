# Trading System

## Stack
Python 3.11+, IBKR TWS API, asyncio, click CLI

## Commands
python -m trading_algo --help    # Show all commands
python -m trading_algo --paper   # Paper trading (port 4002)
python -m trading_algo --live    # Live trading (port 4001)
pytest                           # Run tests

## Architecture
- Entry: trading_algo/cli.py
- Broker: trading_algo/broker/ibkr.py
- Config: trading_algo/config.py (TradingConfig with safety rails)
- Strategies: trading_algo/strategies/ (14 equity + 9 crypto)

## Rules
- ALWAYS use IBKR TWS API first for live data before WorldMonitor or web searches
- Paper port: 4002, Live port: 4001 — NEVER mix these up
- IB Gateway runs 24/7 via IBC in tmux, auto-restart at 23:55 ET
- All risk limits enforced at config level, not strategy level
- Type hints on ALL function signatures
- --paper and --live are mutually exclusive CLI flags

## Self-Improvement
After every bug fix or correction, add a rule here to prevent repeating it.
