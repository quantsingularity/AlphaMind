"""
Strategy service — CRUD for trading strategies and backtest lifecycle.

Strategy parameters are stored as a JSON blob in the DB.  Performance
metrics are either supplied at creation or refreshed by an async compute
task (not yet wired to a live execution engine — see TODO below).
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from db.repositories.backtest_repository import BacktestRepository
from db.repositories.strategy_repository import StrategyRepository
from sqlalchemy.ext.asyncio import AsyncSession


class StrategyService:
    """Manages trading strategy records and their backtest results."""

    def __init__(self, session: AsyncSession) -> None:
        self._strategy_repo = StrategyRepository(session)
        self._backtest_repo = BacktestRepository(session)

    # ------------------------------------------------------------------
    # Strategy CRUD
    # ------------------------------------------------------------------

    async def list_strategies(self) -> List[Dict[str, Any]]:
        strategies = await self._strategy_repo.list(limit=200)
        return [self._strategy_to_dict(s) for s in strategies]

    async def get_strategy(self, strategy_id: str) -> Dict[str, Any] | None:
        s = await self._strategy_repo.get(strategy_id)
        return self._strategy_to_dict(s) if s else None

    async def get_performance(self, strategy_id: str) -> Dict[str, Any] | None:
        """Return just the performance metrics block for a strategy."""
        s = await self._strategy_repo.get(strategy_id)
        if not s:
            return None
        return self._strategy_to_dict(s)["performance"]

    async def get_equity_curve(
        self, strategy_id: str, days: int = 90
    ) -> Dict[str, Any] | None:
        """Build a deterministic equity curve for a strategy.

        The curve is derived from the strategy's stored performance metrics
        (total return / volatility) using a per-strategy seed so the result is
        stable across requests. Returns the shape the web client expects:
        ``{"strategyId": ..., "equityCurve": [{"day", "value", "benchmark"}]}``.
        """
        s = await self._strategy_repo.get(strategy_id)
        if not s:
            return None

        total_return = float(getattr(s, "total_return", 0.0) or 0.0)
        volatility = float(getattr(s, "volatility", 0.1) or 0.1)
        beta = float(getattr(s, "beta", 1.0) or 1.0)

        rng = random.Random(strategy_id)
        # Per-period drift so the curve ends near the stored total return.
        drift = total_return / days if days else 0.0
        daily_vol = volatility / (days**0.5) if days else 0.0
        # Benchmark grows at a flat market rate, scaled down by the strategy beta.
        bench_drift = (0.08 / days) if days else 0.0

        value = 100.0
        benchmark = 100.0
        curve: List[Dict[str, Any]] = [
            {"day": 0, "value": round(value, 2), "benchmark": round(benchmark, 2)}
        ]
        for day in range(1, days + 1):
            value *= 1 + drift + rng.gauss(0, daily_vol)
            benchmark *= (
                1 + bench_drift + rng.gauss(0, daily_vol * 0.6 / max(beta, 0.1))
            )
            curve.append(
                {
                    "day": day,
                    "value": round(value, 2),
                    "benchmark": round(benchmark, 2),
                }
            )
        return {"strategyId": strategy_id, "equityCurve": curve}

    async def create_strategy(
        self,
        name: str,
        description: str,
        strategy_type: str,
        parameters: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        strategy = await self._strategy_repo.create(
            id=f"strat-{uuid.uuid4().hex[:8]}",
            name=name,
            description=description,
            strategy_type=strategy_type,
            parameters_json=json.dumps(parameters or {}),
            status="paused",
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            profit_factor=0.0,
            win_rate=0.0,
            total_return=0.0,
            volatility=0.0,
            alpha=0.0,
            beta=1.0,
        )
        return self._strategy_to_dict(strategy)

    async def update_strategy(
        self, strategy_id: str, **kwargs: Any
    ) -> Dict[str, Any] | None:
        if "parameters" in kwargs:
            kwargs["parameters_json"] = json.dumps(kwargs.pop("parameters"))
        if "type" in kwargs:
            kwargs["strategy_type"] = kwargs.pop("type")
        kwargs["updated_at"] = datetime.now(timezone.utc)
        updated = await self._strategy_repo.update(strategy_id, **kwargs)
        return self._strategy_to_dict(updated) if updated else None

    async def delete_strategy(self, strategy_id: str) -> bool:
        return await self._strategy_repo.delete(strategy_id)

    async def activate_strategy(self, strategy_id: str) -> Dict[str, Any] | None:
        return await self.update_strategy(strategy_id, status="active")

    async def deactivate_strategy(self, strategy_id: str) -> Dict[str, Any] | None:
        return await self.update_strategy(strategy_id, status="paused")

    # ------------------------------------------------------------------
    # Backtest lifecycle
    # ------------------------------------------------------------------

    async def create_backtest(
        self,
        strategy_id: str,
        start_date: str,
        end_date: str,
        initial_capital: float,
    ) -> Dict[str, Any]:
        """
        Enqueue a new backtest run with status 'pending'.

        A background worker (e.g. Celery / Ray task) should pick this up,
        run the simulation, and update the record to 'completed'.
        """
        run = await self._backtest_repo.create(
            id=f"bt-{uuid.uuid4().hex[:8]}",
            strategy_id=strategy_id,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            status="pending",
        )
        return self._backtest_to_dict(run)

    async def get_backtests(self, strategy_id: str) -> List[Dict[str, Any]]:
        runs = await self._backtest_repo.get_by_strategy(strategy_id)
        return [self._backtest_to_dict(r) for r in runs]

    async def get_backtest(self, backtest_id: str) -> Dict[str, Any] | None:
        run = await self._backtest_repo.get(backtest_id)
        return self._backtest_to_dict(run) if run else None

    async def complete_backtest(
        self,
        backtest_id: str,
        final_capital: float,
        total_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        total_trades: int,
        win_rate: float,
    ) -> Dict[str, Any] | None:
        """Mark a backtest as completed with its results."""
        updated = await self._backtest_repo.update(
            backtest_id,
            status="completed",
            final_capital=final_capital,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            win_rate=win_rate,
            completed_at=datetime.now(timezone.utc),
        )
        return self._backtest_to_dict(updated) if updated else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strategy_to_dict(s: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        try:
            params = json.loads(s.parameters_json or "{}")
        except (json.JSONDecodeError, TypeError):
            pass
        return {
            "id": s.id,
            "name": s.name,
            "description": s.description,
            "type": s.strategy_type,
            "status": s.status,
            "performance": {
                "sharpeRatio": s.sharpe_ratio,
                "maxDrawdown": s.max_drawdown,
                "profitFactor": s.profit_factor,
                "winRate": s.win_rate,
                "totalReturn": s.total_return,
                "volatility": s.volatility,
                "alpha": s.alpha,
                "beta": s.beta,
            },
            "parameters": params,
            "createdAt": s.created_at.isoformat() if s.created_at else None,
            "updatedAt": s.updated_at.isoformat() if s.updated_at else None,
        }

    @staticmethod
    def _backtest_to_dict(r: Any) -> Dict[str, Any]:
        return {
            "id": r.id,
            "strategyId": r.strategy_id,
            "startDate": r.start_date,
            "endDate": r.end_date,
            "initialCapital": r.initial_capital,
            "finalCapital": r.final_capital,
            "totalReturn": r.total_return,
            "sharpeRatio": r.sharpe_ratio,
            "maxDrawdown": r.max_drawdown,
            "totalTrades": r.total_trades,
            "winRate": r.win_rate,
            "status": r.status,
            "createdAt": r.created_at.isoformat() if r.created_at else None,
            "completedAt": r.completed_at.isoformat() if r.completed_at else None,
        }
