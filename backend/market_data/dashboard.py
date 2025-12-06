"""
## Dashboard Module for AlphaMind

## This module provides a real-time dashboard for monitoring trading performance,
## market data, portfolio status, and system health metrics.
"""

from datetime import datetime, timedelta
import logging
from typing import Any, Dict, Optional
import dash
from dash import Input, Output, dcc, html
import dash_bootstrap_components as dbc
from flask import Flask
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DashboardMetrics:
    """Class for calculating and storing dashboard metrics."""

    def __init__(self) -> Any:
        """Initialize dashboard metrics."""
        self.portfolio_value_history = []
        self.cash_history = []
        self.equity_history = []
        self.positions = {}
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.daily_returns = []
        self.cumulative_returns = []
        self.drawdowns = []
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        self.var_95 = 0.0
        self.var_99 = 0.0
        self.expected_shortfall = 0.0
        self.beta = 0.0
        self.volatility = 0.0
        self.trades_history = []
        self.orders_history = []
        self.active_orders = []
        self.market_data = {}
        self.system_health = {
            "cpu_usage": [],
            "memory_usage": [],
            "latency": [],
            "errors": [],
        }

    def update_portfolio(
        self,
        timestamp: datetime,
        portfolio_value: float,
        cash: float,
        positions: Dict[str, Dict[str, Any]],
    ) -> Any:
        """
        Update portfolio metrics.

        Args:
            timestamp: Current timestamp
            portfolio_value: Total portfolio value
            cash: Cash balance
            positions: Dictionary of positions
        """
        self.portfolio_value_history.append(
            {"timestamp": timestamp, "value": portfolio_value}
        )
        self.cash_history.append({"timestamp": timestamp, "value": cash})
        self.equity_history.append(
            {"timestamp": timestamp, "value": portfolio_value - cash}
        )
        self.positions = positions
        self.unrealized_pnl = sum(
            (pos.get("unrealized_pnl", 0.0) for pos in positions.values())
        )
        self._calculate_performance_metrics()

    def update_trade(self, trade: Dict[str, Any]) -> Any:
        """
        Update trade metrics.

        Args:
            trade: Trade information
        """
        self.trades_history.append(trade)
        if "realized_pnl" in trade:
            self.realized_pnl += trade["realized_pnl"]
        winning_trades = sum(
            (1 for t in self.trades_history if t.get("realized_pnl", 0) > 0)
        )
        total_trades = len(self.trades_history)
        self.win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    def update_order(self, order: Dict[str, Any]) -> Any:
        """
        Update order metrics.

        Args:
            order: Order information
        """
        self.orders_history.append(order)
        self.active_orders = [
            o
            for o in self.orders_history
            if o.get("status") in ["pending", "open", "partially_filled"]
        ]

    def update_market_data(self, symbol: str, data: Dict[str, Any]) -> Any:
        """
        Update market data.

        Args:
            symbol: Trading symbol
            data: Market data
        """
        if symbol not in self.market_data:
            self.market_data[symbol] = []
        self.market_data[symbol].append(data)
        if len(self.market_data[symbol]) > 1000:
            self.market_data[symbol] = self.market_data[symbol][-1000:]

    def update_system_health(
        self, cpu_usage: float, memory_usage: float, latency: float, errors: int
    ) -> Any:
        """
        Update system health metrics.

        Args:
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage percentage
            latency: System latency in milliseconds
            errors: Number of errors
        """
        timestamp = datetime.now()
        self.system_health["cpu_usage"].append(
            {"timestamp": timestamp, "value": cpu_usage}
        )
        self.system_health["memory_usage"].append(
            {"timestamp": timestamp, "value": memory_usage}
        )
        self.system_health["latency"].append({"timestamp": timestamp, "value": latency})
        self.system_health["errors"].append({"timestamp": timestamp, "value": errors})
        for key in self.system_health:
            if len(self.system_health[key]) > 1000:
                self.system_health[key] = self.system_health[key][-1000:]

    def _calculate_performance_metrics(self) -> Any:
        """Calculate performance metrics from portfolio history."""
        if len(self.portfolio_value_history) < 2:
            return
        df = pd.DataFrame(self.portfolio_value_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        daily_values = df.resample("D").last()
        daily_returns = daily_values["value"].pct_change().dropna()
        self.daily_returns = [
            {"timestamp": timestamp, "value": value}
            for timestamp, value in zip(daily_returns.index, daily_returns.values)
        ]
        cumulative_returns = (1 + daily_returns).cumprod() - 1
        self.cumulative_returns = [
            {"timestamp": timestamp, "value": value}
            for timestamp, value in zip(
                cumulative_returns.index, cumulative_returns.values
            )
        ]
        rolling_max = daily_values["value"].cummax()
        drawdowns = (daily_values["value"] - rolling_max) / rolling_max
        self.drawdowns = [
            {"timestamp": timestamp, "value": value}
            for timestamp, value in zip(drawdowns.index, drawdowns.values)
        ]
        self.max_drawdown = drawdowns.min()
        if len(daily_returns) > 0:
            mean_return = daily_returns.mean()
            std_return = daily_returns.std()
            if std_return > 0:
                self.sharpe_ratio = mean_return / std_return * np.sqrt(252)
            negative_returns = daily_returns[daily_returns < 0]
            if len(negative_returns) > 0:
                downside_std = negative_returns.std()
                if downside_std > 0:
                    self.sortino_ratio = mean_return / downside_std * np.sqrt(252)
            self.volatility = std_return * np.sqrt(252)
            self.var_95 = np.percentile(daily_returns, 5)
            self.var_99 = np.percentile(daily_returns, 1)
            self.expected_shortfall = negative_returns.mean()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics.

        Returns:
            Dictionary with summary metrics
        """
        current_portfolio_value = (
            self.portfolio_value_history[-1]["value"]
            if self.portfolio_value_history
            else 0.0
        )
        current_cash = self.cash_history[-1]["value"] if self.cash_history else 0.0
        return {
            "portfolio": {
                "value": current_portfolio_value,
                "cash": current_cash,
                "equity": current_portfolio_value - current_cash,
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": self.unrealized_pnl,
                "total_pnl": self.realized_pnl + self.unrealized_pnl,
                "positions_count": len(self.positions),
                "positions": self.positions,
            },
            "performance": {
                "daily_return": (
                    self.daily_returns[-1]["value"] if self.daily_returns else 0.0
                ),
                "cumulative_return": (
                    self.cumulative_returns[-1]["value"]
                    if self.cumulative_returns
                    else 0.0
                ),
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "max_drawdown": self.max_drawdown,
                "volatility": self.volatility,
                "win_rate": self.win_rate,
            },
            "risk": {
                "var_95": self.var_95,
                "var_99": self.var_99,
                "expected_shortfall": self.expected_shortfall,
                "beta": self.beta,
            },
            "trading": {
                "total_trades": len(self.trades_history),
                "active_orders": len(self.active_orders),
            },
            "system": {
                "cpu_usage": (
                    self.system_health["cpu_usage"][-1]["value"]
                    if self.system_health["cpu_usage"]
                    else 0.0
                ),
                "memory_usage": (
                    self.system_health["memory_usage"][-1]["value"]
                    if self.system_health["memory_usage"]
                    else 0.0
                ),
                "latency": (
                    self.system_health["latency"][-1]["value"]
                    if self.system_health["latency"]
                    else 0.0
                ),
                "errors": (
                    sum((item["value"] for item in self.system_health["errors"][-100:]))
                    if self.system_health["errors"]
                    else 0
                ),
            },
        }


class DashboardServer:
    """Server for AlphaMind dashboard."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8050,
        debug: bool = False,
        metrics: Optional[DashboardMetrics] = None,
    ) -> Any:
        """
        Initialize dashboard server.

        Args:
            host: Server host
            port: Server port
            debug: Whether to enable debug mode
            metrics: Dashboard metrics instance
        """
        self.host = host
        self.port = port
        self.debug = debug
        self.metrics = metrics or DashboardMetrics()
        self.app = None
        self.server = None
        self.update_interval = 1000

    def setup(self) -> Any:
        """Set up dashboard application."""
        server = Flask(__name__)
        app = dash.Dash(
            __name__,
            server=server,
            external_stylesheets=[dbc.themes.DARKLY],
            suppress_callback_exceptions=True,
        )
        app.title = "AlphaMind Trading Dashboard"
        app.layout = self._create_layout()
        self._register_callbacks(app)
        self.app = app
        self.server = server
        logger.info("Dashboard server set up")

    def run(self) -> Any:
        """Run dashboard server."""
        if not self.app:
            self.setup()
        logger.info(f"Starting dashboard server on {self.host}:{self.port}")
        self.app.run_server(host=self.host, port=self.port, debug=self.debug)

    def _create_layout(self) -> Any:
        """
        Create dashboard layout.

        Returns:
            Dash layout
        """
        navbar = dbc.Navbar(
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Img(src="/assets/logo.png", height="30px"),
                                width="auto",
                            ),
                            dbc.Col(
                                dbc.NavbarBrand(
                                    "AlphaMind Trading Dashboard", className="ms-2"
                                )
                            ),
                        ],
                        align="center",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Nav(
                                    [
                                        dbc.NavItem(
                                            dbc.NavLink("Overview", href="#overview")
                                        ),
                                        dbc.NavItem(
                                            dbc.NavLink("Portfolio", href="#portfolio")
                                        ),
                                        dbc.NavItem(
                                            dbc.NavLink(
                                                "Performance", href="#performance"
                                            )
                                        ),
                                        dbc.NavItem(
                                            dbc.NavLink(
                                                "Market Data", href="#market-data"
                                            )
                                        ),
                                        dbc.NavItem(
                                            dbc.NavLink("Orders", href="#orders")
                                        ),
                                        dbc.NavItem(
                                            dbc.NavLink("System", href="#system")
                                        ),
                                    ],
                                    navbar=True,
                                ),
                                align="center",
                            )
                        ]
                    ),
                ],
                fluid=True,
            ),
            color="dark",
            dark=True,
            className="mb-4",
        )
        overview_cards = dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Portfolio Value"),
                            dbc.CardBody(
                                [
                                    html.H3(id="portfolio-value", children="$0.00"),
                                    html.P(id="portfolio-change", children="0.00%"),
                                ]
                            ),
                        ]
                    ),
                    width=3,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Daily P&L"),
                            dbc.CardBody(
                                [
                                    html.H3(id="daily-pnl", children="$0.00"),
                                    html.P(id="daily-pnl-percent", children="0.00%"),
                                ]
                            ),
                        ]
                    ),
                    width=3,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Total P&L"),
                            dbc.CardBody(
                                [
                                    html.H3(id="total-pnl", children="$0.00"),
                                    html.P(id="total-pnl-percent", children="0.00%"),
                                ]
                            ),
                        ]
                    ),
                    width=3,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Sharpe Ratio"),
                            dbc.CardBody(
                                [
                                    html.H3(id="sharpe-ratio", children="0.00"),
                                    html.P("Annualized"),
                                ]
                            ),
                        ]
                    ),
                    width=3,
                ),
            ],
            className="mb-4",
        )
        portfolio_section = html.Div(
            [
                html.H2("Portfolio", id="portfolio", className="mb-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Equity Curve"),
                                    dbc.CardBody(dcc.Graph(id="equity-curve")),
                                ]
                            ),
                            width=8,
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Asset Allocation"),
                                    dbc.CardBody(dcc.Graph(id="asset-allocation")),
                                ]
                            ),
                            width=4,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Card(
                    [
                        dbc.CardHeader("Positions"),
                        dbc.CardBody(html.Div(id="positions-table")),
                    ],
                    className="mb-4",
                ),
            ]
        )
        performance_section = html.Div(
            [
                html.H2("Performance", id="performance", className="mb-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Returns"),
                                    dbc.CardBody(dcc.Graph(id="returns-chart")),
                                ]
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Drawdowns"),
                                    dbc.CardBody(dcc.Graph(id="drawdowns-chart")),
                                ]
                            ),
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Performance Metrics"),
                                    dbc.CardBody(html.Div(id="performance-metrics")),
                                ]
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Risk Metrics"),
                                    dbc.CardBody(html.Div(id="risk-metrics")),
                                ]
                            ),
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
            ]
        )
        market_data_section = html.Div(
            [
                html.H2("Market Data", id="market-data", className="mb-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col("Price Chart", width=8),
                                                    dbc.Col(
                                                        dcc.Dropdown(
                                                            id="market-symbol-dropdown",
                                                            options=[],
                                                            value=None,
                                                            placeholder="Select Symbol",
                                                            className="dash-dropdown",
                                                        ),
                                                        width=4,
                                                    ),
                                                ]
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(dcc.Graph(id="price-chart")),
                                ]
                            ),
                            width=8,
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Market Overview"),
                                    dbc.CardBody(html.Div(id="market-overview")),
                                ]
                            ),
                            width=4,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Card(
                    [
                        dbc.CardHeader("Order Book"),
                        dbc.CardBody(dcc.Graph(id="order-book")),
                    ],
                    className="mb-4",
                ),
            ]
        )
        orders_section = html.Div(
            [
                html.H2("Orders", id="orders", className="mb-3"),
                dbc.Card(
                    [
                        dbc.CardHeader("Active Orders"),
                        dbc.CardBody(html.Div(id="active-orders-table")),
                    ],
                    className="mb-4",
                ),
                dbc.Card(
                    [
                        dbc.CardHeader("Recent Trades"),
                        dbc.CardBody(html.Div(id="recent-trades-table")),
                    ],
                    className="mb-4",
                ),
            ]
        )
        system_section = html.Div(
            [
                html.H2("System", id="system", className="mb-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("CPU Usage"),
                                    dbc.CardBody(dcc.Graph(id="cpu-usage-chart")),
                                ]
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Memory Usage"),
                                    dbc.CardBody(dcc.Graph(id="memory-usage-chart")),
                                ]
                            ),
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Latency"),
                                    dbc.CardBody(dcc.Graph(id="latency-chart")),
                                ]
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Errors"),
                                    dbc.CardBody(dcc.Graph(id="errors-chart")),
                                ]
                            ),
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
            ]
        )
        layout = html.Div(
            [
                navbar,
                dbc.Container(
                    [
                        dcc.Interval(
                            id="interval-component",
                            interval=self.update_interval,
                            n_intervals=0,
                        ),
                        html.H1("AlphaMind Trading Dashboard", className="mb-4"),
                        overview_cards,
                        portfolio_section,
                        performance_section,
                        market_data_section,
                        orders_section,
                        system_section,
                    ],
                    fluid=True,
                ),
            ]
        )
        return layout

    def _register_callbacks(self, app: Any) -> Any:
        """
        Register dashboard callbacks.

        Args:
            app: Dash application
        """

        @app.callback(
            [
                Output("portfolio-value", "children"),
                Output("portfolio-change", "children"),
                Output("portfolio-change", "className"),
                Output("daily-pnl", "children"),
                Output("daily-pnl-percent", "children"),
                Output("daily-pnl-percent", "className"),
                Output("total-pnl", "children"),
                Output("total-pnl-percent", "children"),
                Output("total-pnl-percent", "className"),
                Output("sharpe-ratio", "children"),
            ],
            [Input("interval-component", "n_intervals")],
        )
        def update_overview_cards(n):
            summary = self.metrics.get_summary()
            portfolio_value = summary["portfolio"]["value"]
            portfolio_value_str = f"${portfolio_value:,.2f}"
            daily_return = summary["performance"]["daily_return"]
            portfolio_change_str = f"{daily_return:.2%}"
            portfolio_change_class = (
                "text-success" if daily_return >= 0 else "text-danger"
            )
            daily_pnl = portfolio_value * daily_return
            daily_pnl_str = f"${daily_pnl:,.2f}"
            daily_pnl_percent_str = f"{daily_return:.2%}"
            daily_pnl_class = "text-success" if daily_pnl >= 0 else "text-danger"
            total_pnl = summary["portfolio"]["total_pnl"]
            total_pnl_str = f"${total_pnl:,.2f}"
            initial_value = portfolio_value - total_pnl
            total_pnl_percent = total_pnl / initial_value if initial_value > 0 else 0
            total_pnl_percent_str = f"{total_pnl_percent:.2%}"
            total_pnl_class = "text-success" if total_pnl >= 0 else "text-danger"
            sharpe_ratio = summary["performance"]["sharpe_ratio"]
            sharpe_ratio_str = f"{sharpe_ratio:.2f}"
            return (
                portfolio_value_str,
                portfolio_change_str,
                portfolio_change_class,
                daily_pnl_str,
                daily_pnl_percent_str,
                daily_pnl_class,
                total_pnl_str,
                total_pnl_percent_str,
                total_pnl_class,
                sharpe_ratio_str,
            )

        @app.callback(
            Output("equity-curve", "figure"),
            [Input("interval-component", "n_intervals")],
        )
        def update_equity_curve(n):
            fig = go.Figure()
            if self.metrics.portfolio_value_history:
                df = pd.DataFrame(self.metrics.portfolio_value_history)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["value"],
                        mode="lines",
                        name="Portfolio Value",
                        line=dict(color="#2FA4E7", width=2),
                    )
                )
            if self.metrics.cash_history:
                df = pd.DataFrame(self.metrics.cash_history)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["value"],
                        mode="lines",
                        name="Cash",
                        line=dict(color="#73B9EE", width=2, dash="dash"),
                    )
                )
            if self.metrics.equity_history:
                df = pd.DataFrame(self.metrics.equity_history)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["value"],
                        mode="lines",
                        name="Equity",
                        line=dict(color="#2780E3", width=2, dash="dot"),
                    )
                )
            fig.update_layout(
                title="Equity Curve",
                xaxis_title="Time",
                yaxis_title="Value ($)",
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_dark",
                hovermode="x unified",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )
            return fig

        @app.callback(
            Output("asset-allocation", "figure"),
            [Input("interval-component", "n_intervals")],
        )
        def update_asset_allocation(n):
            summary = self.metrics.get_summary()
            positions = summary["portfolio"]["positions"]
            labels = []
            values = []
            for symbol, pos_data in positions.items():
                market_value = pos_data.get("market_value", 0.0)
                if market_value > 0:
                    labels.append(symbol)
                    values.append(market_value)
            cash = summary["portfolio"]["cash"]
            if cash > 0 or not values:
                labels.append("Cash")
                values.append(cash)
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.3,
                        marker_colors=dbc.themes.DARKLY_colors,
                    )
                ]
            )
            fig.update_layout(
                title="Asset Allocation",
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_dark",
                showlegend=True,
            )
            return fig

        @app.callback(
            Output("positions-table", "children"),
            [Input("interval-component", "n_intervals")],
        )
        def update_positions_table(n):
            summary = self.metrics.get_summary()
            positions = summary["portfolio"]["positions"]
            if not positions:
                return html.P("No active positions.")
            data = []
            for symbol, pos_data in positions.items():
                unrealized_pnl = pos_data.get("unrealized_pnl", 0.0)
                pnl_class = "text-success" if unrealized_pnl >= 0 else "text-danger"
                data.append(
                    (
                        symbol,
                        pos_data.get("quantity", 0),
                        f"${pos_data.get('average_price', 0.0):,.2f}",
                        f"${pos_data.get('current_price', 0.0):,.2f}",
                        f"${pos_data.get('market_value', 0.0):,.2f}",
                        html.Span(f"${unrealized_pnl:,.2f}", className=pnl_class),
                        html.Span(
                            f"{pos_data.get('unrealized_pnl_percent', 0.0):.2%}",
                            className=pnl_class,
                        ),
                    )
                )
            table_header = [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Symbol"),
                            html.Th("Quantity"),
                            html.Th("Avg Price"),
                            html.Th("Current Price"),
                            html.Th("Market Value"),
                            html.Th("Unrealized P&L ($)"),
                            html.Th("Unrealized P&L (%)"),
                        ]
                    )
                )
            ]
            table_body = [
                html.Tbody([html.Tr([html.Td(col) for col in row]) for row in data])
            ]
            return dbc.Table(
                table_header + table_body,
                striped=True,
                bordered=True,
                hover=True,
                dark=True,
                responsive=True,
            )

        @app.callback(
            Output("returns-chart", "figure"),
            [Input("interval-component", "n_intervals")],
        )
        def update_returns_chart(n):
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=("Cumulative Returns", "Daily Returns"),
            )
            if self.metrics.cumulative_returns:
                df = pd.DataFrame(self.metrics.cumulative_returns)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["value"] * 100,
                        mode="lines",
                        name="Cumulative Returns (%)",
                        line=dict(color="#17A2B8", width=2),
                    ),
                    row=1,
                    col=1,
                )
            if self.metrics.daily_returns:
                df = pd.DataFrame(self.metrics.daily_returns)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                fig.add_trace(
                    go.Bar(
                        x=df["timestamp"],
                        y=df["value"] * 100,
                        name="Daily Returns (%)",
                        marker_color=[
                            "#28A745" if val >= 0 else "#DC3545" for val in df["value"]
                        ],
                    ),
                    row=2,
                    col=1,
                )
            fig.update_layout(
                title="Trading Strategy Returns",
                margin=dict(l=20, r=20, t=60, b=20),
                template="plotly_dark",
                hovermode="x unified",
            )
            fig.update_yaxes(title_text="Returns (%)", row=1, col=1)
            fig.update_yaxes(title_text="Returns (%)", row=2, col=1)
            return fig

        @app.callback(
            Output("drawdowns-chart", "figure"),
            [Input("interval-component", "n_intervals")],
        )
        def update_drawdowns_chart(n):
            fig = go.Figure()
            if self.metrics.drawdowns:
                df = pd.DataFrame(self.metrics.drawdowns)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["value"] * 100,
                        mode="lines",
                        name="Drawdown (%)",
                        line=dict(color="#DC3545", width=2, shape="hvh"),
                        fill="tozeroy",
                        fillcolor="rgba(220, 53, 69, 0.3)",
                    )
                )
            fig.update_layout(
                title=f"Drawdowns (Max: {self.metrics.max_drawdown:.2%})",
                xaxis_title="Time",
                yaxis_title="Drawdown (%)",
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_dark",
                hovermode="x unified",
            )
            fig.update_yaxes(tickformat=".2f")
            return fig

        @app.callback(
            Output("performance-metrics", "children"),
            [Input("interval-component", "n_intervals")],
        )
        def update_performance_metrics(n):
            summary = self.metrics.get_summary()
            perf_metrics = summary["performance"]
            return dbc.Table(
                [
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td("Cumulative Return"),
                                    html.Td(
                                        f"{perf_metrics['cumulative_return']:.2%}",
                                        className=(
                                            "text-success"
                                            if perf_metrics["cumulative_return"] >= 0
                                            else "text-danger"
                                        ),
                                    ),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Daily Return"),
                                    html.Td(
                                        f"{perf_metrics['daily_return']:.2%}",
                                        className=(
                                            "text-success"
                                            if perf_metrics["daily_return"] >= 0
                                            else "text-danger"
                                        ),
                                    ),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Sharpe Ratio (Annualized)"),
                                    html.Td(f"{perf_metrics['sharpe_ratio']:.2f}"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Sortino Ratio (Annualized)"),
                                    html.Td(f"{perf_metrics['sortino_ratio']:.2f}"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Max Drawdown"),
                                    html.Td(
                                        f"{perf_metrics['max_drawdown']:.2%}",
                                        className="text-danger",
                                    ),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Win Rate"),
                                    html.Td(f"{perf_metrics['win_rate']:.2%}"),
                                ]
                            ),
                        ]
                    )
                ],
                striped=True,
                bordered=True,
                hover=True,
                dark=True,
            )

        @app.callback(
            Output("risk-metrics", "children"),
            [Input("interval-component", "n_intervals")],
        )
        def update_risk_metrics(n):
            summary = self.metrics.get_summary()
            risk_metrics = summary["risk"]
            return dbc.Table(
                [
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td("Volatility (Annualized)"),
                                    html.Td(f"{risk_metrics['volatility']:.2%}"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("VaR 95%"),
                                    html.Td(f"{risk_metrics['var_95']:.2%}"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("VaR 99%"),
                                    html.Td(f"{risk_metrics['var_99']:.2%}"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Expected Shortfall (CVaR)"),
                                    html.Td(
                                        f"{risk_metrics['expected_shortfall']:.2%}"
                                    ),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Beta"),
                                    html.Td(f"{risk_metrics['beta']:.2f}"),
                                ]
                            ),
                        ]
                    )
                ],
                striped=True,
                bordered=True,
                hover=True,
                dark=True,
            )

        @app.callback(
            Output("market-symbol-dropdown", "options"),
            [Input("interval-component", "n_intervals")],
        )
        def update_market_symbol_options(n):
            symbols = list(self.metrics.market_data.keys())
            options = [{"label": s, "value": s} for s in symbols]
            return options

        @app.callback(
            [
                Output("price-chart", "figure"),
                Output("market-overview", "children"),
                Output("order-book", "figure"),
            ],
            [
                Input("interval-component", "n_intervals"),
                Input("market-symbol-dropdown", "value"),
            ],
        )
        def update_market_data_charts(n, selected_symbol):
            price_fig = go.Figure()
            order_book_fig = go.Figure()
            market_overview_content = html.P("Select a symbol to view market data.")
            if selected_symbol and selected_symbol in self.metrics.market_data:
                data_list = self.metrics.market_data[selected_symbol]
                df = pd.DataFrame(data_list)
                if (
                    "timestamp" in df.columns
                    and "price" in df.columns
                    and (len(df) > 0)
                ):
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    price_fig = go.Figure(
                        data=[
                            go.Scatter(
                                x=df["timestamp"],
                                y=df["price"],
                                mode="lines",
                                name="Price",
                                line=dict(color="#2FA4E7", width=2),
                            )
                        ]
                    )
                    price_fig.update_layout(
                        title=f"{selected_symbol} Price",
                        xaxis_title="Time",
                        yaxis_title="Price",
                        margin=dict(l=20, r=20, t=40, b=20),
                        template="plotly_dark",
                        hovermode="x unified",
                    )
                    latest_data = df.iloc[-1]
                    market_overview_content = dbc.Table(
                        [
                            html.Tbody(
                                [
                                    html.Tr(
                                        [
                                            html.Td("Latest Price"),
                                            html.Td(
                                                f"${latest_data.get('price', 0.0):,.2f}"
                                            ),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td("Volume"),
                                            html.Td(
                                                f"{latest_data.get('volume', 0.0):,.0f}"
                                            ),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td("Bid Price"),
                                            html.Td(
                                                f"${latest_data.get('bid_price', 0.0):,.2f}"
                                            ),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td("Ask Price"),
                                            html.Td(
                                                f"${latest_data.get('ask_price', 0.0):,.2f}"
                                            ),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td("Spread"),
                                            html.Td(
                                                f"${latest_data.get('ask_price', 0.0) - latest_data.get('bid_price', 0.0):,.2f}"
                                            ),
                                        ]
                                    ),
                                ]
                            )
                        ],
                        striped=True,
                        bordered=True,
                        hover=True,
                        dark=True,
                    )
                    if (
                        "bid_price" in latest_data
                        and "ask_price" in latest_data
                        and ("volume" in latest_data)
                    ):
                        bids = latest_data.get(
                            "bids",
                            [
                                (
                                    latest_data.get("bid_price", 0.0),
                                    latest_data.get("volume", 0.0),
                                )
                            ],
                        )
                        asks = latest_data.get(
                            "asks",
                            [
                                (
                                    latest_data.get("ask_price", 0.0),
                                    latest_data.get("volume", 0.0),
                                )
                            ],
                        )
                        bid_prices = [b[0] for b in bids]
                        bid_volumes = [b[1] for b in bids]
                        ask_prices = [a[0] for a in asks]
                        ask_volumes = [a[1] for a in asks]
                        cumulative_bid_volume = np.cumsum(bid_volumes[::-1])[::-1]
                        cumulative_ask_volume = np.cumsum(ask_volumes)
                        order_book_fig = go.Figure(
                            data=[
                                go.Scatter(
                                    x=bid_prices,
                                    y=cumulative_bid_volume,
                                    mode="lines",
                                    line=dict(shape="hvh", color="#28A745"),
                                    name="Bids (Cumulative)",
                                    fill="tozeroy",
                                ),
                                go.Scatter(
                                    x=ask_prices,
                                    y=cumulative_ask_volume,
                                    mode="lines",
                                    line=dict(shape="hvh", color="#DC3545"),
                                    name="Asks (Cumulative)",
                                    fill="tozeroy",
                                ),
                            ]
                        )
                        order_book_fig.update_layout(
                            title=f"{selected_symbol} Order Book",
                            xaxis_title="Price",
                            yaxis_title="Cumulative Volume",
                            margin=dict(l=20, r=20, t=40, b=20),
                            template="plotly_dark",
                            hovermode="x unified",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1,
                            ),
                        )
            return (price_fig, market_overview_content, order_book_fig)

        @app.callback(
            Output("active-orders-table", "children"),
            [Input("interval-component", "n_intervals")],
        )
        def update_active_orders_table(n):
            summary = self.metrics.get_summary()
            active_orders = summary["trading"]["active_orders"]
            if not active_orders:
                return html.P("No active orders.")
            data = []
            for order in active_orders[-10:]:
                data.append(
                    (
                        (
                            order.get("timestamp", "").strftime("%Y-%m-%d %H:%M:%S")
                            if isinstance(order.get("timestamp"), datetime)
                            else order.get("timestamp", "N/A")
                        ),
                        order.get("symbol", "N/A"),
                        order.get("type", "N/A"),
                        order.get("side", "N/A"),
                        order.get("quantity", 0),
                        (
                            f"${order.get('price', 0.0):,.2f}"
                            if order.get("price") is not None
                            else "Market"
                        ),
                        order.get("status", "N/A"),
                    )
                )
            table_header = [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Time"),
                            html.Th("Symbol"),
                            html.Th("Type"),
                            html.Th("Side"),
                            html.Th("Quantity"),
                            html.Th("Price"),
                            html.Th("Status"),
                        ]
                    )
                )
            ]
            table_body = [
                html.Tbody([html.Tr([html.Td(col) for col in row]) for row in data])
            ]
            return dbc.Table(
                table_header + table_body,
                striped=True,
                bordered=True,
                hover=True,
                dark=True,
                responsive=True,
            )

        @app.callback(
            Output("recent-trades-table", "children"),
            [Input("interval-component", "n_intervals")],
        )
        def update_recent_trades_table(n):
            trades = self.metrics.trades_history
            if not trades:
                return html.P("No recent trades.")
            data = []
            for trade in trades[-10:]:
                pnl = trade.get("realized_pnl")
                pnl_class = (
                    "text-success"
                    if pnl >= 0
                    else "text-danger" if pnl is not None else ""
                )
                data.append(
                    (
                        (
                            trade.get("timestamp", "").strftime("%Y-%m-%d %H:%M:%S")
                            if isinstance(trade.get("timestamp"), datetime)
                            else trade.get("timestamp", "N/A")
                        ),
                        trade.get("symbol", "N/A"),
                        trade.get("side", "N/A"),
                        trade.get("quantity", 0),
                        f"${trade.get('price', 0.0):,.2f}",
                        html.Span(
                            f"${pnl:,.2f}" if pnl is not None else "N/A",
                            className=pnl_class,
                        ),
                    )
                )
            table_header = [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Time"),
                            html.Th("Symbol"),
                            html.Th("Side"),
                            html.Th("Quantity"),
                            html.Th("Price"),
                            html.Th("Realized P&L"),
                        ]
                    )
                )
            ]
            table_body = [
                html.Tbody([html.Tr([html.Td(col) for col in row]) for row in data])
            ]
            return dbc.Table(
                table_header + table_body,
                striped=True,
                bordered=True,
                hover=True,
                dark=True,
                responsive=True,
            )

        @app.callback(
            [
                Output("cpu-usage-chart", "figure"),
                Output("memory-usage-chart", "figure"),
                Output("latency-chart", "figure"),
                Output("errors-chart", "figure"),
            ],
            [Input("interval-component", "n_intervals")],
        )
        def update_system_charts(n):
            cpu_data = self.metrics.system_health["cpu_usage"]
            memory_data = self.metrics.system_health["memory_usage"]
            latency_data = self.metrics.system_health["latency"]
            errors_data = self.metrics.system_health["errors"]

            def create_chart(data, title, yaxis_title, color):
                fig = go.Figure()
                if data:
                    df = pd.DataFrame(data)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    fig.add_trace(
                        go.Scatter(
                            x=df["timestamp"],
                            y=df["value"],
                            mode="lines",
                            line=dict(color=color, width=2),
                        )
                    )
                fig.update_layout(
                    title=title,
                    xaxis_title="Time",
                    yaxis_title=yaxis_title,
                    margin=dict(l=20, r=20, t=40, b=20),
                    template="plotly_dark",
                    hovermode="x unified",
                )
                return fig

            cpu_fig = create_chart(cpu_data, "CPU Usage", "Usage (%)", "#F0AD4E")
            memory_fig = create_chart(
                memory_data, "Memory Usage", "Usage (%)", "#5BC0DE"
            )
            latency_fig = create_chart(
                latency_data, "Latency", "Latency (ms)", "#428BCA"
            )
            errors_fig = create_chart(errors_data, "Errors", "Count", "#DC3545")
            return (cpu_fig, memory_fig, latency_fig, errors_fig)


if __name__ == "__main__":
    metrics = DashboardMetrics()
    now = datetime.now()
    for i in range(100):
        timestamp = now - timedelta(hours=100 - i)
        metrics.update_portfolio(
            timestamp=timestamp,
            portfolio_value=100000 + i * 100 + np.random.normal(0, 500),
            cash=50000 + i * 50 + np.random.normal(0, 200),
            positions={
                "AAPL": {
                    "quantity": 10 + i % 5,
                    "average_price": 150.0,
                    "current_price": 150.0 + np.random.normal(0, 1),
                    "market_value": (10 + i % 5) * (150.0 + np.random.normal(0, 1)),
                    "unrealized_pnl": (10 + i % 5) * np.random.normal(0, 1),
                    "unrealized_pnl_percent": np.random.normal(0, 0.01),
                },
                "GOOGL": {
                    "quantity": 5 + i % 3,
                    "average_price": 2500.0,
                    "current_price": 2500.0 + np.random.normal(0, 5),
                    "market_value": (5 + i % 3) * (2500.0 + np.random.normal(0, 5)),
                    "unrealized_pnl": (5 + i % 3) * np.random.normal(0, 5),
                    "unrealized_pnl_percent": np.random.normal(0, 0.005),
                },
            },
        )
        if i % 10 == 0 and i > 0:
            metrics.update_trade(
                {
                    "timestamp": timestamp,
                    "symbol": "AAPL",
                    "side": "SELL",
                    "quantity": 1,
                    "price": 150.0 + np.random.normal(0, 1),
                    "realized_pnl": np.random.normal(0, 50),
                }
            )
        metrics.update_system_health(
            cpu_usage=np.random.uniform(10, 80),
            memory_usage=np.random.uniform(20, 70),
            latency=np.random.uniform(5, 50),
            errors=np.random.randint(0, 2),
        )
        metrics.update_market_data(
            "AAPL",
            {
                "timestamp": timestamp,
                "price": 150.0 + np.random.normal(0, 1),
                "volume": np.random.randint(100, 1000),
                "bid_price": 150.0 + np.random.normal(0, 0.5),
                "ask_price": 150.0 + np.random.normal(0, 0.5) + 0.01,
            },
        )
        metrics.update_market_data(
            "GOOGL",
            {
                "timestamp": timestamp,
                "price": 2500.0 + np.random.normal(0, 5),
                "volume": np.random.randint(10, 100),
                "bid_price": 2500.0 + np.random.normal(0, 2),
                "ask_price": 2500.0 + np.random.normal(0, 2) + 0.1,
            },
        )
    server = DashboardServer(metrics=metrics)
    server.run()
