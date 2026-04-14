from __future__ import annotations

import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from trading_algo.quant_core.edge_validation.types import (
    EdgeValidationReport,
    PatternEdgeReport,
    Verdict,
)


VERDICT_COLORS: dict[Verdict, str] = {"PASS": "green", "WEAK": "yellow", "FAIL": "red"}


def _fmt_pval(p: float) -> str:
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def _fmt_pts(v: float) -> str:
    return f"{v:.2f}"


def _verdict_text(verdict: Verdict, conditions_met: int, conditions_total: int) -> Text:
    color = VERDICT_COLORS[verdict]
    return Text(f"{verdict} ({conditions_met}/{conditions_total})", style=f"bold {color}")


class EdgeValidationReporter:
    def __init__(self) -> None:
        self.console = Console()

    def render(self, report: EdgeValidationReport, point_values: dict[str, float] | None = None) -> None:
        c = self.console
        c.print()
        c.rule("[bold cyan]Edge Validation Report[/bold cyan]")
        c.print()

        for symbol, patterns in report.symbol_reports.items():
            if not patterns:
                continue

            c.rule(f"[bold]{symbol}[/bold]")

            for pr in patterns:
                self._render_pattern(pr, point_values)

        if report.whites_rc is not None:
            c.print()
            rc = report.whites_rc
            rc_color = "green" if rc.significant else "red"
            rc_panel = Table(show_header=False, box=None, padding=(0, 2))
            rc_panel.add_column(style="dim")
            rc_panel.add_column()
            rc_panel.add_row("Best Strategy", rc.best_strategy)
            rc_panel.add_row("Observed Best Metric", _fmt_pts(rc.observed_best_metric))
            rc_panel.add_row("Data Mining Bias", _fmt_pts(rc.data_mining_bias))
            rc_panel.add_row("Adjusted Metric", _fmt_pts(rc.adjusted_metric))
            rc_panel.add_row("p-value", _fmt_pval(rc.p_value))
            rc_panel.add_row("Significant", Text("YES" if rc.significant else "NO", style=f"bold {rc_color}"))
            c.print(Panel(rc_panel, title="[bold]White's Reality Check[/bold]", border_style="cyan"))

        c.print()
        c.rule("[bold cyan]Summary[/bold cyan]")
        summary = Table(show_header=False, box=None, padding=(0, 2))
        summary.add_column(style="dim")
        summary.add_column()
        summary.add_row("Total Tested", str(report.total_patterns_tested))
        summary.add_row("Passed", Text(str(report.patterns_passed), style="bold green"))
        summary.add_row("Weak", Text(str(report.patterns_weak), style="bold yellow"))
        summary.add_row("Failed", Text(str(report.patterns_failed), style="bold red"))
        c.print(summary)
        c.print()

    def _render_pattern(self, pr: PatternEdgeReport, point_values: dict[str, float] | None) -> None:
        c = self.console
        color = VERDICT_COLORS[pr.verdict]

        header = Text()
        header.append(f"{pr.pattern_name}", style="bold")
        header.append("  ")
        header.append_text(_verdict_text(pr.verdict, pr.conditions_met, pr.conditions_total))

        # --- Summary stats ---
        stats = Table(show_header=False, box=None, padding=(0, 2))
        stats.add_column(style="dim")
        stats.add_column()
        stats.add_row("Occurrences", str(pr.n_occurrences))
        stats.add_row("Win Rate", f"{pr.win_rate:.1%}")
        stats.add_row("Mean Return", f"{_fmt_pts(pr.mean_return_points)} pts")
        stats.add_row("Median Return", f"{_fmt_pts(pr.median_return_points)} pts")
        pf_str = f"{pr.profit_factor:.2f}" if pr.profit_factor != float("inf") else "inf"
        stats.add_row("Profit Factor", pf_str)

        # --- Excursion ---
        exc = pr.excursion
        med_mfe = float(np.median(exc.pattern.mfe_points)) if exc.pattern.n_occurrences > 0 else 0.0
        med_mae = float(np.median(exc.pattern.mae_points)) if exc.pattern.n_occurrences > 0 else 0.0
        ratio = med_mfe / med_mae if med_mae > 0 else float("inf")
        ratio_str = f"{ratio:.2f}" if ratio != float("inf") else "inf"
        stats.add_row("Median MFE", f"{_fmt_pts(med_mfe)} pts")
        stats.add_row("Median MAE", f"{_fmt_pts(med_mae)} pts")
        stats.add_row("MFE/MAE", ratio_str)

        # --- Significance table ---
        sig = pr.significance
        sig_table = Table(title="Significance Tests", padding=(0, 1))
        sig_table.add_column("Test", style="bold")
        sig_table.add_column("Statistic", justify="right")
        sig_table.add_column("p-value", justify="right")
        sig_table.add_column("Result", justify="center")

        def _result_text(passed: bool) -> Text:
            return Text("PASS" if passed else "FAIL", style="green" if passed else "red")

        sig_table.add_row(
            "Binomial",
            f"{sig.binomial.win_rate:.3f}",
            _fmt_pval(sig.binomial.p_value),
            _result_text(sig.binomial.significant),
        )
        sig_table.add_row(
            "t-test",
            f"{sig.ttest.t_statistic:.3f}",
            _fmt_pval(sig.ttest.p_value),
            _result_text(sig.ttest.significant),
        )
        sig_table.add_row(
            "PSR",
            f"{sig.psr.psr:.3f}",
            "",
            _result_text(sig.psr.significant),
        )
        sig_table.add_row(
            "DSR",
            f"{sig.dsr.dsr:.3f}",
            "",
            _result_text(sig.dsr.significant),
        )
        sig_table.add_row(
            "MinTRL",
            f"{sig.min_trl.min_trl}",
            "",
            _result_text(sig.min_trl.sufficient),
        )

        # --- Monte Carlo ---
        mc = pr.monte_carlo
        mc_table = Table(show_header=False, box=None, padding=(0, 2))
        mc_table.add_column(style="dim")
        mc_table.add_column()
        mc_table.add_row(
            "Sharpe CI",
            f"[{_fmt_pts(mc.bootstrap_sharpe.ci_lower)}, {_fmt_pts(mc.bootstrap_sharpe.ci_upper)}]",
        )
        mc_table.add_row(
            "Profit Factor CI",
            f"[{_fmt_pts(mc.bootstrap_profit_factor.ci_lower)}, {_fmt_pts(mc.bootstrap_profit_factor.ci_upper)}]",
        )
        mc_table.add_row(
            "Permutation p-value",
            _fmt_pval(mc.permutation.p_value),
        )

        # --- Walk-forward ---
        wf = pr.walk_forward
        wf_table = Table(show_header=False, box=None, padding=(0, 2))
        wf_table.add_column(style="dim")
        wf_table.add_column()
        wf_table.add_row("WF Efficiency", f"{wf.wf_efficiency:.2%}")
        wf_table.add_row("Folds Positive", f"{wf.pct_folds_positive:.0%}")
        wf_table.add_row("Stable", Text("YES" if wf.is_stable else "NO", style="green" if wf.is_stable else "red"))

        # --- Regime ---
        regime = pr.regime
        regime_table = Table(title="Regime Breakdown", padding=(0, 1))
        regime_table.add_column("Regime", style="bold")
        regime_table.add_column("N", justify="right")
        regime_table.add_column("Win Rate", justify="right")
        regime_table.add_column("Mean Ret", justify="right")
        regime_table.add_column("Sharpe", justify="right")
        regime_table.add_column("p-value", justify="right")

        for rb in regime.breakdowns:
            regime_table.add_row(
                rb.regime_name,
                str(rb.n_occurrences),
                f"{rb.win_rate:.1%}",
                f"{_fmt_pts(rb.mean_return)} pts",
                f"{rb.sharpe:.2f}",
                _fmt_pval(rb.p_value),
            )

        # --- Expected daily P&L ---
        pv = (point_values or {}).get(pr.symbol, 1.0)
        daily_mean_dollar = pr.mean_return_points * pv
        daily_med_dollar = pr.median_return_points * pv

        pnl_table = Table(show_header=False, box=None, padding=(0, 2))
        pnl_table.add_column(style="dim")
        pnl_table.add_column()
        pnl_table.add_row("Point Value", f"${pv:.0f}")
        pnl_table.add_row("Mean P&L / trade", f"${daily_mean_dollar:,.2f}")
        pnl_table.add_row("Median P&L / trade", f"${daily_med_dollar:,.2f}")

        # --- Assemble panel ---
        from rich.columns import Columns

        c.print(Panel(
            Columns([stats], equal=True),
            title=header,
            border_style=color,
            subtitle=f"{pr.symbol}",
        ))
        c.print(sig_table)
        c.print()
        c.print(Panel(mc_table, title="[bold]Monte Carlo[/bold]", border_style="dim"))
        c.print(Panel(wf_table, title="[bold]Walk-Forward[/bold]", border_style="dim"))
        c.print(regime_table)
        c.print(Panel(pnl_table, title="[bold]Expected P&L[/bold]", border_style="dim"))
        c.print()
