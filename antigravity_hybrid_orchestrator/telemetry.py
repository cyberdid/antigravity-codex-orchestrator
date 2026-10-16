"""Telemetry and cost tracking engine for Antigravity Hybrid Orchestrator.

Tracks real-time token usage across Cloud Architect (Codex CLI)
and On-Device Workforce (Gemma 4 12B). Computes generation throughput (tokens/sec)
and a reference cost estimate using legacy Gemini token rates. This is not the
actual billing amount for Codex CLI.
"""

from dataclasses import dataclass
from typing import Optional


# Reference pricing benchmarks per 1,000,000 tokens (not Codex CLI billing).
GEMINI_INPUT_PRICE_PER_1M: float = 0.15   # $0.15 / 1M tokens
GEMINI_OUTPUT_PRICE_PER_1M: float = 0.60  # $0.60 / 1M tokens
GEMINI_BLENDED_PRICE_PER_1M: float = 0.375 # ($0.15 + $0.60) / 2 = $0.375 / 1M tokens
LOCAL_PRICE_PER_1M: float = 0.00          # Local RTX 4060 GPU runs at $0 marginal API cost


@dataclass
class TelemetrySummary:
    """Summary of token consumption, financial costs, and generation throughput."""
    cloud_tokens: int
    cloud_cost_usd: float
    local_tokens: int
    local_cost_usd: float
    total_tokens: int
    cost_savings_pct: float
    tokens_per_second: float


class TelemetryTracker:
    """Tracks token consumption and financial accounting across hybrid runtimes."""

    def __init__(self) -> None:
        self.cloud_input_tokens: int = 0
        self.cloud_output_tokens: int = 0
        self.cloud_thinking_tokens: int = 0
        # Cost the CLI reported itself (e.g. claude total_cost_usd), and the tokens
        # of calls that reported no cost, which are priced with the reference rate.
        self.reported_cloud_cost_usd: float = 0.0
        self.unreported_cloud_input_tokens: int = 0
        self.unreported_cloud_output_tokens: int = 0
        self.local_tokens: int = 0
        self.local_prompt_tokens: int = 0
        self.local_duration_s: float = 0.0

    def add_cloud(
        self,
        input_tokens: int,
        output_tokens: int,
        thinking_tokens: int = 0,
        reported_cost_usd: Optional[float] = None,
        unreported_input_tokens: Optional[int] = None,
        unreported_output_tokens: Optional[int] = None,
    ) -> None:
        """Accumulate token and cost telemetry from Cloud Architect invocations.

        Args:
            input_tokens: Prompt/context tokens sent to cloud (all calls).
            output_tokens: Generation tokens emitted by cloud (all calls).
            thinking_tokens: Optional reasoning tokens included in generation.
            reported_cost_usd: Cost the CLI reported for its calls, if any.
            unreported_input_tokens: Input tokens of calls without a reported cost.
                Defaults to all input tokens when no cost was reported, else 0.
            unreported_output_tokens: Output tokens of calls without a reported cost.
        """
        self.cloud_input_tokens += max(0, int(input_tokens))
        self.cloud_output_tokens += max(0, int(output_tokens))
        self.cloud_thinking_tokens += max(0, int(thinking_tokens))
        if reported_cost_usd is not None:
            self.reported_cloud_cost_usd += max(0.0, float(reported_cost_usd))
        if unreported_input_tokens is None:
            unreported_input_tokens = 0 if reported_cost_usd is not None else input_tokens
        if unreported_output_tokens is None:
            unreported_output_tokens = 0 if reported_cost_usd is not None else output_tokens
        self.unreported_cloud_input_tokens += max(0, int(unreported_input_tokens))
        self.unreported_cloud_output_tokens += max(0, int(unreported_output_tokens))

    def add_local(
        self,
        eval_tokens: int,
        duration_s: float,
        prompt_tokens: int = 0,
    ) -> None:
        """Accumulate token telemetry from On-Device Workforce (Gemma 4) invocations.

        Args:
            eval_tokens: Number of evaluation (generated) tokens from local GPU.
            duration_s: Wall-clock generation duration in seconds.
            prompt_tokens: Optional prompt evaluation tokens processed locally.
        """
        self.local_tokens += max(0, int(eval_tokens))
        self.local_duration_s += max(0.0, float(duration_s))
        self.local_prompt_tokens += max(0, int(prompt_tokens))

    def get_summary(self) -> TelemetrySummary:
        """Compute and return an aggregated telemetry and cost summary.

        Calculations:
            - Cloud Cost: (in_tokens * $0.15/1M) + (out_tokens * $0.60/1M)
            - Local Cost: $0.00 (local hardware inference)
            - Tokens Per Second: local_eval_tokens / local_eval_duration_s
            - Cost Savings %: Hypothetical savings vs the reference-rate baseline
        """
        cloud_total = self.cloud_input_tokens + self.cloud_output_tokens
        total_tokens = cloud_total + self.local_tokens

        # Reported costs are used as-is; only tokens without a reported cost are
        # estimated with the reference rate.
        cloud_cost = (
            self.reported_cloud_cost_usd
            + (self.unreported_cloud_input_tokens * (GEMINI_INPUT_PRICE_PER_1M / 1_000_000.0))
            + (self.unreported_cloud_output_tokens * (GEMINI_OUTPUT_PRICE_PER_1M / 1_000_000.0))
        )
        local_cost = 0.0

        # Throughput calculation (tokens / second)
        tps = (self.local_tokens / self.local_duration_s) if self.local_duration_s > 0.0 else 0.0

        # Hypothetical cloud cost if all local tokens had been sent to cloud
        hypothetical_cloud_cost = cloud_cost + (
            self.local_tokens * (GEMINI_BLENDED_PRICE_PER_1M / 1_000_000.0)
        )

        if hypothetical_cloud_cost > 0.0:
            savings_pct = (
                (hypothetical_cloud_cost - cloud_cost) / hypothetical_cloud_cost
            ) * 100.0
        else:
            savings_pct = 0.0

        return TelemetrySummary(
            cloud_tokens=cloud_total,
            cloud_cost_usd=round(cloud_cost, 6),
            local_tokens=self.local_tokens,
            local_cost_usd=local_cost,
            total_tokens=total_tokens,
            cost_savings_pct=round(savings_pct, 2),
            tokens_per_second=round(tps, 2),
        )

    def reset(self) -> None:
        """Reset all tracked counters to initial zero state."""
        self.cloud_input_tokens = 0
        self.cloud_output_tokens = 0
        self.cloud_thinking_tokens = 0
        self.reported_cloud_cost_usd = 0.0
        self.unreported_cloud_input_tokens = 0
        self.unreported_cloud_output_tokens = 0
        self.local_tokens = 0
        self.local_prompt_tokens = 0
        self.local_duration_s = 0.0

