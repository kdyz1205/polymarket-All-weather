"""
Prompt templates for AI factor generation.

These prompts constrain the LLM to produce structured, safe factor expressions
using only the allowed operator primitives.
"""

SYSTEM_PROMPT = """\
You are a quantitative researcher embedded in an automated trading system.
Your job is to generate new predictive factors for prediction markets.

CONSTRAINTS — you MUST follow these exactly:
1. You can ONLY use these operators: {allowed_operators}
2. You can ONLY reference these data fields: mid_price, last_trade_price, \
volume_24h, spread, bid_depth, ask_depth, book_imbalance
3. Each data field is a list[float] representing the time series history
4. Your expression must be a single Python expression that evaluates to a float
5. The output should naturally fall in [-1, 1] range — use clip() if needed
6. Do NOT use any imports, loops, or multi-line code
7. Do NOT use lambda, def, class, or any Python keywords

EXAMPLES of valid expressions:
- clip(ts_zscore(book_imbalance, 50))
- ts_decay_linear(log_return(mid_price, 5), 20)
- clip(ts_corr(book_imbalance, volume_24h, 30))
- sign(ts_delta(ts_mean(spread, 10), 5))
- clip(volume_ratio(volume_24h, 5, 50) - 1.0)
"""

FACTOR_GENERATION_PROMPT = """\
Current market state for {market_name} ({market_id}):
- Recent price trend: {price_trend}
- Current spread: {current_spread:.6f}
- Book imbalance: {book_imbalance:.4f}
- Volume ratio (5/50): {volume_ratio:.2f}
- Current regime: {regime}
- Number of active factors: {active_factor_count}
- Recent factor performance (IC values): {recent_ics}

Existing active factor expressions (do NOT create duplicates):
{existing_factors}

TASK: Generate {n_factors} NEW factor expressions that capture different market dynamics.
For each factor, provide:
1. A short name (snake_case, max 40 chars)
2. A one-sentence description of the economic intuition
3. The expression using ONLY allowed operators

Respond in this exact JSON format:
{{
  "factors": [
    {{
      "name": "factor_name_here",
      "description": "Economic intuition explanation",
      "expression": "clip(ts_zscore(book_imbalance, 50))"
    }}
  ]
}}
"""

FACTOR_REFLECTION_PROMPT = """\
A factor you generated failed sandbox testing.

Factor: {factor_name}
Expression: {expression}
Sandbox result: {sandbox_result}

Analyze why it failed and generate an improved version.
Common failure modes:
- IC too low: the signal has no predictive power, try a different market dynamic
- IR too low: the signal is inconsistent, try smoothing or different windows
- High autocorrelation: signal changes too slowly, add delta or zscore wrapper

Provide an improved factor in the same JSON format.
"""

ATTRIBUTION_PROMPT = """\
Analyze the following market events and factor performance data for {market_name}.

Recent market data (last {window} ticks):
- Price change: {price_change_pct:.2f}%
- Volume change: {volume_change_pct:.2f}%
- Spread change: {spread_change_pct:.2f}%
- Regime: {regime}

Factor performance during this period:
{factor_performance}

Portfolio PnL during this period: {pnl:.2f} USD

TASK: Write a concise attribution analysis (3-5 sentences) explaining:
1. What happened in the market
2. Which factors contributed positively/negatively
3. Any recommended adjustments

Be specific and quantitative. No vague statements.
"""
