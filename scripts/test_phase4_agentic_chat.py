from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase4.chat_handler import answer_question  # noqa: E402


def _contains_source(sources: list[dict], token: str) -> bool:
    for s in sources:
        if token in str(s.get("path", "")):
            return True
    return False


def main() -> None:
    run_id = "2025011600"

    q1 = "What are the forecasted temperature at times when forecasted load is above capacity?"
    a1 = answer_question(run_id, q1)
    plan1 = [p.get("tool") for p in a1.get("plan", [])]
    assert "tool_exceedance_hours" in plan1, f"q1 plan missing exceedance tool: {plan1}"
    assert "tool_weather_at_times" in plan1, f"q1 plan missing weather tool: {plan1}"
    assert _contains_source(a1.get("sources", []), "exceedance_weather.csv"), "q1 missing exceedance_weather source"

    q2 = "Which hours are most risky tomorrow and why?"
    a2 = answer_question(run_id, q2)
    plan2 = [p.get("tool") for p in a2.get("plan", [])]
    assert "tool_top_risk_hours" in plan2, f"q2 plan missing top risk tool: {plan2}"
    text2 = (a2.get("final_markdown") or "").lower()
    assert ("capacity watch" in text2) or ("top capacity watch" in text2) or ("capacity" in text2), "q2 answer missing capacity watchlist grounding"
    assert ("stability watch" in text2) or ("stability" in text2), "q2 answer missing stability watchlist grounding"

    q3 = "How unstable is this forecast?"
    a3 = answer_question(run_id, q3)
    plan3 = [p.get("tool") for p in a3.get("plan", [])]
    assert "tool_compare_revisions" in plan3, f"q3 plan missing compare revisions tool: {plan3}"
    text3 = (a3.get("final_markdown") or "").lower()
    assert ("max range" in text3) or ("avg revision volatility" in text3) or ("revision volatility" in text3), "q3 answer missing revision metrics"

    print("q1 plan:", plan1)
    print("q1 status:", a1.get("status"))
    print("q2 plan:", plan2)
    print("q2 status:", a2.get("status"))
    print("q3 plan:", plan3)
    print("q3 status:", a3.get("status"))
    print("Phase 4-B agentic chat checks passed.")


if __name__ == "__main__":
    main()
