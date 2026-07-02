from config import ACCOUNT_SIZE, RISK_GRADE_A, RISK_GRADE_B, RISK_GRADE_C, RR_RATIO, RR_RATIO_B, RR_RATIO_C


def calculate_position(entry: float, stop_loss: float, grade: str = "C",
                       account_size: float | None = None) -> dict:
    """
    Grade-tiered position sizing and reward targets:
      A  (ORB + sector + RS + vol ≥2×)   → 1.0% risk, 2:1 R:R
      B  (3 of 4, vol ≥1.5×)             → 0.75% risk, 2:1 R:R
      C  (borderline)                     → 0.5% risk, 1.5:1 R:R

    Pass account_size to compound on live equity (default = config ACCOUNT_SIZE).
    """
    effective_account = account_size if account_size is not None else ACCOUNT_SIZE

    if grade in ("A", "A+"):
        risk_pct = RISK_GRADE_A
        rr       = RR_RATIO
    elif grade == "B":
        risk_pct = RISK_GRADE_B
        rr       = RR_RATIO_B
    else:
        risk_pct = RISK_GRADE_C
        rr       = RR_RATIO_C

    risk_amount   = effective_account * risk_pct
    risk_per_unit = abs(entry - stop_loss)

    if risk_per_unit < 0.01:
        return {}

    units          = max(1, int(risk_amount / risk_per_unit))
    position_value = round(units * entry, 2)
    target_pnl     = round(risk_amount * rr, 2)
    pct_of_account = round(position_value / effective_account * 100, 1)

    return {
        "units":           units,
        "position_value":  position_value,
        "risk_amount":     round(risk_amount, 2),
        "target_pnl":      target_pnl,
        "pct_of_account":  pct_of_account,
        "grade":           grade,
        "risk_pct":        round(risk_pct * 100, 2),
        "rr":              rr,
    }
