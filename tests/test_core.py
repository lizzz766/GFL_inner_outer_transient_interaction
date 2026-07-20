from __future__ import annotations

from gfl_interaction.energies import pi_energy_trace
from gfl_interaction.model import GFLParameters, simulate_fourth_order
from gfl_interaction.small_gain import swing_certificate_audit


def test_reproduces_qualitative_bandwidth_split() -> None:
    params = GFLParameters()
    case_300 = simulate_fourth_order(300.0, params=params)
    case_500 = simulate_fourth_order(500.0, params=params)
    assert not case_300.stable
    assert case_500.stable


def test_pi_energy_identity_is_numerically_consistent() -> None:
    params = GFLParameters()
    case = simulate_fourth_order(500.0, params=params)
    trace = pi_energy_trace(case, params)
    assert trace.max_balance_error < 2e-3


def test_one_shot_swing_certificate_rejects_before_fast_penalty() -> None:
    params = GFLParameters()
    audit = swing_certificate_audit(params, 500.0)
    assert audit.initial_over_critical > 1.0
    assert audit.local_small_gain.gamma_over_damping < 1.0
