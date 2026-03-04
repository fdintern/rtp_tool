#!/usr/bin/env python3
#
# Copyright (c) 2023 LateGenXer
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#


import dataclasses
import sys

from typing import Any

import lp

from data import hmrc

import tax.uk as UK
import tax.pt as PT
import tax.jp as JP


verbosity = 0

uid = 0


@dataclasses.dataclass
class LPState:
    sipp_uf_1: Any
    sipp_uf_2: Any
    sipp_df_1: Any
    sipp_df_2: Any
    contrib_1: Any
    contrib_2: Any
    tfc_1: Any
    tfc_2: Any
    lsa_1: Any
    lsa_2: Any
    isa: Any
    gia: Any
    cg: Any
    drawdown_1: Any
    drawdown_2: Any
    drawdown_isa: Any
    drawdown_gia: Any
    ann_income_1: Any
    ann_income_2: Any
    income_gross_1: Any
    income_gross_2: Any
    income_net: Any
    tax_1: Any
    tax_2: Any
    cgt: Any


@dataclasses.dataclass
class ResState:
    year: int
    ann_1: float
    ann_2: float
    sipp_uf_1: float
    sipp_uf_2: float
    sipp_df_1: float
    sipp_df_2: float
    sipp_delta_1: float
    sipp_delta_2: float
    contrib_1: float
    contrib_2: float
    tfc_1: float
    tfc_2: float
    lsa_ratio_1: float
    lsa_ratio_2: float
    isa: float
    isa_delta: float
    gia: float
    gia_delta: float
    income_gross_1: float
    income_gross_2: float
    cg: float
    income_net: float
    income_tax_1: float
    income_tax_2: float
    income_tax_rate_1: float
    income_tax_rate_2: float
    cgt: float
    cgt_rate: float


@dataclasses.dataclass
class Result:
    retirement_income_net: float = 0
    net_worth_start: float = 0
    net_worth_end: float = 0
    total_tax: float = 0
    data: list[ResState] = dataclasses.field(default_factory=list)
    ls_sipp_1: float = 0
    ls_sipp_2: float = 0
    ls_isa: float = 0
    ls_gia: float = 0


def income_tax_lp(prob, gross_income, income_tax_bands, factor=1.0):
    global uid
    total = 0
    tax = 0
    lbound = 0
    for ubound, rate in income_tax_bands:
        if ubound is None:
            ub = None
        else:
            ubound *= factor
            ub = ubound - lbound
        income_tax_band = lp.LpVariable(f'net_{uid}_{int(rate*1000)}', 0, ub)
        uid += 1
        total = total + income_tax_band
        tax = tax + income_tax_band * rate
        lbound = ubound
    prob += total == gross_income
    return tax


def uk_cgt_lp(prob, cg, cgt_rate, cgt_allowance):
    global uid
    cg_00 = lp.LpVariable(f'cgt_{uid}_00', 0, cgt_allowance)
    cg_xx = lp.LpVariable(f'cgt_{uid}_xx', 0)
    uid += 1
    prob += cg_00 + cg_xx == cg
    tax = cg_xx * cgt_rate
    return tax


def uk_tax_lp(prob, gross_income, cg, itt:UK.IncomeTaxThresholds, marriage_allowance:int=0):
    assert not isinstance(marriage_allowance, bool)
    global uid

    personal_allowance    = itt.income_tax_threshold_20 + marriage_allowance
    basic_rate_allowance  = itt.income_tax_threshold_40 - itt.income_tax_threshold_20
    higher_rate_allowance = itt.pa_limit - personal_allowance - basic_rate_allowance
    # FIXME: we can't model the 45% tax rate, as it's no longer convex

    income_pa                = lp.LpVariable(f'income_pa_{uid}', 0, personal_allowance)
    income_basic_rate        = lp.LpVariable(f'income_basic_rate_{uid}', 0, basic_rate_allowance)
    income_higher_rate  :lp.LpVariable|int
    income_adjusted_rate:lp.LpVariable|int
    if marriage_allowance == 0:
        income_higher_rate   = lp.LpVariable(f'income_higher_rate_{uid}', 0, higher_rate_allowance)
        income_adjusted_rate = lp.LpVariable(f'income_adjusted_rate_{uid}', 0)
    else:
        income_higher_rate   = 0
        income_adjusted_rate = 0

    prob += income_pa + income_basic_rate + income_higher_rate + income_adjusted_rate == gross_income

    income_tax = income_basic_rate    * 0.20 \
               + income_higher_rate   * 0.40 \
               + income_adjusted_rate * 0.60

    cg_allowance   = lp.LpVariable(f'cg_pa_{uid}', 0, UK.cgt_allowance)
    cg_basic_rate  = lp.LpVariable(f'cg_basic_rate_{uid}', 0)
    cg_higher_rate = lp.LpVariable(f'cg_higher_rate_{uid}', 0)

    prob += cg_allowance + cg_basic_rate + cg_higher_rate == cg

    prob += income_pa + income_basic_rate + cg_basic_rate <= itt.income_tax_threshold_40

    cgt_rate_basic, cgt_rate_higher = UK.cgt_rates

    cgt = cg_basic_rate  * cgt_rate_basic \
        + cg_higher_rate * cgt_rate_higher

    uid += 1

    return income_tax, cgt


def pt_income_tax_lp(prob, gross_income, factor=1.0):
    return income_tax_lp(prob, gross_income, PT.income_tax_bands, factor)


def normalize(x, ndigits=None):
    # https://bugs.python.org/issue45995
    return round(x, ndigits) + 0.0


# https://www.investopedia.com/terms/i/inflation_adjusted_return.asp
def inflation_ajusted_return(return_rate, inflation_rate):
    return (1.0 + return_rate) / (1.0 + inflation_rate) - 1.0


class DCP:
    """Defined Contribution Pension.

    Uncrystallised (uf) and drawdown (df) funds are each split into an equity
    sleeve and an ILG (Index-Linked Gilt) sleeve so that different growth rates
    can be applied.  All external-facing properties (uf, df, lsa) return the
    combined value across both sleeves.
    """

    def __init__(self, prob,
                 uf_eq, uf_ilg,
                 df_eq, df_ilg,
                 growth_rate_real_eq, growth_rate_real_ilg,
                 inflation_rate, lsa, nmpa):
        self.prob = prob

        self._uf_eq  = uf_eq
        self._uf_ilg = uf_ilg
        self._df_eq  = df_eq
        self._df_ilg = df_ilg
        self._df_cost_eq  = df_eq   # XXX
        self._df_cost_ilg = df_ilg  # XXX

        self.growth_rate_real_eq  = growth_rate_real_eq
        self.growth_rate_real_ilg = growth_rate_real_ilg
        self.inflation_rate = inflation_rate

        self.lsa = lsa
        self.nmpa = nmpa

    # ------------------------------------------------------------------
    # Combined views (read-only, used by the rest of the model)
    # ------------------------------------------------------------------
    @property
    def uf(self):
        return self._uf_eq + self._uf_ilg

    @uf.setter
    def uf(self, value):
        # Used by the eps stability nudge: apply proportionally across sleeves
        # so that the ratio is preserved.  We receive an expression of the form
        # old_uf * scalar, so we have to decompose it back.
        # Instead we keep the nudge on each sleeve individually (see drawdown).
        raise AttributeError("Set uf sleeves directly via _uf_eq / _uf_ilg")

    @property
    def df(self):
        return self._df_eq + self._df_ilg

    @property
    def df_cost(self):
        return self._df_cost_eq + self._df_cost_ilg

    # ------------------------------------------------------------------
    def contrib(self, contrib):
        # New contributions are assumed to follow the equity sleeve allocation
        # (identical to how a single-sleeve DCP worked before).
        self._uf_eq = self._uf_eq + contrib

    def drawdown(self, drawdown, age):
        # Flexible-Access Drawdown
        if age >= self.nmpa:
            tfc = self.tfc_lp(age)
        else:
            tfc = 0

        # Drawdown reduces the combined df; we split proportionally across sleeves
        # by drawing from equity first (a conservative, simple approach).
        self._df_eq  = self._df_eq  - drawdown
        self.prob   += self._df_eq  >= 0   # equity sleeve must stay non-negative

        # Grow each sleeve at its own real rate
        self._uf_eq  *= 1.0 + self.growth_rate_real_eq
        self._uf_ilg *= 1.0 + self.growth_rate_real_ilg

        self._df_cost_eq  *= 1.0 / (1.0 + self.inflation_rate)
        self._df_cost_ilg *= 1.0 / (1.0 + self.inflation_rate)

        self._df_eq  *= 1.0 + self.growth_rate_real_eq
        self._df_ilg *= 1.0 + self.growth_rate_real_ilg

        return tfc

    def tfc_lp(self, age):
        global uid
        crystalized_tfc = lp.LpVariable(f'crystalized_tfc_{uid}', 0)
        crystalized_inc = lp.LpVariable(f'crystalized_inc_{uid}', 0)
        uid += 1
        self.prob += 3*crystalized_tfc <= crystalized_inc
        self.lsa = self.lsa - crystalized_tfc
        self.prob += self.lsa >= 0
        # Crystallisation reduces equity sleeve first
        self._uf_eq = self._uf_eq - (crystalized_tfc + crystalized_inc)
        self.prob  += self._uf_eq >= 0
        self._df_eq = self._df_eq + crystalized_inc
        tfc = crystalized_tfc
        return tfc

    def apply_eps(self, eps):
        """Apply the stability-nudge epsilon to both sleeves."""
        self._uf_eq  *= 1.0 + eps
        self._uf_ilg *= 1.0 + eps


class ISA:
    """ISA wrapper holding separate equity and ILG sleeves."""

    def __init__(self, balance_eq, balance_ilg, growth_rate_real_eq, growth_rate_real_ilg):
        self._eq  = balance_eq
        self._ilg = balance_ilg
        self.growth_rate_real_eq  = growth_rate_real_eq
        self.growth_rate_real_ilg = growth_rate_real_ilg

    @property
    def value(self):
        return self._eq + self._ilg

    def withdraw(self, amount, prob):
        """Reduce ISA by *amount* (drawn from equity sleeve first)."""
        self._eq = self._eq - amount
        prob    += self._eq >= 0

    def grow(self):
        self._eq  *= 1.0 + self.growth_rate_real_eq
        self._ilg *= 1.0 + self.growth_rate_real_ilg

    def liquidate(self):
        """Return combined value and zero both sleeves (used on emigration)."""
        total     = self.value
        self._eq  = 0
        self._ilg = 0
        return total


class GIA:
    """GIA with separate equity and ILG sleeves, each tracked for CGT."""

    def __init__(self, prob,
                 balance_eq, balance_ilg,
                 growth_rate_eq, growth_rate_ilg,
                 inflation_rate):
        self.prob = prob

        # Each sleeve is a list of tranches (one per year, oldest last)
        self._assets_eq  = [balance_eq]
        self._assets_ilg = [balance_ilg]

        self.growth_rate_eq  = growth_rate_eq
        self.growth_rate_ilg = growth_rate_ilg
        self.inflation_rate  = inflation_rate
        self.growth_rate_real_eq  = inflation_ajusted_return(growth_rate_eq,  inflation_rate)
        self.growth_rate_real_ilg = inflation_ajusted_return(growth_rate_ilg, inflation_rate)

    def flow(self, inflation_adjusted=False):
        global uid

        total = 0
        gains = 0

        # Equity sleeve
        purchase_eq = lp.LpVariable(f'gia_purchase_eq_{uid}', 0)
        self._assets_eq.insert(0, purchase_eq)
        growth_rate_eq = self.growth_rate_real_eq if inflation_adjusted else self.growth_rate_eq
        for yr in range(1, len(self._assets_eq)):
            proceeds = lp.LpVariable(f'gia_proceeds_eq_{uid}_{yr}', 0)
            self._assets_eq[yr] = self._assets_eq[yr] - proceeds
            self.prob += self._assets_eq[yr] >= 0
            total = total + proceeds
            gains = gains + proceeds * (1.0 - (1.0 + growth_rate_eq) ** -yr)

        # ILG sleeve
        purchase_ilg = lp.LpVariable(f'gia_purchase_ilg_{uid}', 0)
        self._assets_ilg.insert(0, purchase_ilg)
        growth_rate_ilg = self.growth_rate_real_ilg if inflation_adjusted else self.growth_rate_ilg
        for yr in range(1, len(self._assets_ilg)):
            proceeds = lp.LpVariable(f'gia_proceeds_ilg_{uid}_{yr}', 0)
            self._assets_ilg[yr] = self._assets_ilg[yr] - proceeds
            self.prob += self._assets_ilg[yr] >= 0
            total = total + proceeds
            gains = gains + proceeds * (1.0 - (1.0 + growth_rate_ilg) ** -yr)

        # Net flow = total proceeds - new purchases
        purchase = purchase_eq + purchase_ilg
        for yr in range(0, len(self._assets_eq)):
            self._assets_eq[yr]  *= (1.0 + self.growth_rate_real_eq)  * (1.0 - eps)
        for yr in range(0, len(self._assets_ilg)):
            self._assets_ilg[yr] *= (1.0 + self.growth_rate_real_ilg) * (1.0 - eps)

        uid += 1

        return total - purchase, gains

    def value(self):
        total = 0
        for balance in self._assets_eq:
            total = total + balance
        for balance in self._assets_ilg:
            total = total + balance
        return total


# Introduce a tiny bias towards SIPPs uncrystalized funds and against
# GIAs to stabilize results, and prevent redundant money flows that
# arise when the optimal solution is not unique
eps = 2**-14


def solve(prob):
    prob.checkDuplicateVars()

    #prob.writeLP('retirement.lp')

    solvers = lp.listSolvers(onlyAvailable=True)
    if 'PULP_CBC_CMD' in solvers:
        solver = lp.PULP_CBC_CMD(msg=0)
    else:
        assert 'COIN_CMD' in solvers
        solver = lp.COIN_CMD(msg=0)

    status = prob.solve(solver)
    if status != lp.LpStatusOptimal:
        statusMsg = {
            lp.LpStatusNotSolved: "Not Solved",
            lp.LpStatusOptimal: "Optimal",
            lp.LpStatusInfeasible: "Infeasible",
            lp.LpStatusUnbounded: "Unbounded",
            lp.LpStatusUndefined: "Undefined",
        }.get(status, "Unexpected")
        raise ValueError(f"Failed to solve the problem ({statusMsg})")


def model(
        joint,
        dob_1,
        dob_2,
        present_year,
        retirement_year,
        inflation_rate,
        retirement_income_net,
        country,
        # SIPP uncrystallised — equity and ILG split
        sipp_1_eq,
        sipp_1_ilg,
        sipp_2_eq,
        sipp_2_ilg,
        # SIPP drawdown — equity and ILG split
        sipp_df_1_eq,
        sipp_df_1_ilg,
        sipp_df_2_eq,
        sipp_df_2_ilg,
        sipp_growth_rate_1,
        sipp_growth_rate_2,
        # ILG real growth rate for SIPPs (default: 0% real, i.e. inflation only)
        sipp_ilg_growth_rate_real_1=0.0,
        sipp_ilg_growth_rate_real_2=0.0,
        sipp_contrib_1=0,
        sipp_contrib_2=0,
        sipp_extra_contrib=False,
        db_payments_1=(),
        db_payments_2=(),
        db_ages_1=(),
        db_ages_2=(),
        lsa_ratio_1=1.0,
        lsa_ratio_2=1.0,
        # ISA — equity and ILG split
        isa_eq=0,
        isa_ilg=0,
        isa_growth_rate=0.0,
        isa_ilg_growth_rate_real=0.0,
        # GIA — equity and ILG split
        gia_eq=0,
        gia_ilg=0,
        gia_growth_rate=0.0,
        gia_ilg_growth_rate=0.0,
        misc_contrib=0,
        marginal_income_tax_1=0.0,
        marginal_income_tax_2=0.0,
        state_pension_years_1=0,
        state_pension_years_2=0,
        lump_sum=0,
        aa_1=UK.aa,
        aa_2=UK.aa,
        marriage_allowance: bool = False,
        end_age=90,
    ):
    """
    Parameters
    ----------
    sipp_1_eq, sipp_1_ilg
        Uncrystallised SIPP balance for person 1, split by asset class.
    sipp_2_eq, sipp_2_ilg
        Uncrystallised SIPP balance for person 2, split by asset class.
    sipp_df_1_eq, sipp_df_1_ilg
        Crystallised (drawdown) SIPP balance for person 1, split by asset class.
    sipp_df_2_eq, sipp_df_2_ilg
        Crystallised (drawdown) SIPP balance for person 2, split by asset class.
    sipp_ilg_growth_rate_real_1/2
        Real (after-inflation) annual growth rate for the ILG sleeve of each SIPP.
        Defaults to 0.0 (ILGs grow exactly with inflation, i.e. zero real return).
    isa_eq, isa_ilg
        ISA balance split by asset class.
    isa_ilg_growth_rate_real
        Real annual growth rate for the ISA ILG sleeve (default 0.0).
    gia_eq, gia_ilg
        GIA balance split by asset class.
    gia_ilg_growth_rate
        Nominal annual growth rate for the GIA ILG sleeve.
    """

    if joint:
        N = 2
        end_year = max(dob_1, dob_2) + end_age
    else:
        N = 1
        dob_2 = sys.maxsize
        sipp_2_eq = 0
        sipp_2_ilg = 0
        sipp_growth_rate_2 = 0
        sipp_contrib_2 = 0
        marginal_income_tax_2 = 0
        state_pension_years_2 = 0
        end_year = dob_1 + end_age

    assert N in (1, 2)
    income_ratio_1 = float(1    ) / float(N)
    income_ratio_2 = float(N - 1) / float(N)
    assert income_ratio_1 + income_ratio_2 == 1.0

    result = Result()

    sipp_1_total    = sipp_1_eq    + sipp_1_ilg
    sipp_2_total    = sipp_2_eq    + sipp_2_ilg
    sipp_df_1_total = sipp_df_1_eq + sipp_df_1_ilg
    sipp_df_2_total = sipp_df_2_eq + sipp_df_2_ilg
    isa_total       = isa_eq       + isa_ilg
    gia_total       = gia_eq       + gia_ilg

    result.net_worth_start = normalize(
        sipp_1_total + sipp_df_1_total +
        sipp_2_total + sipp_df_2_total +
        isa_total + gia_total, 2)

    assert state_pension_years_1 <= 35
    assert state_pension_years_2 <= 35
    state_pension_1 = UK.state_pension_full * state_pension_years_1 / 35
    state_pension_2 = UK.state_pension_full * state_pension_years_2 / 35

    lsa = UK.lsa

    sipp_growth_rate_real_1 = inflation_ajusted_return(sipp_growth_rate_1, inflation_rate)
    sipp_growth_rate_real_2 = inflation_ajusted_return(sipp_growth_rate_2, inflation_rate)
    isa_growth_rate_real    = inflation_ajusted_return(isa_growth_rate,    inflation_rate)

    assert sipp_contrib_1 <= UK.aa
    assert sipp_contrib_2 <= UK.aa

    if country == 'PT':
        gbpeur = float(hmrc.exchange_rate('EUR'))
    if country == 'JP':
        gbpjpy = float(hmrc.exchange_rate('JPY'))

    prob = lp.LpProblem("Retirement")

    max_income = retirement_income_net == 0
    if max_income:
        retirement_income_net = lp.LpVariable("income", 0)

    isa_allowance = UK.isa_allowance

    # XXX: Lump sum analysis
    if lump_sum:
        ls_sipp_1 = lp.LpVariable("ls_sipp_1", 0)
        ls_sipp_2 = lp.LpVariable("ls_sipp_2", 0, None if joint else 0)
        ls_isa    = lp.LpVariable("ls_isa", 0, N*isa_allowance)
        ls_gia    = lp.LpVariable("ls_gia", 0)
        prob += ls_sipp_1 + ls_sipp_2 + ls_isa + ls_gia == lump_sum
        ls_sipp_gross_1 = ls_sipp_1 * (1.0 / (1.0 - max(marginal_income_tax_1, 0.20)))
        ls_sipp_gross_2 = ls_sipp_2 * (1.0 / (1.0 - max(marginal_income_tax_2, 0.20)))
        prob += sipp_contrib_1 + ls_sipp_gross_1 <= aa_1
        prob += sipp_contrib_2 + ls_sipp_gross_2 <= aa_2
        # Lump-sum additions go into the equity sleeve
        sipp_1_eq = sipp_1_eq + ls_sipp_gross_1
        sipp_2_eq = sipp_2_eq + ls_sipp_gross_2
        isa_eq    = isa_eq    + ls_isa
        gia_eq    = gia_eq    + ls_gia * (1 - eps)

    nmpa_1 = UK.nmpa(dob_1)
    nmpa_2 = UK.nmpa(dob_2)

    lsa_1 = lsa * lsa_ratio_1
    lsa_2 = lsa * lsa_ratio_2

    sipp_1 = DCP(
        prob=prob,
        uf_eq=sipp_1_eq,   uf_ilg=sipp_1_ilg,
        df_eq=sipp_df_1_eq, df_ilg=sipp_df_1_ilg,
        growth_rate_real_eq=sipp_growth_rate_real_1,
        growth_rate_real_ilg=sipp_ilg_growth_rate_real_1,
        inflation_rate=inflation_rate,
        lsa=lsa_1, nmpa=nmpa_1,
    )
    sipp_2 = DCP(
        prob=prob,
        uf_eq=sipp_2_eq,   uf_ilg=sipp_2_ilg,
        df_eq=sipp_df_2_eq, df_ilg=sipp_df_2_ilg,
        growth_rate_real_eq=sipp_growth_rate_real_2,
        growth_rate_real_ilg=sipp_ilg_growth_rate_real_2,
        inflation_rate=inflation_rate,
        lsa=lsa_2, nmpa=nmpa_2,
    )

    isa_obj = ISA(
        balance_eq=isa_eq, balance_ilg=isa_ilg,
        growth_rate_real_eq=isa_growth_rate_real,
        growth_rate_real_ilg=isa_ilg_growth_rate_real,
    )

    gia_obj = GIA(
        prob=prob,
        balance_eq=gia_eq,  balance_ilg=gia_ilg,
        growth_rate_eq=gia_growth_rate, growth_rate_ilg=gia_ilg_growth_rate,
        inflation_rate=inflation_rate,
    )

    states = {}

    # XXX: SIPP contributions
    if sipp_extra_contrib:
        sipp_contrib_limit = UK.uiaa
        sipp_contrib_limit_1 = min(sipp_contrib_1 * 1.30, sipp_contrib_limit, UK.mpaa)
        sipp_contrib_limit_2 = min(sipp_contrib_2 * 1.30, sipp_contrib_limit, UK.mpaa)

    itt = UK.IncomeTaxThresholds()

    for yr in range(present_year, end_year):
        retirement = yr >= retirement_year
        uk_yr = not retirement or country == 'UK'

        age_1 = yr - dob_1
        age_2 = yr - dob_2

        # SIPP contributions
        if not retirement:
            contrib_1 = sipp_contrib_1
            contrib_2 = sipp_contrib_2
        else:
            contrib_1 = 0
            contrib_2 = 0
            if sipp_extra_contrib:
                if country == 'UK' or yr < retirement_year + 5:
                    if age_1 < 75:
                        contrib_1 = lp.LpVariable(f'contrib_1@{yr}', 0, sipp_contrib_limit_1)
                    if age_2 < 75 and joint:
                        contrib_2 = lp.LpVariable(f'contrib_2@{yr}', 0, sipp_contrib_limit_2)

        sipp_1.contrib(contrib_1)
        sipp_2.contrib(contrib_2)

        drawdown_1 = lp.LpVariable(f'dd_1@{yr}', 0) if age_1 >= nmpa_1 and (retirement or sipp_contrib_1 <= UK.mpaa) else 0
        drawdown_2 = lp.LpVariable(f'dd_2@{yr}', 0) if age_2 >= nmpa_2 and (retirement or sipp_contrib_2 <= UK.mpaa) else 0

        tfc_1 = sipp_1.drawdown(drawdown_1, age=age_1)
        tfc_2 = sipp_2.drawdown(drawdown_2, age=age_2)

        drawdown_isa: lp.LpVariable | int
        if uk_yr:
            isa_allowance_yr = isa_allowance * N
            # Negative drawdown_isa means new money going into the ISA (Bed & ISA)
            drawdown_isa = lp.LpVariable(f'dd_isa@{yr}', -isa_allowance_yr)
            isa_obj.withdraw(drawdown_isa, prob)
            isa_obj.grow()
        elif yr == retirement_year:
            drawdown_isa = isa_obj.liquidate()
        else:
            drawdown_isa = 0
            assert isa_obj.value == 0

        if yr < 2030:
            isa_allowance /= 1.0 + inflation_rate

        drawdown_gia, cg = gia_obj.flow(not uk_yr and country == 'PT')

        # Stability-nudge epsilon applied to both SIPP sleeves
        sipp_1.apply_eps(eps)
        sipp_2.apply_eps(eps)

        # State pension
        spa_1 = UK.state_pension_age(dob_1)
        spa_2 = UK.state_pension_age(dob_2)
        income_state_1 = state_pension_1 if age_1 >= spa_1 else 0
        income_state_2 = state_pension_2 if age_2 >= spa_2 else 0
        if country not in ('UK', 'PT'):
            income_state_1 *= (1.0/(1.0 + inflation_rate)) ** max(age_1 - spa_1, 0)
            income_state_2 *= (1.0/(1.0 + inflation_rate)) ** max(age_2 - spa_2, 0)

        # DB pensions
        ann_income_1 = income_state_1
        for pay, age in zip(db_payments_1, db_ages_1):
            ann_income_1 += pay if age_1 >= age else 0
        ann_income_2 = income_state_2
        for pay, age in zip(db_payments_2, db_ages_2):
            ann_income_2 += pay if age_2 >= age else 0

        income_gross_1 = ann_income_1 + drawdown_1
        income_gross_2 = ann_income_2 + drawdown_2

        isa_value = isa_obj.value

        if uk_yr:
            cg_2: lp.LpVariable | int
            if joint:
                cg_1 = lp.LpVariable(f'cg_1@{yr}', 0)
                cg_2 = lp.LpVariable(f'cg_2@{yr}', 0)
                prob += cg_1 + cg_2 == cg
            else:
                cg_1 = cg
                cg_2 = 0
            if yr < retirement_year:
                marginal_income_tax_to_base_salary = {
                    0.00: 0,
                    0.20: itt.income_tax_threshold_20,
                    0.40: itt.income_tax_threshold_40,
                    0.45: itt.income_tax_threshold_45,
                }
                base_salary_1 = marginal_income_tax_to_base_salary[marginal_income_tax_1]
                base_salary_2 = marginal_income_tax_to_base_salary[marginal_income_tax_2]
                base_income_tax_1, _ = UK.tax(itt, base_salary_1, 0)
                base_income_tax_2, _ = UK.tax(itt, base_salary_2, 0)
                tax_1, cgt_1 = uk_tax_lp(prob, base_salary_1 + income_gross_1, cg_1, itt)
                tax_2, cgt_2 = uk_tax_lp(prob, base_salary_2 + income_gross_2, cg_2, itt)
                tax_1 = tax_1 - base_income_tax_1
                tax_2 = tax_2 - base_income_tax_2
            else:
                if marriage_allowance and ann_income_2 <= itt.income_tax_threshold_20:
                    prob += income_gross_1 <= itt.income_tax_threshold_40
                    prob += income_gross_2 <= itt.income_tax_threshold_20
                    tax_1, cgt_1 = uk_tax_lp(prob, income_gross_1, cg_1, itt, marriage_allowance=itt.marriage_allowance)
                    tax_2, cgt_2 = uk_tax_lp(prob, income_gross_2, cg_2, itt, marriage_allowance=-itt.marriage_allowance)
                else:
                    tax_1, cgt_1 = uk_tax_lp(prob, income_gross_1, cg_1, itt)
                    tax_2, cgt_2 = uk_tax_lp(prob, income_gross_2, cg_2, itt)
            cgt = cgt_1 + cgt_2
        elif country == 'PT':
            income_gross = (income_gross_1 + tfc_1 +
                            income_gross_2 + tfc_2)
            tax = pt_income_tax_lp(prob, income_gross, factor=N/gbpeur)
            cgt = cg * PT.cgt_rate
            income_gross_1 = income_gross * income_ratio_1
            income_gross_2 = income_gross * income_ratio_2
            tax_1 = tax * income_ratio_1
            tax_2 = tax * income_ratio_2
        elif country == 'JP':
            income_gross_1 = income_gross_1 + tfc_1
            income_gross_2 = income_gross_2 + tfc_2
            tax_1 = income_tax_lp(prob, income_gross_1, JP.income_tax_bands, 1/gbpjpy)
            tax_2 = income_tax_lp(prob, income_gross_2, JP.income_tax_bands, 1/gbpjpy)
            cgt = cg * JP.cgt_rate
        else:
            raise NotImplementedError

        incomings = income_gross_1 + income_gross_2 + drawdown_isa + drawdown_gia
        if uk_yr:
            incomings = incomings + tfc_1 + tfc_2
        if yr < retirement_year:
            incomings = incomings + misc_contrib
        outgoings = tax_1 + tax_2 + cgt
        if yr >= retirement_year:
            income_net = retirement_income_net
            outgoings = outgoings + retirement_income_net
            outgoings = outgoings + contrib_1 * 0.80
            outgoings = outgoings + contrib_2 * 0.80
        else:
            income_net = 0

        prob += incomings == outgoings

        states[yr] = LPState(
            sipp_uf_1=sipp_1.uf,
            sipp_uf_2=sipp_2.uf,
            sipp_df_1=sipp_1.df,
            sipp_df_2=sipp_2.df,
            contrib_1=contrib_1,
            contrib_2=contrib_2,
            tfc_1=tfc_1,
            tfc_2=tfc_2,
            lsa_1=sipp_1.lsa,
            lsa_2=sipp_2.lsa,
            isa=isa_value,
            gia=gia_obj.value(),
            cg=cg,
            drawdown_1=drawdown_1,
            drawdown_2=drawdown_2,
            drawdown_isa=drawdown_isa,
            drawdown_gia=drawdown_gia,
            ann_income_1=ann_income_1,
            ann_income_2=ann_income_2,
            income_gross_1=income_gross_1,
            income_gross_2=income_gross_2,
            income_net=income_net,
            tax_1=tax_1,
            tax_2=tax_2,
            cgt=cgt,
        )

        # Freeze income tax thresholds until 2031
        if yr < 2031:
            itt.income_tax_threshold_20 = round(itt.income_tax_threshold_20 / (1.0 + inflation_rate))
            itt.income_tax_threshold_40 = round(itt.income_tax_threshold_40 / (1.0 + inflation_rate))
            itt.income_tax_threshold_45 = round(itt.income_tax_threshold_45 / (1.0 + inflation_rate))
            itt.pa_limit                = round(itt.pa_limit                / (1.0 + inflation_rate))
            itt.marriage_allowance      = round(itt.marriage_allowance      / (1.0 + inflation_rate))

    if max_income:
        prob.setObjective(-retirement_income_net)
    else:
        net_worth = sipp_1.uf + sipp_2.uf + sipp_1.df + sipp_2.df + isa_obj.value + gia_obj.value()
        prob.setObjective(-net_worth)

    solve(prob)

    result.net_worth_end = normalize(
        lp.value(sipp_1.uf + sipp_1.df + sipp_2.uf + sipp_2.df + isa_obj.value + gia_obj.value()), 0)

    if max_income:
        result.retirement_income_net = lp.value(retirement_income_net)
    else:
        result.retirement_income_net = retirement_income_net

    if max_income:
        retirement_income_net = lp.value(retirement_income_net)

    for yr in range(present_year, end_year):
        s = states[yr]
        if verbosity > 1:
            if yr == retirement_year:
                for n, v in dataclasses.asdict(s).items():
                    print(f' {n} = {lp.value(v)}')

        contrib_1 = lp.value(s.contrib_1)
        contrib_2 = lp.value(s.contrib_2)

        sipp_uf_1 = lp.value(s.sipp_uf_1)
        sipp_uf_2 = lp.value(s.sipp_uf_2)

        sipp_df_1 = lp.value(s.sipp_df_1)
        sipp_df_2 = lp.value(s.sipp_df_2)

        tfc_1 = lp.value(s.tfc_1)
        tfc_2 = lp.value(s.tfc_2)

        lsa_1 = lp.value(s.lsa_1)
        lsa_2 = lp.value(s.lsa_2)

        isa = lp.value(s.isa)
        gia = lp.value(s.gia)

        drawdown_1   = lp.value(s.drawdown_1)
        drawdown_2   = lp.value(s.drawdown_2)
        drawdown_isa = lp.value(s.drawdown_isa)
        drawdown_gia = lp.value(s.drawdown_gia)

        cg = lp.value(s.cg)

        ann_income_1 = s.ann_income_1
        ann_income_2 = s.ann_income_2

        income_gross_1 = lp.value(s.income_gross_1)
        income_gross_2 = lp.value(s.income_gross_2)

        income_net = lp.value(s.income_net)

        tax_1 = lp.value(s.tax_1)
        tax_2 = lp.value(s.tax_2)
        cgt = lp.value(s.cgt)

        tax_rate_1 = tax_1 / max(income_gross_1, 1)
        tax_rate_2 = tax_2 / max(income_gross_2, 1)
        cgt_rate   = cgt   / max(cg, 1)

        if verbosity > 0:
            print(' '.join((
                    '%4u:',
                    'DB %5.0f',
                    'SIPP1 [%7.0f %7.0f] (%6.0f %7.0f) %5.1f%%',
                    'SIPP2 [%7.0f %7.0f] (%6.0f %7.0f) %5.1f%%',
                    'ISA %7.0f (%7.0f)',
                    'GIA %7.0f (%7.0f)',
                    'Inc Gr %6.0f %6.0f Nt %6.0f',
                    'Tax %6.0f %4.1f%% %6.0f %4.1f%% %6.0f %4.1f%%'
                )) % (
                    yr,
                    ann_income_1 + ann_income_2,
                    sipp_uf_1, sipp_df_1, contrib_1, -tfc_1 - drawdown_1, 100*lsa_1/lsa,
                    sipp_uf_2, sipp_df_2, contrib_2, -tfc_2 - drawdown_2, 100*lsa_2/lsa,
                    isa, -drawdown_isa,
                    gia, -drawdown_gia,
                    income_gross_1, income_gross_2, income_net,
                    tax_1, 100 * tax_rate_1,
                    tax_2, 100 * tax_rate_2,
                    cgt, 100 * cgt_rate
                ))
        tax = tax_1 + tax_2 + cgt
        result.total_tax += tax

        rs = ResState(
            year=yr,
            ann_1=ann_income_1,
            ann_2=ann_income_2,
            sipp_uf_1=normalize(sipp_uf_1, 2),
            sipp_uf_2=normalize(sipp_uf_2, 2),
            sipp_df_1=normalize(sipp_df_1, 2),
            sipp_df_2=normalize(sipp_df_2, 2),
            contrib_1=normalize(contrib_1, 2),
            contrib_2=normalize(contrib_2, 2),
            sipp_delta_1=normalize(-drawdown_1, 2),
            sipp_delta_2=normalize(-drawdown_2, 2),
            tfc_1=normalize(tfc_1, 2),
            tfc_2=normalize(tfc_2, 2),
            lsa_ratio_1=normalize(lsa_1/lsa, 4),
            lsa_ratio_2=normalize(lsa_2/lsa, 4),
            isa=isa,
            isa_delta=normalize(-drawdown_isa, 2),
            gia=normalize(gia, 2),
            gia_delta=normalize(-drawdown_gia, 2),
            income_gross_1=income_gross_1,
            income_gross_2=income_gross_2,
            cg=cg,
            income_net=income_net,
            income_tax_1=tax_1,
            income_tax_2=tax_2,
            income_tax_rate_1=tax_rate_1,
            income_tax_rate_2=tax_rate_2,
            cgt=cgt,
            cgt_rate=cgt_rate
        )

        result.data.append(rs)

    if lump_sum:
        result.ls_sipp_1 = lp.value(ls_sipp_1)
        result.ls_sipp_2 = lp.value(ls_sipp_2)
        result.ls_isa    = lp.value(ls_isa)
        result.ls_gia    = lp.value(ls_gia)

    return result


# Columns headers for DataFrame
column_headers = {
    'year': 'Year',

    'ann_1': 'A1',
    'ann_2': 'A2',

    'sipp_uf_1': 'UF1',
    'contrib_1': '(+\u0394)',
    'tfc_1': 'TFC1',
    'sipp_df_1': 'DF1',
    'sipp_delta_1': '(\u0394)',
    'lsa_ratio_1': 'LSA1',
    'sipp_uf_2': 'UF2',
    'contrib_2': '(+\u0394)',
    'tfc_2': 'TFC2',
    'sipp_df_2': 'DF2',
    'sipp_delta_2': '(\u0394)',
    'lsa_ratio_2': 'LSA2',

    'isa': 'ISAs',
    'isa_delta': '(\u0394)',
    'gia': 'GIA',
    'gia_delta': '(\u0394)',
    'income_gross_1': 'GI1',
    'income_gross_2': 'GI2',
    'cg': 'CG',
    'income_net': 'NI',
    'income_tax_1': 'IT1',
    'income_tax_rate_1': '(%)',
    'income_tax_2': 'IT2',
    'income_tax_rate_2': '(%)',
    'cgt': 'CGT',
    'cgt_rate': '(%)',
}


def dataframe(data):
    import pandas as pd

    df = pd.DataFrame(data)

    return df


def run(params):

    result = model(**params)

    df = dataframe(result.data)

    float_format = '{:5.0f}'.format
    perc_format = '{:5.1%}'.format
    delta_format = '{:+4.0f}'.format
    formatters = {
        'year': '{:}'.format,
        'sipp_delta_1': delta_format,
        'sipp_delta_2': delta_format,
        'contrib_1': delta_format,
        'contrib_2': delta_format,
        'isa_delta': delta_format,
        'gia_delta': delta_format,
        'lsa_ratio_1':  perc_format,
        'lsa_ratio_2':  perc_format,
        'income_tax_rate_1': perc_format,
        'income_tax_rate_2': perc_format,
        'cgt_rate':     perc_format,
    }

    print(df.to_string(
        index=False,
        columns=column_headers.keys(),
        header=column_headers.values(),
        justify='center',
        float_format=float_format,
        formatters=formatters
    ))

    print(f"Start net worth:       {result.net_worth_start:10,.0f}")
    print(f"Retirement net income: {result.retirement_income_net:10,.0f}")

    country = params['country']
    if country == 'PT':
        gbpeur = float(hmrc.exchange_rate('EUR'))
        print(f"Retirement net income: {result.retirement_income_net*gbpeur:10,.0f} EUR")
    if country == 'JP':
        gbpjpy = float(hmrc.exchange_rate('JPY'))
        print(f"Retirement net income: {result.retirement_income_net*gbpjpy:10,.0f} JPY")

    print(f"End net worth:         {result.net_worth_end:10,.0f}")
    print(f"Total tax:             {result.total_tax:10,.0f}")

    if result.ls_sipp_1 + result.ls_sipp_2 + result.ls_isa + result.ls_gia:
        print("Lump sump allocation:")
        print(f"  SIPP1: {result.ls_sipp_1:8.0f}")
        print(f"  SIPP2: {result.ls_sipp_2:8.0f}")
        print(f"  ISA:   {result.ls_isa:8.0f}")
        print(f"  GIA:   {result.ls_gia:8.0f}")
