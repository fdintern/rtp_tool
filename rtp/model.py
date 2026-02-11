#!/usr/bin/env python3
#
# Enhanced Retirement Tax Planning Model
# Based on LateGenXer's RTP model, extended with:
#   - Asset location optimisation (ILGs vs equities across wrappers)
#   - ILG CGT exemption in GIA
#   - IHT as terminal cost
#   - Oversized SIPP detection and strategy adjustment
#
# Copyright (c) 2023 LateGenXer (original)
# Enhancements for Fowler Drew by Claude/Anthropic
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#

import dataclasses
import sys
import math
from typing import Any, Optional

import lp

from data import hmrc

import tax.uk as UK
import tax.pt as PT
import tax.jp as JP


verbosity = 0
uid = 0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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
    gia_ilg: Any          # GIA index-linked gilts portion
    gia_eq: Any           # GIA equities portion
    cg: Any               # Only from equities (ILGs exempt)
    drawdown_1: Any
    drawdown_2: Any
    drawdown_isa: Any
    drawdown_gia_ilg: Any # Separate drawdown tracking
    drawdown_gia_eq: Any
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
    gia_ilg: float
    gia_ilg_delta: float
    gia_eq: float
    gia_eq_delta: float
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
    iht_estate: float     # Running IHT-liable estate value


@dataclasses.dataclass
class Result:
    retirement_income_net: float = 0
    net_worth_start: float = 0
    net_worth_end: float = 0
    total_tax: float = 0
    total_iht: float = 0
    data: list[ResState] = dataclasses.field(default_factory=list)
    ls_sipp_1: float = 0
    ls_sipp_2: float = 0
    ls_isa: float = 0
    ls_gia: float = 0
    sipp_oversized_1: bool = False
    sipp_oversized_2: bool = False
    asset_location_note: str = ""


# ---------------------------------------------------------------------------
# Tax helper functions (unchanged from original)
# ---------------------------------------------------------------------------

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


def uk_tax_lp(prob, gross_income, cg, itt: UK.IncomeTaxThresholds, marriage_allowance: int = 0):
    assert not isinstance(marriage_allowance, bool)
    global uid

    personal_allowance    = itt.income_tax_threshold_20 + marriage_allowance
    basic_rate_allowance  = itt.income_tax_threshold_40 - itt.income_tax_threshold_20
    higher_rate_allowance = itt.pa_limit - personal_allowance - basic_rate_allowance

    income_pa           = lp.LpVariable(f'income_pa_{uid}', 0, personal_allowance)
    income_basic_rate   = lp.LpVariable(f'income_basic_rate_{uid}', 0, basic_rate_allowance)
    income_higher_rate: lp.LpVariable | int
    income_adjusted_rate: lp.LpVariable | int
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


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def normalize(x, ndigits=None):
    return round(x, ndigits) + 0.0


def inflation_adjusted_return(return_rate, inflation_rate):
    return (1.0 + return_rate) / (1.0 + inflation_rate) - 1.0


# ---------------------------------------------------------------------------
# SIPP oversized detection
# ---------------------------------------------------------------------------

def is_sipp_oversized(sipp_total, guaranteed_income, retirement_years, inflation_rate, itt):
    """
    Determine whether a SIPP is too large to draw down within the basic-rate
    band over a realistic retirement horizon.

    Returns True if the SIPP annuitised over retirement_years, plus guaranteed
    income, would consistently exceed the higher-rate threshold.
    """
    if retirement_years <= 0 or sipp_total <= 0:
        return False

    # Real annual drawdown if spread evenly
    annual_drawdown = sipp_total / retirement_years

    # Total annual taxable income
    total_taxable = annual_drawdown + guaranteed_income

    # Oversized if total taxable exceeds the basic-rate band ceiling
    # Use a margin — if it's within 10% of the threshold, it's borderline
    return total_taxable > itt.income_tax_threshold_40 * 0.90


def recommend_asset_location(
    sipp_total_1, sipp_total_2,
    guaranteed_income_1, guaranteed_income_2,
    retirement_years_1, retirement_years_2,
    inflation_rate, itt,
    ilg_pct, post_2027_pension_iht
):
    """
    Recommend asset location strategy based on SIPP size analysis.

    Returns:
        sipp_ilg_pct_1: recommended % of SIPP 1 in ILGs
        sipp_ilg_pct_2: recommended % of SIPP 2 in ILGs
        note: explanation string
    """
    oversized_1 = is_sipp_oversized(
        sipp_total_1, guaranteed_income_1, retirement_years_1, inflation_rate, itt
    )
    oversized_2 = is_sipp_oversized(
        sipp_total_2, guaranteed_income_2, retirement_years_2, inflation_rate, itt
    )

    notes = []

    if oversized_1:
        # Large SIPP: hold ILGs in SIPP to slow growth, equities in GIA
        sipp_ilg_pct_1 = min(ilg_pct * 2.0, 1.0)  # Overweight ILGs in SIPP
        notes.append(
            f"Person 1 SIPP is oversized ({sipp_total_1:,.0f}). "
            f"Recommending {sipp_ilg_pct_1:.0%} ILGs in SIPP to slow growth; "
            f"equities in GIA where CGT extinguishes on death."
        )
    else:
        # Normal SIPP: conventional wisdom — equities in SIPP for tax-free growth
        sipp_ilg_pct_1 = max(ilg_pct * 0.5, 0.0)  # Underweight ILGs in SIPP
        notes.append(
            f"Person 1 SIPP ({sipp_total_1:,.0f}) can be drawn down at basic rate. "
            f"Equities in SIPP for tax-sheltered growth."
        )

    if oversized_2:
        sipp_ilg_pct_2 = min(ilg_pct * 2.0, 1.0)
        notes.append(
            f"Person 2 SIPP is oversized ({sipp_total_2:,.0f}). "
            f"Recommending {sipp_ilg_pct_2:.0%} ILGs in SIPP; equities in GIA."
        )
    else:
        sipp_ilg_pct_2 = max(ilg_pct * 0.5, 0.0)
        notes.append(
            f"Person 2 SIPP ({sipp_total_2:,.0f}) can be drawn down at basic rate. "
            f"Equities in SIPP for tax-sheltered growth."
        )

    if post_2027_pension_iht:
        notes.append(
            "Post-2027 pension IHT: SIPP loses IHT exemption. "
            "GIA equities with death uplift become relatively more attractive."
        )

    return sipp_ilg_pct_1, sipp_ilg_pct_2, " | ".join(notes)


# ---------------------------------------------------------------------------
# Defined Contribution Pension (unchanged from original)
# ---------------------------------------------------------------------------

class DCP:
    """Defined Contribution Pension."""

    def __init__(self, prob, uf, df, growth_rate_real, inflation_rate, lsa, nmpa):
        self.prob = prob
        self.uf = uf
        self.df = df
        self.df_cost = df
        self.growth_rate_real = growth_rate_real
        self.inflation_rate = inflation_rate
        self.lsa = lsa
        self.nmpa = nmpa

    def contrib(self, contrib):
        self.uf = self.uf + contrib

    def drawdown(self, drawdown, age):
        if age >= self.nmpa:
            tfc = self.tfc_lp(age)
        else:
            tfc = 0
        self.df = self.df - drawdown
        self.prob += self.df >= 0
        self.uf *= 1.0 + self.growth_rate_real
        self.df_cost *= 1.0 / (1.0 + self.inflation_rate)
        self.df *= 1.0 + self.growth_rate_real
        return tfc

    def tfc_lp(self, age):
        global uid
        crystalized_tfc = lp.LpVariable(f'crystalized_tfc_{uid}', 0)
        crystalized_inc = lp.LpVariable(f'crystalized_inc_{uid}', 0)
        uid += 1
        self.prob += 3 * crystalized_tfc <= crystalized_inc
        self.lsa = self.lsa - crystalized_tfc
        self.prob += self.lsa >= 0
        self.uf = self.uf - (crystalized_tfc + crystalized_inc)
        self.prob += self.uf >= 0
        self.df = self.df + crystalized_inc
        tfc = crystalized_tfc
        return tfc


# ---------------------------------------------------------------------------
# Enhanced GIA — separate ILG and equity tracking
# ---------------------------------------------------------------------------

class GIA_ILG:
    """
    GIA holding index-linked gilts.
    Key property: NO CGT on disposal (qualifying government securities).
    Only tax cost is income tax on real coupon (modelled as ilg_income_rate).
    """

    def __init__(self, prob, balance, growth_rate_real, ilg_income_rate=0.005):
        self.prob = prob
        self.balance = balance
        self.growth_rate_real = growth_rate_real
        self.ilg_income_rate = ilg_income_rate  # Real coupon yield

    def flow(self):
        """
        Withdraw from ILG holdings. No capital gains arise.
        Returns (net_flow, taxable_income_from_coupons)
        """
        global uid
        drawdown = lp.LpVariable(f'gia_ilg_dd_{uid}', 0)
        self.balance = self.balance - drawdown
        self.prob += self.balance >= 0

        # Coupon income is taxable (but small for ILGs)
        coupon_income = self.balance * self.ilg_income_rate

        self.balance *= 1.0 + self.growth_rate_real
        uid += 1
        return drawdown, coupon_income

    def value(self):
        return self.balance


class GIA_Equity:
    """
    GIA holding equities.
    Subject to CGT on disposal, but gains extinguish on death.
    Also subject to dividend tax on income.
    """

    def __init__(self, prob, balance, growth_rate, inflation_rate):
        self.prob = prob
        self.assets = [balance]
        self.growth_rate = growth_rate
        self.inflation_rate = inflation_rate
        self.growth_rate_real = inflation_adjusted_return(self.growth_rate, self.inflation_rate)

    def flow(self, inflation_adjusted=False):
        global uid
        total = 0
        gains = 0

        purchase = lp.LpVariable(f'gia_eq_purchase_{uid}', 0)
        self.assets.insert(0, purchase)

        growth_rate = self.growth_rate_real if inflation_adjusted else self.growth_rate

        for yr in range(1, len(self.assets)):
            proceeds = lp.LpVariable(f'gia_eq_proceeds_{uid}_{yr}', 0)
            self.assets[yr] = self.assets[yr] - proceeds
            self.prob += self.assets[yr] >= 0
            total = total + proceeds
            gains = gains + proceeds * (1.0 - (1.0 + growth_rate) ** -yr)

        for yr in range(0, len(self.assets)):
            self.assets[yr] *= (1.0 + self.growth_rate_real) * (1.0 - eps)

        uid += 1
        return total - purchase, gains

    def value(self):
        total = 0
        for balance in self.assets:
            total = total + balance
        return total

    def unrealised_gains(self):
        """Estimate unrealised gains for IHT/death uplift analysis."""
        total_gains = 0
        for yr in range(len(self.assets)):
            gain_fraction = 1.0 - (1.0 + self.growth_rate) ** -(yr + 1)
            total_gains = total_gains + self.assets[yr] * gain_fraction
        return total_gains


# Stabilisation epsilon (unchanged)
eps = 2**-14


# ---------------------------------------------------------------------------
# Solver (unchanged)
# ---------------------------------------------------------------------------

def solve(prob):
    prob.checkDuplicateVars()
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


# ---------------------------------------------------------------------------
# IHT calculation
# ---------------------------------------------------------------------------

def compute_iht(estate_value, nil_rate_band=325_000, rnrb=175_000, has_property=False):
    """
    Compute IHT liability on death.
    Simplified: uses nil-rate band + residence nil-rate band if applicable.
    For married couples, assume transferable nil-rate band on second death.
    """
    threshold = nil_rate_band
    if has_property:
        threshold += rnrb
    taxable = max(estate_value - threshold, 0)
    return taxable * 0.40


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

def model(
        joint,
        dob_1,
        dob_2,
        present_year,
        retirement_year,
        inflation_rate,
        retirement_income_net,
        country,
        sipp_1,
        sipp_2,
        sipp_df_1,
        sipp_df_2,
        sipp_growth_rate_1,
        sipp_growth_rate_2,
        sipp_contrib_1,
        sipp_contrib_2,
        sipp_extra_contrib,
        db_payments_1,
        db_payments_2,
        db_ages_1,
        db_ages_2,
        lsa_ratio_1,
        lsa_ratio_2,
        isa,
        isa_growth_rate,
        gia,
        gia_growth_rate,
        misc_contrib,
        marginal_income_tax_1,
        marginal_income_tax_2,
        state_pension_years_1,
        state_pension_years_2,
        lump_sum,
        aa_1,
        aa_2,
        marriage_allowance: bool,
        end_age,
        # === NEW PARAMETERS ===
        ilg_pct: float = 0.50,            # Portfolio % in ILGs (0.0 to 1.0)
        ilg_growth_rate: float = 0.005,    # Real return on ILGs
        ilg_income_rate: float = 0.005,    # ILG coupon yield (taxable in GIA)
        equity_growth_rate: Optional[float] = None,  # If None, uses gia_growth_rate
        auto_asset_location: bool = True,  # Auto-optimise asset location
        post_2027_pension_iht: bool = True,  # Model pension IHT from 2027
        iht_nil_rate_band: float = 325_000,
        iht_rnrb: float = 175_000,
        has_property: bool = True,
        iht_weight: float = 0.1,  # Weight of IHT in objective (0=ignore, 1=full)
    ):

    if equity_growth_rate is None:
        equity_growth_rate = gia_growth_rate

    if joint:
        N = 2
        end_year = max(dob_1, dob_2) + end_age
    else:
        N = 1
        dob_2 = sys.maxsize
        sipp_2 = 0
        sipp_growth_rate_2 = 0
        sipp_contrib_2 = 0
        marginal_income_tax_2 = 0
        state_pension_years_2 = 0
        end_year = dob_1 + end_age

    assert N in (1, 2)
    income_ratio_1 = float(1) / float(N)
    income_ratio_2 = float(N - 1) / float(N)

    result = Result()

    # -----------------------------------------------------------------------
    # SIPP oversized analysis & asset location recommendation
    # -----------------------------------------------------------------------
    itt = UK.IncomeTaxThresholds()

    state_pension_1 = UK.state_pension_full * state_pension_years_1 / 35
    state_pension_2 = UK.state_pension_full * state_pension_years_2 / 35

    retirement_years_1 = end_year - retirement_year
    retirement_years_2 = end_year - retirement_year

    guaranteed_income_1 = state_pension_1 + sum(db_payments_1)
    guaranteed_income_2 = state_pension_2 + sum(db_payments_2)

    sipp_total_1 = sipp_1 + sipp_df_1
    sipp_total_2 = sipp_2 + sipp_df_2

    result.sipp_oversized_1 = is_sipp_oversized(
        sipp_total_1, guaranteed_income_1, retirement_years_1, inflation_rate, itt
    )
    result.sipp_oversized_2 = is_sipp_oversized(
        sipp_total_2, guaranteed_income_2, retirement_years_2, inflation_rate, itt
    )

    if auto_asset_location:
        sipp_ilg_pct_1, sipp_ilg_pct_2, location_note = recommend_asset_location(
            sipp_total_1, sipp_total_2,
            guaranteed_income_1, guaranteed_income_2,
            retirement_years_1, retirement_years_2,
            inflation_rate, itt,
            ilg_pct, post_2027_pension_iht
        )
        result.asset_location_note = location_note
    else:
        # Default: portfolio-proportionate allocation across all wrappers
        sipp_ilg_pct_1 = ilg_pct
        sipp_ilg_pct_2 = ilg_pct

    # -----------------------------------------------------------------------
    # Compute blended SIPP growth rates based on asset location
    # -----------------------------------------------------------------------
    # SIPP growth = weighted average of ILG and equity returns
    # This reflects the asset location decision
    eq_growth_rate_real = inflation_adjusted_return(equity_growth_rate, inflation_rate)
    ilg_growth_rate_real = ilg_growth_rate  # Already real

    sipp_growth_rate_real_1 = (
        sipp_ilg_pct_1 * ilg_growth_rate_real +
        (1 - sipp_ilg_pct_1) * eq_growth_rate_real
    )
    sipp_growth_rate_real_2 = (
        sipp_ilg_pct_2 * ilg_growth_rate_real +
        (1 - sipp_ilg_pct_2) * eq_growth_rate_real
    )

    isa_growth_rate_real = inflation_adjusted_return(isa_growth_rate, inflation_rate)

    # -----------------------------------------------------------------------
    # Split GIA into ILG and equity portions
    # -----------------------------------------------------------------------
    # GIA asset location: inverse of SIPP (what's not in SIPP goes to GIA)
    # Total portfolio ILG allocation = ilg_pct
    # ILGs in SIPP = sipp_total * sipp_ilg_pct
    # ILGs in GIA = total_ilg - ilgs_in_sipp - ilgs_in_isa

    total_portfolio = sipp_total_1 + sipp_total_2 + isa + gia
    total_ilg = total_portfolio * ilg_pct
    total_eq  = total_portfolio * (1 - ilg_pct)

    ilg_in_sipp = sipp_total_1 * sipp_ilg_pct_1 + sipp_total_2 * sipp_ilg_pct_2
    # ISA: proportionate to portfolio split (ISA wrapper doesn't matter for tax)
    ilg_in_isa = isa * ilg_pct

    ilg_in_gia = max(total_ilg - ilg_in_sipp - ilg_in_isa, 0)
    eq_in_gia = max(gia - ilg_in_gia, 0)

    if verbosity > 0:
        print(f"Asset location:")
        print(f"  SIPP 1: {sipp_ilg_pct_1:.0%} ILGs, {1-sipp_ilg_pct_1:.0%} equities")
        print(f"  SIPP 2: {sipp_ilg_pct_2:.0%} ILGs, {1-sipp_ilg_pct_2:.0%} equities")
        print(f"  GIA: {ilg_in_gia:,.0f} ILGs + {eq_in_gia:,.0f} equities = {gia:,.0f}")
        print(f"  SIPP 1 oversized: {result.sipp_oversized_1}")
        print(f"  SIPP 2 oversized: {result.sipp_oversized_2}")

    result.net_worth_start = normalize(
        sipp_1 + sipp_df_1 + sipp_2 + sipp_df_2 + isa + gia, 2
    )

    assert state_pension_years_1 <= 35
    assert state_pension_years_2 <= 35

    lsa = UK.lsa

    assert sipp_contrib_1 <= UK.aa
    assert sipp_contrib_2 <= UK.aa

    if country == 'PT':
        gbpeur = float(hmrc.exchange_rate('EUR'))
    if country == 'JP':
        gbpjpy = float(hmrc.exchange_rate('JPY'))

    prob = lp.LpProblem("Retirement_Enhanced")

    max_income = retirement_income_net == 0
    if max_income:
        retirement_income_net = lp.LpVariable("income", 0)

    isa_allowance = UK.isa_allowance

    # Lump sum logic (unchanged)
    if lump_sum:
        ls_sipp_1 = lp.LpVariable("ls_sipp_1", 0)
        ls_sipp_2 = lp.LpVariable("ls_sipp_2", 0, None if joint else 0)
        ls_isa    = lp.LpVariable("ls_isa", 0, N * isa_allowance)
        ls_gia    = lp.LpVariable("ls_gia", 0)
        prob += ls_sipp_1 + ls_sipp_2 + ls_isa + ls_gia == lump_sum
        ls_sipp_gross_1 = ls_sipp_1 * (1.0 / (1.0 - max(marginal_income_tax_1, 0.20)))
        ls_sipp_gross_2 = ls_sipp_2 * (1.0 / (1.0 - max(marginal_income_tax_2, 0.20)))
        prob += sipp_contrib_1 + ls_sipp_gross_1 <= aa_1
        prob += sipp_contrib_2 + ls_sipp_gross_2 <= aa_2
        sipp_1 = sipp_1 + ls_sipp_gross_1
        sipp_2 = sipp_2 + ls_sipp_gross_2
        isa    = isa + ls_isa
        # Lump sum to GIA: split between ILG and equity per portfolio allocation
        ls_gia_ilg = ls_gia * ilg_pct
        ls_gia_eq  = ls_gia * (1 - ilg_pct)
        ilg_in_gia = ilg_in_gia + ls_gia_ilg
        eq_in_gia  = eq_in_gia + ls_gia_eq * (1 - eps)

    nmpa_1 = UK.nmpa(dob_1)
    nmpa_2 = UK.nmpa(dob_2)

    lsa_1 = lsa * lsa_ratio_1
    lsa_2 = lsa * lsa_ratio_2

    sipp_1 = DCP(
        prob=prob, uf=sipp_1, df=sipp_df_1,
        growth_rate_real=sipp_growth_rate_real_1,
        inflation_rate=inflation_rate, lsa=lsa_1, nmpa=nmpa_1
    )
    sipp_2 = DCP(
        prob=prob, uf=sipp_2, df=sipp_df_2,
        growth_rate_real=sipp_growth_rate_real_2,
        inflation_rate=inflation_rate, lsa=lsa_2, nmpa=nmpa_2
    )

    # Separate GIA objects for ILGs and equities
    gia_ilg = GIA_ILG(
        prob=prob, balance=ilg_in_gia,
        growth_rate_real=ilg_growth_rate_real,
        ilg_income_rate=ilg_income_rate
    )
    gia_eq = GIA_Equity(
        prob=prob, balance=eq_in_gia,
        growth_rate=equity_growth_rate,
        inflation_rate=inflation_rate
    )

    states = {}

    # SIPP contributions setup
    if sipp_extra_contrib:
        sipp_contrib_limit = UK.uiaa
        sipp_contrib_limit_1 = min(sipp_contrib_1 * 1.30, sipp_contrib_limit, UK.mpaa)
        sipp_contrib_limit_2 = min(sipp_contrib_2 * 1.30, sipp_contrib_limit, UK.mpaa)

    for yr in range(present_year, end_year):
        retirement = yr >= retirement_year
        uk_yr = not retirement or country == 'UK'

        age_1 = yr - dob_1
        age_2 = yr - dob_2

        # SIPP contributions (unchanged logic)
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

        # SIPP drawdowns
        drawdown_1 = lp.LpVariable(f'dd_1@{yr}', 0) \
            if age_1 >= nmpa_1 and (retirement or sipp_contrib_1 <= UK.mpaa) else 0
        drawdown_2 = lp.LpVariable(f'dd_2@{yr}', 0) \
            if age_2 >= nmpa_2 and (retirement or sipp_contrib_2 <= UK.mpaa) else 0

        tfc_1 = sipp_1.drawdown(drawdown_1, age=age_1)
        tfc_2 = sipp_2.drawdown(drawdown_2, age=age_2)

        # ISA drawdown (unchanged)
        drawdown_isa: lp.LpVariable | int
        if uk_yr:
            isa_allowance_yr = isa_allowance * N
            drawdown_isa = lp.LpVariable(f'dd_isa@{yr}', -isa_allowance_yr)
            isa = isa - drawdown_isa
            prob += isa >= 0
            isa *= 1.0 + isa_growth_rate_real
        elif yr == retirement_year:
            drawdown_isa = isa
            isa = 0
        else:
            drawdown_isa = 0
            assert isa == 0
        if yr < 2030:
            isa_allowance /= 1.0 + inflation_rate

        # === ENHANCED GIA LOGIC ===
        # ILG portion: no CGT, free to draw
        drawdown_gia_ilg, ilg_coupon_income = gia_ilg.flow()

        # Equity portion: CGT applies on gains
        drawdown_gia_eq, cg = gia_eq.flow(not uk_yr and country == 'PT')

        sipp_1.uf *= 1.0 + eps
        sipp_2.uf *= 1.0 + eps

        # State pension
        spa_1 = UK.state_pension_age(dob_1)
        spa_2 = UK.state_pension_age(dob_2)
        income_state_1 = state_pension_1 if age_1 >= spa_1 else 0
        income_state_2 = state_pension_2 if age_2 >= spa_2 else 0
        if country not in ('UK', 'PT'):
            income_state_1 *= (1.0 / (1.0 + inflation_rate)) ** max(age_1 - spa_1, 0)
            income_state_2 *= (1.0 / (1.0 + inflation_rate)) ** max(age_2 - spa_2, 0)

        # DB pensions
        ann_income_1 = income_state_1
        for pay, age in zip(db_payments_1, db_ages_1):
            ann_income_1 += pay if age_1 >= age else 0
        ann_income_2 = income_state_2
        for pay, age in zip(db_payments_2, db_ages_2):
            ann_income_2 += pay if age_2 >= age else 0

        # === INCOME TAX ===
        # ILG coupon income from GIA is taxable income, split between persons
        ilg_coupon_1 = ilg_coupon_income * income_ratio_1
        ilg_coupon_2 = ilg_coupon_income * income_ratio_2

        income_gross_1 = ann_income_1 + drawdown_1 + ilg_coupon_1
        income_gross_2 = ann_income_2 + drawdown_2 + ilg_coupon_2

        if uk_yr:
            # Capital gains only from equity portion of GIA (ILGs exempt)
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
                    tax_1, cgt_1 = uk_tax_lp(
                        prob, income_gross_1, cg_1, itt,
                        marriage_allowance=itt.marriage_allowance
                    )
                    tax_2, cgt_2 = uk_tax_lp(
                        prob, income_gross_2, cg_2, itt,
                        marriage_allowance=-itt.marriage_allowance
                    )
                else:
                    tax_1, cgt_1 = uk_tax_lp(prob, income_gross_1, cg_1, itt)
                    tax_2, cgt_2 = uk_tax_lp(prob, income_gross_2, cg_2, itt)
            cgt = cgt_1 + cgt_2

        elif country == 'PT':
            income_gross = (income_gross_1 + tfc_1 +
                            income_gross_2 + tfc_2)
            tax = pt_income_tax_lp(prob, income_gross, factor=N / gbpeur)
            # CGT only on equity gains (ILGs exempt even in PT as UK gov securities)
            cgt = cg * PT.cgt_rate
            income_gross_1 = income_gross * income_ratio_1
            income_gross_2 = income_gross * income_ratio_2
            tax_1 = tax * income_ratio_1
            tax_2 = tax * income_ratio_2

        elif country == 'JP':
            income_gross_1 = income_gross_1 + tfc_1
            income_gross_2 = income_gross_2 + tfc_2
            tax_1 = income_tax_lp(prob, income_gross_1, JP.income_tax_bands, 1 / gbpjpy)
            tax_2 = income_tax_lp(prob, income_gross_2, JP.income_tax_bands, 1 / gbpjpy)
            cgt = cg * JP.cgt_rate
        else:
            raise NotImplementedError

        # === FLOW BALANCE ===
        incomings = (
            income_gross_1 + income_gross_2 +
            drawdown_isa +
            drawdown_gia_ilg + drawdown_gia_eq  # Separate GIA flows
        )
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
            isa=isa,
            gia_ilg=gia_ilg.value(),
            gia_eq=gia_eq.value(),
            cg=cg,
            drawdown_1=drawdown_1,
            drawdown_2=drawdown_2,
            drawdown_isa=drawdown_isa,
            drawdown_gia_ilg=drawdown_gia_ilg,
            drawdown_gia_eq=drawdown_gia_eq,
            ann_income_1=ann_income_1,
            ann_income_2=ann_income_2,
            income_gross_1=income_gross_1,
            income_gross_2=income_gross_2,
            income_net=income_net,
            tax_1=tax_1,
            tax_2=tax_2,
            cgt=cgt,
        )

        # Threshold freeze logic (unchanged)
        if yr < 2031:
            itt.income_tax_threshold_20 = round(itt.income_tax_threshold_20 / (1.0 + inflation_rate))
            itt.income_tax_threshold_40 = round(itt.income_tax_threshold_40 / (1.0 + inflation_rate))
            itt.income_tax_threshold_45 = round(itt.income_tax_threshold_45 / (1.0 + inflation_rate))
            itt.pa_limit                = round(itt.pa_limit                / (1.0 + inflation_rate))
            itt.marriage_allowance      = round(itt.marriage_allowance      / (1.0 + inflation_rate))

    # === OBJECTIVE FUNCTION ===
    # Enhanced: includes IHT penalty on terminal estate
    net_worth = sipp_1.uf + sipp_2.uf + sipp_1.df + sipp_2.df + isa + gia_ilg.value() + gia_eq.value()

    if iht_weight > 0:
        # IHT-liable estate: ISA + GIA (always), SIPP (from 2027 if modelled)
        iht_estate = isa + gia_ilg.value() + gia_eq.value()
        if post_2027_pension_iht and end_year >= 2027:
            iht_estate = iht_estate + sipp_1.uf + sipp_1.df + sipp_2.uf + sipp_2.df

        # IHT threshold (combined for couple on second death)
        iht_threshold = iht_nil_rate_band * N
        if has_property:
            iht_threshold += iht_rnrb * N

        # Linearise IHT: penalise estate above threshold
        # This is an approximation — we add a penalty term to the objective
        iht_excess = lp.LpVariable("iht_excess", 0)
        prob += iht_excess >= iht_estate - iht_threshold
        iht_cost = iht_excess * 0.40
    else:
        iht_cost = 0

    if max_income:
        # Maximise income, penalised by IHT on terminal estate
        prob.setObjective(-retirement_income_net + iht_cost * iht_weight)
    else:
        # Maximise net worth after IHT
        prob.setObjective(-net_worth + iht_cost * iht_weight)

    solve(prob)

    # === EXTRACT RESULTS ===
    result.net_worth_end = normalize(
        lp.value(sipp_1.uf + sipp_1.df + sipp_2.uf + sipp_2.df + isa + gia_ilg.value() + gia_eq.value()), 0
    )

    if max_income:
        result.retirement_income_net = lp.value(retirement_income_net)
    else:
        result.retirement_income_net = retirement_income_net

    if max_income:
        retirement_income_net = lp.value(retirement_income_net)

    if iht_weight > 0:
        iht_estate_val = lp.value(isa + gia_ilg.value() + gia_eq.value())
        if post_2027_pension_iht:
            iht_estate_val += lp.value(sipp_1.uf + sipp_1.df + sipp_2.uf + sipp_2.df)
        iht_threshold = iht_nil_rate_band * N + (iht_rnrb * N if has_property else 0)
        result.total_iht = max(iht_estate_val - iht_threshold, 0) * 0.40

    for yr in range(present_year, end_year):
        s = states[yr]

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
        isa_val = lp.value(s.isa)
        gia_ilg_val = lp.value(s.gia_ilg)
        gia_eq_val = lp.value(s.gia_eq)
        drawdown_1 = lp.value(s.drawdown_1)
        drawdown_2 = lp.value(s.drawdown_2)
        drawdown_isa = lp.value(s.drawdown_isa)
        drawdown_gia_ilg = lp.value(s.drawdown_gia_ilg)
        drawdown_gia_eq = lp.value(s.drawdown_gia_eq)
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
        cgt_rate = cgt / max(cg, 1)

        tax = tax_1 + tax_2 + cgt
        result.total_tax += tax

        # IHT-liable estate at this point
        iht_estate_yr = isa_val + gia_ilg_val + gia_eq_val
        if post_2027_pension_iht and yr >= 2027:
            iht_estate_yr += sipp_uf_1 + sipp_df_1 + sipp_uf_2 + sipp_df_2

        if verbosity > 0:
            print(
                f'{yr}: '
                f'SIPP1[{sipp_uf_1:7.0f} {sipp_df_1:7.0f}] '
                f'SIPP2[{sipp_uf_2:7.0f} {sipp_df_2:7.0f}] '
                f'ISA {isa_val:7.0f} '
                f'GIA-ILG {gia_ilg_val:7.0f} ({-drawdown_gia_ilg:+7.0f}) '
                f'GIA-EQ {gia_eq_val:7.0f} ({-drawdown_gia_eq:+7.0f}) '
                f'Inc {income_gross_1:6.0f}/{income_gross_2:6.0f} '
                f'Tax {tax_1:5.0f}/{tax_2:5.0f} CGT {cgt:5.0f} '
                f'IHT-est {iht_estate_yr:9.0f}'
            )

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
            lsa_ratio_1=normalize(lsa_1 / lsa, 4),
            lsa_ratio_2=normalize(lsa_2 / lsa, 4),
            isa=isa_val,
            isa_delta=normalize(-drawdown_isa, 2),
            gia_ilg=normalize(gia_ilg_val, 2),
            gia_ilg_delta=normalize(-drawdown_gia_ilg, 2),
            gia_eq=normalize(gia_eq_val, 2),
            gia_eq_delta=normalize(-drawdown_gia_eq, 2),
            income_gross_1=income_gross_1,
            income_gross_2=income_gross_2,
            cg=cg,
            income_net=income_net,
            income_tax_1=tax_1,
            income_tax_2=tax_2,
            income_tax_rate_1=tax_rate_1,
            income_tax_rate_2=tax_rate_2,
            cgt=cgt,
            cgt_rate=cgt_rate,
            iht_estate=iht_estate_yr,
        )
        result.data.append(rs)

    if lump_sum:
        result.ls_sipp_1 = lp.value(ls_sipp_1)
        result.ls_sipp_2 = lp.value(ls_sipp_2)
        result.ls_isa    = lp.value(ls_isa)
        result.ls_gia    = lp.value(ls_gia)

    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

column_headers = {
    'year': 'Year',
    'ann_1': 'A1',
    'ann_2': 'A2',
    'sipp_uf_1': 'UF1',
    'contrib_1': '(+Δ)',
    'tfc_1': 'TFC1',
    'sipp_df_1': 'DF1',
    'sipp_delta_1': '(Δ)',
    'lsa_ratio_1': 'LSA1',
    'sipp_uf_2': 'UF2',
    'contrib_2': '(+Δ)',
    'tfc_2': 'TFC2',
    'sipp_df_2': 'DF2',
    'sipp_delta_2': '(Δ)',
    'lsa_ratio_2': 'LSA2',
    'isa': 'ISAs',
    'isa_delta': '(Δ)',
    'gia_ilg': 'GIA-ILG',
    'gia_ilg_delta': '(Δ)',
    'gia_eq': 'GIA-EQ',
    'gia_eq_delta': '(Δ)',
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
    'iht_estate': 'IHT-Est',
}


def dataframe(data):
    import pandas as pd
    return pd.DataFrame(data)


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
        'gia_ilg_delta': delta_format,
        'gia_eq_delta': delta_format,
        'lsa_ratio_1': perc_format,
        'lsa_ratio_2': perc_format,
        'income_tax_rate_1': perc_format,
        'income_tax_rate_2': perc_format,
        'cgt_rate': perc_format,
    }

    print(df.to_string(
        index=False,
        columns=column_headers.keys(),
        header=column_headers.values(),
        justify='center',
        float_format=float_format,
        formatters=formatters
    ))

    print()
    print(f"{'=' * 60}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Start net worth:         {result.net_worth_start:>12,.0f}")
    print(f"Retirement net income:   {result.retirement_income_net:>12,.0f}")

    country = params['country']
    if country == 'PT':
        gbpeur = float(hmrc.exchange_rate('EUR'))
        print(f"Retirement net income:   {result.retirement_income_net * gbpeur:>12,.0f} EUR")
    if country == 'JP':
        gbpjpy = float(hmrc.exchange_rate('JPY'))
        print(f"Retirement net income:   {result.retirement_income_net * gbpjpy:>12,.0f} JPY")

    print(f"End net worth:           {result.net_worth_end:>12,.0f}")
    print(f"Total income/CGT tax:    {result.total_tax:>12,.0f}")
    print(f"Estimated IHT:           {result.total_iht:>12,.0f}")
    print(f"Total tax + IHT:         {result.total_tax + result.total_iht:>12,.0f}")

    print()
    print(f"{'=' * 60}")
    print(f"SIPP SIZE ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Person 1 SIPP oversized: {'YES' if result.sipp_oversized_1 else 'No'}")
    print(f"Person 2 SIPP oversized: {'YES' if result.sipp_oversized_2 else 'No'}")

    print()
    print(f"{'=' * 60}")
    print(f"ASSET LOCATION RECOMMENDATION")
    print(f"{'=' * 60}")
    for line in result.asset_location_note.split(" | "):
        print(f"  {line}")

    if result.ls_sipp_1 + result.ls_sipp_2 + result.ls_isa + result.ls_gia:
        print()
        print("Lump sum allocation:")
        print(f"  SIPP1: {result.ls_sipp_1:8.0f}")
        print(f"  SIPP2: {result.ls_sipp_2:8.0f}")
        print(f"  ISA:   {result.ls_isa:8.0f}")
        print(f"  GIA:   {result.ls_gia:8.0f}")

