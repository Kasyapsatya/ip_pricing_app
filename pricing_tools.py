"""
IP Semi-Markov Pricing Tools
=============================
Ported directly from the locked notebook (ip_semi_markov_pricing_agent.ipynb).
Constants, synthetic data generation, and the three pricing tools + guardrails +
plain-English explainers, unchanged from the notebook version.

Data generation (generate_population / simulate_spells) is only ever called by
build_data.py, ONCE, to produce data/pricing_artifacts.pkl. The Streamlit app
never re-simulates — it loads that file via init_from_artifacts().
"""
import numpy as np
import pandas as pd

DEFERRED_WEEKS = 13
STANDARD_DEFERRED_OPTIONS = [4, 13, 26, 52]   # the deferred-period choices actually priced
STUDY_WEEKS = 520          # 10-year observation window (see note above on why not 5 years)
N_LIVES = 6000

# Age-banded base incidence — steadily increasing with age, this is a rating factor that
# lives in Tool 1's pooled base table (25-34 / 35-49 / 50-60).
AGE_BAND_LABELS = {0: '25-34', 1: '35-49', 2: '50-60'}
AGE_BASE_INCIDENCE = {0: 0.035, 1: 0.055, 2: 0.085}


def age_band(age):
    """Bands age into the three rating groups. Anyone below 35 -> band 0, 35-49 -> band 1,
    50 and above -> band 2 (naturally clamps any age, no invented band possible)."""
    if age < 35:
        return 0
    elif age < 50:
        return 1
    return 2


# Occupation class — a couple of high-level categories, the second rating factor in Tool 1's
# base table (same treatment as age: it affects INCIDENCE only, not deferred-recovery or
# claiming-duration behaviour, which stay governed purely by episode history in Tool 3).
OCCUPATION_CLASSES = ['desk', 'manual']
OCCUPATION_MULT = {'desk': 1.00, 'manual': 1.55}
OCCUPATION_MIX = {'desk': 0.65, 'manual': 0.35}   # proportion of the simulated population in each class

# Episode-band incidence MULTIPLIER — applied on top of the age/occupation base rate above,
# capped at band "2+" (originally split into 0/1/2/3+, but band 3 turned out too thin — 9 spells
# — for the credibility weighting to produce a stable loading. Capping one band earlier pools
# band 2 and band 3+ together, giving a much more solidly estimated top band.) Multipliers
# preserve the same relative escalation as the original flat rates (2.4x, 3.3x).
EPISODE_INCIDENCE_MULT = {0: 1.00, 1: 2.40, 2: 3.30}

# TOTAL sickness-spell duration (deferred + claiming combined) is now drawn as ONE underlying
# duration per spell, from onset to eventual recovery/death — the deferred period is then applied
# as a genuine CUTOFF on that single duration, not a separate independent draw. This is what
# makes deferred_weeks an actual lever: a shorter deferred period pulls more of the same
# underlying spells into paid claiming, a longer one keeps more of them fully within the
# (unpaid) deferred window. Same declining-hazard Weibull shape as before, scaled up by episode
# history (repeat episodes take longer to resolve overall, not just once claiming starts).
TERMINAL_DEATH_PROB = 0.020        # probability the whole spell ends in death, not recovery
TOTAL_DURATION_BASE_SCALE_WEEKS = 10.0
EPISODE_DURATION_SCALE_MULT = {0: 1.00, 1: 1.30, 2: 1.50}
CLAIMING_WEIBULL_SHAPE = 0.60      # shape < 1 -> declining hazard, i.e. "stickier" the longer you're sick
MAX_SPELL_WEEKS = 156              # overall cap on total sickness duration (3 years)


def episode_band(prior_episode_count):
    """Caps episode history at band 2, matching the credible-band logic agreed earlier —
    we never model a smooth curve into thin data past episode 2."""
    return min(prior_episode_count, 2)


def generate_population(n=N_LIVES, seed=42):
    rng = np.random.default_rng(seed)
    ages = rng.integers(25, 61, size=n)
    incomes = rng.lognormal(mean=np.log(60000), sigma=0.45, size=n).round(-2)
    incomes = np.clip(incomes, 20000, 400000)
    occupations = rng.choice(OCCUPATION_CLASSES, size=n,
                              p=[OCCUPATION_MIX[c] for c in OCCUPATION_CLASSES])
    return pd.DataFrame({
        'policyholder_id': [f'PH{str(i).zfill(5)}' for i in range(n)],
        'age': ages,
        'monthly_income': incomes.astype(int),
        'occupation': occupations,
    })

def simulate_spells(population, study_weeks=STUDY_WEEKS, deferred_weeks=DEFERRED_WEEKS, seed=7):
    """Simulates sickness spells per life: onset -> Sick(deferred) -> [Sick(claiming)] -> resolution.
    This is the raw claims-generating process; the pricing tools in Sections 2-3 only ever see
    the output of this function, exactly as a real pricing team would only see claims data,
    not the underlying (unobservable) hazard process.

    Each spell draws ONE underlying total-sickness-duration (onset to eventual recovery/death) —
    the deferred period is then applied as a genuine cutoff on that single duration, not a
    separate independent draw. This is what makes deferred_weeks a real lever, and it's also
    exactly why Tool 2 doesn't need to call this function again for other deferred-period
    options: since the full duration is recorded for every spell regardless of outcome, any
    other deferred-period cutoff can be applied directly to this one dataset afterward — see
    Tool 2 below."""
    rng = np.random.default_rng(seed)
    records = []

    for _, ph in population.iterrows():
        week = 0.0
        episode_number = 0
        alive = True
        a_band = age_band(ph['age'])  # static for the life — see note on this simplification in Section 2
        occ = ph['occupation']

        while alive and week < study_weeks:
            band = episode_band(episode_number)
            p_annual = AGE_BASE_INCIDENCE[a_band] * OCCUPATION_MULT[occ] * EPISODE_INCIDENCE_MULT[band]
            weekly_hazard = -np.log(1 - p_annual) / 52
            wait = rng.exponential(1 / weekly_hazard)
            week_onset = week + wait
            if week_onset >= study_weeks:
                break  # censored healthy — no further spells observed

            episode_number += 1

            # --- ONE underlying total-duration draw, then apply the deferred-period cutoff ---
            is_death = rng.random() < TERMINAL_DEATH_PROB
            scale = TOTAL_DURATION_BASE_SCALE_WEEKS * EPISODE_DURATION_SCALE_MULT[band]
            resolution_week = float(rng.weibull(CLAIMING_WEIBULL_SHAPE) * scale)
            resolution_week = min(resolution_week, MAX_SPELL_WEEKS)

            if resolution_week <= deferred_weeks:
                # resolves entirely within the deferred period — no claim ever paid
                deferred_outcome = 'died_in_deferred' if is_death else 'recovered_in_deferred'
                weeks_in_deferred = resolution_week
                claiming_weeks = 0.0
                claim_outcome = 'na'
            else:
                deferred_outcome = 'crossed_to_claiming'
                weeks_in_deferred = float(deferred_weeks)
                claiming_weeks = resolution_week - deferred_weeks
                claim_outcome = 'died_while_claiming' if is_death else 'recovered_while_claiming'
            if is_death:
                alive = False

            week_end = week_onset + weeks_in_deferred + claiming_weeks
            censored = week_end > study_weeks
            if censored:
                overrun = week_end - study_weeks
                claiming_weeks = max(0.0, claiming_weeks - overrun)
                week_end = study_weeks

            monthly_claim_amount = round(0.8 * ph['monthly_income'], 2)
            total_claim_paid = round((claiming_weeks / 4.345) * monthly_claim_amount, 2)

            records.append({
                'policyholder_id': ph['policyholder_id'],
                'age_at_onset': ph['age'],
                'age_band': a_band,
                'occupation': occ,
                'monthly_income': ph['monthly_income'],
                'episode_number': episode_number,
                'episode_band': band,
                'week_onset': round(week_onset, 2),
                'deferred_outcome': deferred_outcome,
                'weeks_in_deferred': round(weeks_in_deferred, 2),
                'crossed_to_claiming': deferred_outcome == 'crossed_to_claiming',
                'claiming_weeks': round(claiming_weeks, 2),
                'claim_outcome': claim_outcome,
                'monthly_claim_amount': monthly_claim_amount,
                'total_claim_paid': total_claim_paid,
                'censored': censored,
            })
            week = week_end

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Module-level state — populated once at app startup via init_from_artifacts()
# ---------------------------------------------------------------------------
population = None
spells = None
base_table = None
loading_table = None
deferred_period_table = None

def calculate_base_incidence_rates(population_df, spells_df):
    """TOOL 1 — Base incidence table: annual probability of Healthy -> Sick(deferred), rated by
    age band AND occupation class (two standard rating factors), pooled across prior-episode
    history. Crude central-exposure rate only; no graduation or smoothing.

    Exposure is Healthy + Sick(deferred) time — i.e. the premium-paying period under
    waiver-of-premium (premium is charged while healthy or in deferred, waived once claiming
    starts). This is what makes the resulting rate an exact breakeven rate against total claims
    cost — see calculate_premium for the algebra."""
    population_df = population_df.copy()
    population_df['age_band'] = population_df['age'].apply(age_band)

    incidence_table = {}
    exposure_table = {}
    for a_band in [0, 1, 2]:
        for occ in OCCUPATION_CLASSES:
            lives_in_cell = population_df[
                (population_df['age_band'] == a_band) & (population_df['occupation'] == occ)]
            n_lives_cell = len(lives_in_cell)
            total_possible_weeks_cell = n_lives_cell * STUDY_WEEKS
            spells_in_cell = spells_df[
                (spells_df['age_band'] == a_band) & (spells_df['occupation'] == occ)]
            # Exposure = Healthy + Sick(deferred) time only — this is the premium-paying period
            # under waiver-of-premium (no premium collected while Sick(claiming)). Excluding only
            # claiming_weeks (not weeks_in_deferred) from the denominator keeps this rate an exact
            # breakeven rate: incidence x p_cross x avg_claim_cost = total claims cost / this
            # exposure base, which is only true if the exposure base matches the premium-collection
            # period exactly.
            no_premium_weeks_cell = spells_in_cell['claiming_weeks'].sum()
            healthy_exposure_years_cell = (total_possible_weeks_cell - no_premium_weeks_cell) / 52
            incidence_cell = (len(spells_in_cell) / healthy_exposure_years_cell
                               if healthy_exposure_years_cell > 0 else float('nan'))
            incidence_table[(a_band, occ)] = round(incidence_cell, 4)
            exposure_table[(a_band, occ)] = round(healthy_exposure_years_cell, 1)

    return {
        'method': 'central_exposure_crude_no_graduation_age_occupation_incidence',
        'states': ['healthy', 'sick_deferred', 'sick_claiming', 'death'],
        'age_band_labels': AGE_BAND_LABELS,
        'occupation_classes': OCCUPATION_CLASSES,
        'incidence_table': incidence_table,          # keyed by (age_band, occupation)
        'incidence_exposure_life_years': exposure_table,
    }



def calculate_deferred_period_table(spells_df, options=STANDARD_DEFERRED_OPTIONS):
    """TOOL 2 — Restates the ONE reference dataset (generated once, at the real 13-week deferred
    period) under each deferred-period option, rather than re-simulating. Every spell's total
    sickness duration (weeks_in_deferred + claiming_weeks) is recorded exactly, regardless of
    whether it happened to cross into claiming — so any deferred-period cutoff can be applied to
    that one duration directly. FREQUENCY = P(cross), SEVERITY = average duration and cost of
    the claims that do."""
    total_duration = spells_df['weeks_in_deferred'] + spells_df['claiming_weeks']
    rows = []
    for D in options:
        crosses = total_duration > D
        claiming_weeks_D = total_duration[crosses] - D
        cost_D = (claiming_weeks_D / 4.345) * spells_df.loc[crosses, 'monthly_claim_amount']
        rows.append({
            'deferred_weeks': D,
            'p_cross_to_claiming': round(float(crosses.mean()), 3),
            'avg_claiming_weeks': round(float(claiming_weeks_D.mean()), 1),
            'avg_claim_cost': round(float(cost_D.mean()), 0),
        })
    return pd.DataFrame(rows).set_index('deferred_weeks')


def calculate_experience_loading(spells_df, base_table, credibility_k=40):
    """TOOL 3 — Experience-rating loading by prior-episode band, Buhlmann-style partial
    credibility blended toward 1.0. This table is deliberately separate from Tool 1 — it is
    applied AFTER the pooled base premium, exactly as agreed for Approach 2.

    IMPORTANT: observed cost is averaged over ALL spells in the band, not just the ones that
    crossed into claiming. A spell that recovered during the deferred period contributes a
    claim cost of 0. This matters because prior-episode history worsens BOTH (a) how likely a
    spell is to reach claiming at all, and (b) how long it runs once it does — averaging only
    over already-claiming spells would silently drop effect (a) and understate the loading."""
    pooled_avg_claim_per_spell = spells_df['total_claim_paid'].mean()

    rows = []
    for band in [0, 1, 2]:  # always report all three bands, even if the top one has few spells so far
        band_spells = spells_df[spells_df['episode_band'] == band]
        n = len(band_spells)
        observed_avg_claim = float(band_spells['total_claim_paid'].mean()) if n else 0.0
        observed_ratio = observed_avg_claim / pooled_avg_claim_per_spell if pooled_avg_claim_per_spell else 1.0
        Z = n / (n + credibility_k)
        loading = Z * observed_ratio + (1 - Z) * 1.0
        band_label = str(band) if band < 2 else '2+'
        rows.append({
            'prior_episodes_band': band_label,
            'n_spells': n,
            'observed_avg_claim_cost': round(observed_avg_claim, 0),
            'observed_ratio': round(observed_ratio, 3),
            'credibility_Z': round(Z, 3),
            'loading_factor': round(loading, 3),
        })
    return pd.DataFrame(rows).sort_values('prior_episodes_band').reset_index(drop=True)


VALID_STATES = ['healthy', 'sick_deferred', 'sick_claiming', 'death']
VALID_EPISODE_BANDS = ['0', '1', '2+']


def check_state_exists(state_name):
    """GUARDRAIL 1 — refuses to discuss a transition state that isn't in the base table."""
    normalized = state_name.strip().lower().replace(' ', '_').replace('(', '').replace(')', '')
    exists = normalized in VALID_STATES
    return {'exists': exists, 'requested': state_name, 'valid_states': VALID_STATES}


def check_occupation_exists(occupation):
    """GUARDRAIL 2 (gates Tool 1) — refuses to price or explain an occupation class outside the
    two modelled categories. A specific job title ('pilot', 'nurse') is not a match — only the
    two broad classes the incidence table was actually built on."""
    normalized = str(occupation).strip().lower()
    exists = normalized in OCCUPATION_CLASSES
    return {'exists': exists, 'requested': occupation, 'valid_classes': OCCUPATION_CLASSES}


def check_deferred_option_exists(deferred_weeks):
    """GUARDRAIL 3 (gates Tool 2) — restricts deferred-period pricing to the standard options
    actually in the lookup table. Never invents a rate for a non-standard deferred period."""
    try:
        weeks = int(deferred_weeks)
        exists = weeks in STANDARD_DEFERRED_OPTIONS
    except (TypeError, ValueError):
        weeks, exists = None, False
    return {'exists': exists, 'requested': deferred_weeks, 'valid_options': STANDARD_DEFERRED_OPTIONS}


def check_episode_band_exists(prior_episode_count):
    """GUARDRAIL 4 (gates Tool 3) — caps episode history lookups at the '2+' band. Never invents
    a loading factor for a band beyond what the credibility-weighted table actually covers."""
    try:
        n = int(prior_episode_count)
        band = str(n) if n < 2 else '2+'
        exists = band in VALID_EPISODE_BANDS
    except (TypeError, ValueError):
        band, exists = None, False
    return {'exists': exists, 'requested': prior_episode_count, 'resolved_band': band,
            'valid_bands': VALID_EPISODE_BANDS}


def explain_transition(state_name):
    """Explains a state's role using ONLY numbers already present in base_table. No invention."""
    check = check_state_exists(state_name)
    if not check['exists']:
        return (f"I can't explain '{state_name}' — it isn't one of the four modelled states "
                f"({', '.join(VALID_STATES)}). I won't invent a number for it.")
    normalized = check['requested'].strip().lower().replace(' ', '_').replace('(', '').replace(')', '')
    if normalized == 'healthy':
        lines = "; ".join(
            f"{AGE_BAND_LABELS[a]}/{occ}: {r:.2%}/yr" for (a, occ), r in base_table['incidence_table'].items())
        return (f"FREQUENCY — how often a healthy person falls sick, by age and occupation: "
                f"{lines}. This is the starting point for every premium; it doesn't yet say "
                f"anything about how bad a claim is once it happens.")
    if normalized == 'sick_deferred':
        return ("Sick(deferred) is the waiting period — no benefit accrues here, and no "
                "premium is charged either way (waiver of premium). A spell exits either by "
                "recovering, dying, or by lasting long enough to cross into Sick(claiming) — "
                "how much of each depends on the deferred period chosen (see Tool 2).")
    if normalized == 'sick_claiming':
        return ("Sick(claiming) is entered once the deferred period elapses while still sick — "
                "income replacement (80% of monthly income) accrues here until recovery or "
                "death. Premium is waived for the whole time a policyholder is in this state.")
    if normalized == 'death':
        return "Death is absorbing. No IP benefit is payable — income payments simply stop."


def explain_occupation(occupation):
    """Explains occupation's effect on incidence using ONLY numbers already present in
    base_table. Occupation is priced as part of the base incidence table (Tool 1) — same
    treatment as age — not as a personal loading like prior-episode history."""
    check = check_occupation_exists(occupation)
    if not check['exists']:
        return (f"I can't price or explain occupation '{occupation}' — the model only covers "
                f"{', '.join(OCCUPATION_CLASSES)}. I won't invent a loading for anything else.")
    occ = str(occupation).strip().lower()
    lines = "; ".join(f"{AGE_BAND_LABELS[a]}: {base_table['incidence_table'][(a, occ)]:.2%}/yr"
                       for a in [0, 1, 2])
    return f"FREQUENCY — '{occ}' occupation's incidence by age band: {lines}."


def explain_deferred_option(deferred_weeks):
    """Explains a deferred-period option using ONLY numbers already present in
    deferred_period_table. No invention for non-standard options."""
    check = check_deferred_option_exists(deferred_weeks)
    if not check['exists']:
        return (f"I can't price a {deferred_weeks}-week deferred period — the standard options "
                f"modelled are {STANDARD_DEFERRED_OPTIONS} weeks. I won't invent a rate for "
                f"anything outside that set.")
    row = deferred_period_table.loc[int(deferred_weeks)]
    return (f"With a {int(deferred_weeks)}-week deferred period — "
            f"FREQUENCY: {row['p_cross_to_claiming']:.1%} of sickness spells go on to reach a "
            f"paid claim. SEVERITY: those claims run about {row['avg_claiming_weeks']:.1f} "
            f"weeks on average, costing roughly Rs {row['avg_claim_cost']:,.0f} in total.")


def explain_loading(prior_episode_count):
    """Explains a loading factor using ONLY numbers already present in loading_table."""
    check = check_episode_band_exists(prior_episode_count)
    if not check['exists']:
        return (f"I can't quote a loading factor for {prior_episode_count} prior episodes — "
                f"the credibility table only covers bands {', '.join(VALID_EPISODE_BANDS)}. "
                f"I won't extrapolate a number beyond what's credibly estimated.")
    row = loading_table[loading_table['prior_episodes_band'] == check['resolved_band']].iloc[0]
    return (f"SEVERITY loading for {check['resolved_band']} prior episode(s): claims in this "
            f"band ran at {row.observed_ratio:.2f}x the typical cost — blending both a higher "
            f"chance of the sickness actually turning into a paid claim, and running longer "
            f"once it does. With only {int(row.n_spells)} spells behind this band, the "
            f"credibility weight is Z={row.credibility_Z:.2f}, so the loading actually applied "
            f"is {row.loading_factor:.2f}x.")


def calculate_premium(age, monthly_income, prior_episodes, occupation='desk', deferred_weeks=DEFERRED_WEEKS):
    """Combines Tool 1 (age x occupation base incidence), Tool 2 (deferred-period table), and
    Tool 3 (episode-based experience loading) into a single annual premium, with a full
    breakdown for the explainer to narrate."""
    band_check = check_episode_band_exists(prior_episodes)
    if not band_check['exists']:
        raise ValueError(f"Cannot price {prior_episodes} prior episodes — outside credible bands.")
    band_label = band_check['resolved_band']
    loading_factor = float(loading_table.loc[loading_table['prior_episodes_band'] == band_label, 'loading_factor'].iloc[0])

    occ_check = check_occupation_exists(occupation)
    if not occ_check['exists']:
        raise ValueError(f"Cannot price occupation '{occupation}' — not one of {OCCUPATION_CLASSES}.")
    occ = str(occupation).strip().lower()

    deferred_check = check_deferred_option_exists(deferred_weeks)
    if not deferred_check['exists']:
        raise ValueError(f"Cannot price a {deferred_weeks}-week deferred period — not one of "
                          f"the standard options {STANDARD_DEFERRED_OPTIONS}.")
    weeks = int(deferred_weeks)

    a_band = age_band(age)
    incidence_for_cell = base_table['incidence_table'][(a_band, occ)]

    deferred_row = deferred_period_table.loc[weeks]
    p_cross = float(deferred_row['p_cross_to_claiming'])
    avg_claiming_weeks = float(deferred_row['avg_claiming_weeks'])
    avg_claim_cost = float(deferred_row['avg_claim_cost'])
    # pooled_base_premium is INCOME-NEUTRAL — incidence x crossing probability x the pooled
    # (portfolio-average-income) claim cost. Income and the experience loading are then applied
    # as two separate, clean multiplicative factors on top — see final_annual_premium below.
    pooled_base_premium = incidence_for_cell * p_cross * avg_claim_cost

    income_scale = monthly_income / population['monthly_income'].mean()
    final_premium = pooled_base_premium * income_scale * loading_factor

    return {
        'age': age,
        'age_band': AGE_BAND_LABELS[a_band],
        'occupation': occ,
        'monthly_income': monthly_income,
        'prior_episodes': prior_episodes,
        'resolved_episode_band': band_label,
        'deferred_weeks': weeks,
        'incidence_for_cell': incidence_for_cell,
        'p_cross_to_claiming': round(p_cross, 3),
        'avg_claiming_weeks': round(avg_claiming_weeks, 1),
        'avg_claim_cost_for_your_income': round(avg_claim_cost * income_scale, 0),
        'pooled_base_premium': round(float(pooled_base_premium), 0),
        'income_scale': round(float(income_scale), 3),
        'loading_factor': loading_factor,
        'final_annual_premium': round(float(final_premium), 0),
    }



# ---------------------------------------------------------------------------
# Artifact build / load — this is the "simulate once, use everywhere" boundary
# ---------------------------------------------------------------------------
def build_artifacts():
    """Generates population + spells and computes all three tool tables fresh.
    Called ONLY by build_data.py, never by the Streamlit app itself."""
    pop = generate_population()
    sp = simulate_spells(pop)
    bt = calculate_base_incidence_rates(pop, sp)
    dt = calculate_deferred_period_table(sp)
    lt = calculate_experience_loading(sp, bt)
    return {
        'population': pop,
        'spells': sp,
        'base_table': bt,
        'deferred_period_table': dt,
        'loading_table': lt,
    }


def init_from_artifacts(artifacts):
    """Sets module-level state from a pre-built artifacts dict (loaded from disk).
    Every tool function above reads these as module globals at call-time, so this
    only needs to run once per app process."""
    global population, spells, base_table, loading_table, deferred_period_table
    population = artifacts['population']
    spells = artifacts['spells']
    base_table = artifacts['base_table']
    deferred_period_table = artifacts['deferred_period_table']
    loading_table = artifacts['loading_table']
