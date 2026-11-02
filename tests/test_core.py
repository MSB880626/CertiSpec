"""
Tests for CertiSpec. Run:  pytest -q

Tipos de prueba estadística (todas con semilla fija):
  DEMOSTRACIÓN  cota SUPERIOR del IC (99.9 %) de la tasa de fallo <= nominal. Solo es
                alcanzable si el procedimiento es conservador (tasa real < nominal).
  EXACTITUD     el IC bilateral (99.9 %) de la tasa de fallo contiene el nominal. Se usa
                para la banda os con F continua, cuya cobertura es EXACTAMENTE el
                nominal: ahí la cota superior siempre supera el nominal, por lo que una
                prueba de demostración fallaría aunque el método sea correcto.
  FALSACIÓN     cota INFERIOR del IC (99.9 %) <= nominal: no hay evidencia de violación.
                Se usa para DKW con F continua: la cota de Massart es casi exacta para
                eta pequeño (no cobertura observada 0.0875 con n=30 y 0.0945 con n=100,
                eta=0.10), así que tampoco admite una prueba de demostración.
Ninguna prueba demuestra una garantía; detectan errores de implementación. Las garantías
están demostradas en el docstring del módulo.
"""
import numpy as np
import mpmath as mp
import pandas as pd
import pytest
from scipy import stats

from certispec import core as cf

CONF = 0.999


def upper(k, n):
    return cf.cp_upper1(k, n, CONF)


def ci2(k, n):
    return cf.cp_ci(k, n, CONF)


# ---------------------------------------------------------------------------------------
# Referencia independiente: fórmula de Steck (1971) en aritmética de alta precisión
# P(a_i <= U_(i) <= b_i) = n! det[(b_i - a_j)_+^(j-i+1) / (j-i+1)!]
# ---------------------------------------------------------------------------------------
def steck(a, b, dps=120):
    n = len(a)
    mp.mp.dps = dps
    Q = mp.matrix(n, n)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            e = j - i + 1
            x = mp.mpf(float(b[i - 1])) - mp.mpf(float(a[j - 1]))
            if e < 0:
                Q[i - 1, j - 1] = 0
            elif e == 0:
                Q[i - 1, j - 1] = 1 if x > 0 else 0
            else:
                Q[i - 1, j - 1] = (max(x, 0) ** e) / mp.factorial(e)
    return float(mp.factorial(n) * mp.det(Q))


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 12, 20, 30])
@pytest.mark.parametrize("g", [0.5, 0.05, 0.001])
def test_dp_equals_steck_beta_bands(n, g):
    a, b = cf.beta_band(n, g)
    assert abs(cf.os_exact_coverage(a, b) - steck(a, b)) < 1e-9


@pytest.mark.parametrize("seed", range(25))
def test_dp_equals_steck_random_bands(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 15))
    a = np.sort(rng.uniform(0, 0.7, n))
    b = np.maximum.accumulate(np.minimum(a + rng.uniform(0.02, 0.6, n), 1.0))
    assert abs(cf.os_exact_coverage(a, b) - steck(a, b)) < 1e-9


def test_dp_analytic_edge_cases():
    assert cf.os_exact_coverage([0.2], [0.7]) == pytest.approx(0.5, abs=1e-12)
    assert cf.os_exact_coverage(np.zeros(10), np.ones(10)) == pytest.approx(1.0, abs=1e-12)
    assert cf.os_exact_coverage([0.5, 0.6], [0.4, 0.9]) == 0.0
    # cotas coincidentes y extremos 0/1
    a = np.array([0.0, 0.3, 0.3]); b = np.array([0.3, 0.3, 1.0])
    assert abs(cf.os_exact_coverage(a, b) - steck(a, b)) < 1e-9


def test_dp_matches_monte_carlo_large_n():
    rng = np.random.default_rng(0)
    n, sims, hits = 150, 300_000, 0
    a, b = cf.beta_band(n, 0.002)
    for _ in range(sims // 20_000):
        U = np.sort(rng.uniform(size=(20_000, n)), 1)
        hits += int(((U >= a) & (U <= b)).all(1).sum())
    mc = hits / sims
    se = np.sqrt(mc * (1 - mc) / sims)
    assert abs(cf.os_exact_coverage(a, b) - mc) < 4.5 * se


def test_coverage_monotone_in_g():
    for n in (10, 50, 200):
        cov = [cf.os_exact_coverage(*cf.beta_band(n, g)) for g in np.logspace(-8, -0.3, 30)]
        assert all(c1 >= c2 - 1e-12 for c1, c2 in zip(cov[:-1], cov[1:]))


@pytest.mark.parametrize("n", [3, 10, 25, 100, 250, 600])
def test_band_levels_valid_maximal_stable(n):
    eta = 0.025
    a, b, g, cov = cf.os_band_levels(n, eta)
    assert np.isfinite(cov) and 1 - eta <= cov <= 1.0
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    assert np.all(np.diff(a) > 0) and np.all(np.diff(b) > 0) and np.all(a < b)
    assert cf.os_exact_coverage(*cf.beta_band(n, g * 1.001)) < 1 - eta + 1e-6


def test_dkw_massart_condition():
    eta = cf.DELTA / 2
    for n in (5, 50, 500):
        eps = np.sqrt(np.log(2 / eta) / (2 * n))
        assert np.exp(-2 * n * eps ** 2) <= 0.5
        assert 2 * np.exp(-2 * n * eps ** 2) == pytest.approx(eta)


# ---------------------------------------------------------------------------------------
# band_eval: convenciones, empates y extremos
# ---------------------------------------------------------------------------------------
def test_band_eval_extremes_and_ties():
    x = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
    band = cf.band_os(x, 0.025)
    _, _, a, b = band
    L, U = cf.band_eval(band, np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 9.0]))
    # k = #{x <= t}: 0, 1, 1, 4, 4, 5, 5 (los empates cuentan todos)
    k = np.array([0, 1, 1, 4, 4, 5, 5])
    expL = np.where(k >= 1, a[np.clip(k - 1, 0, 4)], 0.0)
    expU = np.where(k <= 4, b[np.clip(k, 0, 4)], 1.0)
    assert np.allclose(L, expL) and np.allclose(U, expU)
    assert L[0] == 0.0 and U[-1] == 1.0


def test_band_brackets_empirical_cdf():
    rng = np.random.default_rng(1)
    for x in (rng.normal(size=40), rng.integers(0, 4, 40).astype(float)):
        for band in (cf.band_os(x, 0.025), cf.band_dkw(x, 0.025)):
            t = np.linspace(-4, 6, 400)
            L, U = cf.band_eval(band, t)
            Fn = np.searchsorted(np.sort(x), t, side="right") / len(x)
            assert np.all((0 <= L) & (L <= Fn + 1e-12) & (Fn <= U + 1e-12) & (U <= 1))


# ---------------------------------------------------------------------------------------
# Cobertura simultánea empírica de las bandas
# ---------------------------------------------------------------------------------------
def _band_fail_rate(dist, method, n, eta, reps, seed):
    rng = np.random.default_rng(seed)
    discrete = isinstance(dist.dist, stats.rv_discrete)
    if discrete:
        sup = np.asarray(getattr(dist.dist, "xk", np.arange(dist.ppf(1e-12), dist.ppf(1 - 1e-12) + 1)), float)
        grid = np.concatenate([sup, sup - 1e-6, [sup.min() - 1, sup.max() + 1]])
    fail = 0
    for _ in range(reps):
        x = dist.rvs(n, random_state=rng).astype(float)
        band = cf.band_os(x, eta) if method == "os" else cf.band_dkw(x, eta)
        if not discrete:
            xs = np.sort(x)
            grid = np.concatenate([np.linspace(-7, 7, 400), xs, xs - 1e-9 * (1 + abs(xs))])
        L, U = cf.band_eval(band, grid)
        F = dist.cdf(grid)
        fail += bool(((F < L - 1e-12) | (F > U + 1e-12)).any())
    return fail


def test_os_continuous_coverage_is_exact():
    """EXACTITUD: con F continua la no cobertura de os es exactamente eta."""
    eta, reps, n = 0.10, 6000, 30
    k = _band_fail_rate(stats.norm(0, 1), "os", n, eta, reps, 7)
    lo, hi = ci2(k, reps)
    assert lo <= eta <= hi, (k / reps, lo, hi)


@pytest.mark.parametrize("dist", [stats.binom(10, 0.4), stats.poisson(2.0), cf.rounded_normal(0.0)])
@pytest.mark.parametrize("method", ["os", "dkw"])
def test_band_coverage_discrete_demonstration(dist, method):
    """DEMOSTRACIÓN: con F discreta (empates) ambas bandas son conservadoras."""
    eta, reps, n = 0.10, 3000, 30
    k = _band_fail_rate(dist, method, n, eta, reps, 8)
    assert upper(k, reps) <= eta, (k / reps, upper(k, reps))


@pytest.mark.parametrize("n", [30, 100])
def test_dkw_continuous_coverage_falsification(n):
    """FALSACIÓN: DKW es casi exacta con F continua; se exige ausencia de evidencia de
    violación (la garantía de Massart está demostrada en la literatura)."""
    eta, reps = 0.10, 4000
    k = _band_fail_rate(stats.norm(0, 1), "dkw", n, eta, reps, 9)
    assert ci2(k, reps)[0] <= eta, (k / reps, ci2(k, reps))


# ---------------------------------------------------------------------------------------
# Cotas de riesgo, Clopper-Pearson
# ---------------------------------------------------------------------------------------
def test_risk_bounds_monotone():
    rng = np.random.default_rng(3)
    for _ in range(5000):
        pi = rng.uniform(0.01, 0.99)
        fpr, tpr, fnr, tnr = rng.uniform(1e-6, 1, 4)
        ufpr, ufnr = fpr + rng.uniform(0, 1 - fpr), fnr + rng.uniform(0, 1 - fnr)
        ltpr, ltnr = tpr * rng.uniform(1e-6, 1), tnr * rng.uniform(1e-6, 1)
        assert cf.risk_pos(pi, fpr, tpr) <= cf.risk_pos(pi, ufpr, ltpr) + 1e-12
        assert cf.risk_neg(pi, fnr, tnr) <= cf.risk_neg(pi, ufnr, ltnr) + 1e-12


def test_risk_zero_denominator_is_not_certifiable():
    assert cf.risk_pos(0.3, 0.2, 0.0) == 1.0 and cf.risk_neg(0.3, 0.2, 0.0) == 1.0
    assert cf.risk_pos(0.3, 0.0, 0.0) == 1.0


@pytest.mark.parametrize("n", [5, 10, 30, 80])
@pytest.mark.parametrize("p", [0.0, 0.01, 0.1, 0.4, 0.9, 1.0])
def test_clopper_pearson_exact_level(n, p):
    k = np.arange(n + 1)
    pmf = stats.binom.pmf(k, n, p)
    assert pmf[cf.cp_upper(k, n, 0.05) < p].sum() <= 0.05 + 1e-12
    assert pmf[cf.cp_lower(k, n, 0.05) > p].sum() <= 0.05 + 1e-12


# ---------------------------------------------------------------------------------------
# Selección
# ---------------------------------------------------------------------------------------
def test_trivial_certification_excluded():
    assert cf.informative(0.10, 0.10) == (True, False)
    assert cf.informative(0.95, 0.10) == (False, False)
    rng = np.random.default_rng(0)
    s0, s1 = rng.normal(0, 1, 60), rng.normal(0, 1, 60)
    for m in cf.METHODS:
        res = cf.certify(s0, s1, m, [0.1, 0.5], [0.05, 0.10, 0.20], rng)
        assert all(r["frac"] == 0.0 for r in res.values()), m


def test_choose_pair_disjoint_and_certified():
    rng = np.random.default_rng(2)
    for _ in range(1000):
        T = 30
        okp, okn = rng.random(T) < 0.4, rng.random(T) < 0.4
        jp, jn = cf.choose_pair(okp, okn, rng.random(T), rng.random(T))
        assert jp is None or okp[jp]
        assert jn is None or okn[jn]
        assert jp is None or jn is None or jn < jp


# ---------------------------------------------------------------------------------------
# split: B no interviene en candidatos, orden ni criterio de eficiencia
# ---------------------------------------------------------------------------------------
def test_split_plan_depends_only_on_A_via_certify_split():
    rng = np.random.default_rng(5)
    a0, a1 = rng.normal(0, 1, 40), rng.normal(2.5, 1, 40)
    bB1 = (rng.normal(0, 1, 40), rng.normal(2.5, 1, 40))      # B informativo
    bB2 = (rng.normal(0, 1, 40), rng.normal(0.3, 1, 40))      # B casi sin información
    pis, alphas = [0.1, 0.3, 0.5], [0.05, 0.10, 0.20]
    out1, t1, plan1, dbg1 = cf.certify_split_halves(a0, bB1[0], a1, bB1[1], pis, alphas,
                                                     return_debug=True)
    out2, t2, plan2, dbg2 = cf.certify_split_halves(a0, bB2[0], a1, bB2[1], pis, alphas,
                                                     return_debug=True)
    # candidatos: idénticos y sin ningún valor de B
    assert np.array_equal(t1, t2)
    assert not np.isin(t1, np.concatenate([*bB1, *bB2])).any()
    for c in plan1:
        # orden y criterio de eficiencia: idénticos
        for key in ("order_pos", "order_neg", "eff_pos", "eff_neg"):
            assert np.array_equal(plan1[c][key], plan2[c][key]), (c, key)
        # lo certificado es un prefijo del orden de A y el umbral elegido está certificado
        for dbg in (dbg1, dbg2):
            cp_, cn_ = dbg[c]["cert_pos"], dbg[c]["cert_neg"]
            assert np.array_equal(cp_, plan1[c]["order_pos"][:len(cp_)])
            assert np.array_equal(cn_, plan1[c]["order_neg"][:len(cn_)])
            assert dbg[c]["jp"] is None or dbg[c]["jp"] in cp_
            assert dbg[c]["jn"] is None or dbg[c]["jn"] in cn_
    # B sí cambia lo que se certifica en al menos una celda
    assert any(len(dbg1[c]["cert_pos"]) != len(dbg2[c]["cert_pos"]) or
               len(dbg1[c]["cert_neg"]) != len(dbg2[c]["cert_neg"]) for c in plan1)


def test_fixed_sequence_stops_at_first_failure():
    order = np.array([3, 1, 4, 0, 2])
    ur = np.array([0.01, 0.02, 0.01, 0.03, 0.5])
    assert list(cf.fixed_sequence(order, ur, 0.1)) == [3, 1]
    assert list(cf.fixed_sequence(np.array([], int), ur, 0.1)) == []


# ---------------------------------------------------------------------------------------
# Validez de punta a punta (DEMOSTRACIÓN: los procedimientos son conservadores)
# ---------------------------------------------------------------------------------------
E2E = [(stats.norm(0, 1), stats.norm(2.5, 1), 80, 80),
       (stats.binom(12, 0.3), stats.binom(12, 0.65), 80, 80),
       (cf.rounded_normal(0.0), cf.rounded_normal(2.5), 80, 80),
       (stats.norm(0, 1), stats.norm(2.5, 1), 90, 30)]


@pytest.mark.parametrize("method", ["os", "dkw", "split"])
@pytest.mark.parametrize("d0,d1,n0,n1", E2E)
def test_end_to_end_false_certification(method, d0, d1, n0, n1):
    rng = np.random.default_rng(11)
    reps, pis, alphas = 400, [0.1, 0.3, 0.5], [0.05, 0.10, 0.20]
    per_cell = {(a, p): 0 for a in alphas for p in pis}
    joint = 0
    for _ in range(reps):
        s0 = d0.rvs(n0, random_state=rng).astype(float)
        s1 = d1.rvs(n1, random_state=rng).astype(float)
        res = cf.certify(s0, s1, method, pis, alphas, rng)
        hit = False
        for c, r in res.items():
            tr = cf.true_risks(d0, d1, r, c[1])
            v = (np.isfinite(r["l_pos"]) and tr["R_pos"] > c[0]) or \
                (np.isfinite(r["l_neg"]) and tr["R_neg"] > c[0])
            per_cell[c] += bool(v)
            hit |= bool(v)
        joint += hit
    worst = max(per_cell.values())
    assert upper(worst, reps) <= cf.DELTA, ("por celda", worst / reps)
    if method != "split":
        assert upper(joint, reps) <= cf.DELTA, ("conjunta", joint / reps)


# ---------------------------------------------------------------------------------------
# test_metrics: cobertura simultánea de los IC (DEMOSTRACIÓN; Bonferroni es conservador)
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("d0,d1", [(stats.norm(0, 1), stats.norm(2, 1)),
                                    (stats.binom(12, 0.3), stats.binom(12, 0.6))])
def test_test_metrics_simultaneous_coverage(d0, d1):
    rng = np.random.default_rng(4)
    pi, reps, n0, n1 = 0.3, 3000, 40, 30
    r = dict(l_neg=float(d0.ppf(0.5)), l_pos=float(d1.ppf(0.4)))
    tr = cf.true_risks(d0, d1, r, pi)
    true_frac = tr["f_pos"] + tr["f_neg"]
    miss = 0
    for _ in range(reps):
        t0 = d0.rvs(n0, random_state=rng).astype(float)
        t1 = d1.rvs(n1, random_state=rng).astype(float)
        m = cf.test_metrics(t0, t1, r, pi, conf=0.95)
        ok = (m["riesgo_obs_pos_ic_inf"] - 1e-12 <= tr["R_pos"] <= m["riesgo_obs_pos_ic_sup"] + 1e-12 and
              m["riesgo_obs_neg_ic_inf"] - 1e-12 <= tr["R_neg"] <= m["riesgo_obs_neg_ic_sup"] + 1e-12 and
              m["frac_test_ic_inf"] - 1e-12 <= true_frac <= m["frac_test_ic_sup"] + 1e-12 and
              m["abstencion_test_ic_inf"] - 1e-12 <= 1 - true_frac <= m["abstencion_test_ic_sup"] + 1e-12)
        miss += not ok
    assert upper(miss, reps) <= 0.05, miss / reps


# ---------------------------------------------------------------------------------------
# Datos: validaciones de entrada y ejecución completa
# ---------------------------------------------------------------------------------------
def _write_cohort(path, n_pat=80, reps=2, missing_id=False, inconsistent=False, seed=0):
    rng = np.random.default_rng(seed)
    wn = np.arange(900, 1800, 10.0)
    rows = []
    for p in range(n_pat):
        y = int(p < n_pat // 2)
        for r in range(reps):
            lab = 1 - y if (inconsistent and p == 0 and r == 1) else y
            s = np.exp(-((wn - 1650) / 40) ** 2) + 0.08 * y * np.exp(-((wn - 1030) / 25) ** 2) \
                + rng.normal(0, 0.01, len(wn))
            pid = "" if (missing_id and p == 3) else f"P{p}"
            rows.append([pid, lab, *s])
    pd.DataFrame(rows, columns=["patient_id", "label", *[f"{w:g}" for w in wn]]).to_csv(path, index=False)


@pytest.mark.parametrize("kw,ranges", [(dict(missing_id=True), [(900, 1800)]),
                                       (dict(inconsistent=True), [(900, 1800)]),
                                       (dict(), [(5000, 6000)])])
def test_load_rejects_bad_input(tmp_path, kw, ranges):
    f = tmp_path / "c.csv"
    _write_cohort(f, **kw)
    with pytest.raises(SystemExit):
        cf.load_cohort(str(f), ranges)


def test_run_data_end_to_end(tmp_path, monkeypatch):
    f = tmp_path / "c.csv"
    _write_cohort(f)
    monkeypatch.chdir(tmp_path)
    cf.run_data(str(f), reps=2, fr_train=0.5, fr_cal=0.3, wn_ranges=[(900, 1800)])
    det = pd.read_csv(tmp_path / "viabilidad_c_detalle.csv")
    assert set(det.metodo) == set(cf.METHODS)
    assert det.frac_cal.between(0, 1).all()
    for side in ("pos", "neg"):
        ok = det[f"riesgo_obs_{side}"].notna()
        assert (det.loc[ok, f"riesgo_obs_{side}_ic_inf"] <= det.loc[ok, f"riesgo_obs_{side}"] + 1e-12).all()
        assert (det.loc[ok, f"riesgo_obs_{side}"] <= det.loc[ok, f"riesgo_obs_{side}_ic_sup"] + 1e-12).all()


def test_run_data_without_test_set(tmp_path, monkeypatch):
    f = tmp_path / "c.csv"
    _write_cohort(f)
    monkeypatch.chdir(tmp_path)
    cf.run_data(str(f), reps=1, fr_train=0.4, fr_cal=0.6, wn_ranges=[(900, 1800)])
    det = pd.read_csv(tmp_path / "viabilidad_c_detalle.csv")
    assert det.frac_test.isna().all() and (det.n_test == 0).all()


# ---------------------------------------------------------------------------------------
# Control de atribución
# ---------------------------------------------------------------------------------------
def _write_attr_cohort(path, where, n_pat=400, seed=3):
    """Señal de clase solo en una banda bioquímica (where='bio') o solo en la pendiente de la
    línea base de una región sin bandas de la muestra (where='null'). Un desplazamiento plano
    no serviría: la normalización SNV lo elimina."""
    rng = np.random.default_rng(seed)
    wn = np.arange(900, 4000, 10.0)
    base = np.exp(-((wn - 1650) / 40) ** 2) + 0.5 * np.exp(-((wn - 2930) / 30) ** 2)
    rows = []
    for p in range(n_pat):
        y = int(p < n_pat // 2)
        s = base * (1 + 0.05 * rng.normal()) + rng.normal(0, 0.004, len(wn))
        if where == "bio":
            s += (0.06 * y + 0.02 * rng.normal()) * np.exp(-((wn - 1030) / 20) ** 2)
        else:
            ramp = np.clip((wn - 2400) / 300, 0, 1) * ((wn >= 2400) & (wn <= 2700))
            s += (0.03 * y + 0.01 * rng.normal()) * ramp   # pendiente de línea base (SNV no la elimina)
        rows.append([f"P{p}", y, *s])
    pd.DataFrame(rows, columns=["patient_id", "label", *[f"{w:g}" for w in wn]]).to_csv(path, index=False)


@pytest.mark.parametrize("where,expect_flag", [("bio", False), ("null", True)])
def test_attribution_flags_only_non_band_signal(tmp_path, monkeypatch, where, expect_flag):
    f = tmp_path / "a.csv"
    _write_attr_cohort(f, where)
    monkeypatch.chdir(tmp_path)
    cf.run_data(str(f), reps=1, fr_train=0.5, fr_cal=0.5, wn_ranges=[(900, 1800), (2800, 3700)] if where == "bio"
                else [(900, 1800), (2400, 2700), (2800, 3700)],
                null_ranges=[(2400, 2700), (3800, 3990)] if where == "bio" else [(2400, 2700)])
    det = pd.read_csv(tmp_path / "viabilidad_a_detalle.csv")
    cert = det[det.atribucion != "sin_certificacion"]
    assert len(cert) > 0, "el análisis principal debería certificar con esta señal"
    flagged = (cert.atribucion == "no_atribuible_a_bandas").any()
    assert flagged == expect_flag


def test_attribution_label():
    assert cf.attribution_label(False, True) == "sin_certificacion"
    assert cf.attribution_label(True, True) == "no_atribuible_a_bandas"
    assert cf.attribution_label(True, False) == "sin_alerta"


def test_run_data_without_null_has_no_attribution_columns(tmp_path, monkeypatch):
    f = tmp_path / "c.csv"
    _write_cohort(f)
    monkeypatch.chdir(tmp_path)
    cf.run_data(str(f), reps=1, fr_train=0.5, fr_cal=0.3, wn_ranges=[(900, 1800)])
    det = pd.read_csv(tmp_path / "viabilidad_c_detalle.csv")
    assert "atribucion" not in det.columns
