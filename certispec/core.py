#!/usr/bin/env python3
"""
CertiSpec — core module (version 1.0.0)

Certificación de VPP y VPN a prevalencia objetivo para clasificación de espectros FTIR
por paciente: tres métodos, validación por simulación y aplicación a cohortes.

==========================================================================================
REGLA DE DECISIÓN Y CONVENCIONES
  F(t) = P(S <= t) (continua por la derecha). Umbrales l- < l+.
  positivo si S > l+ ; negativo si S <= l- ; no concluyente si l- < S <= l+.
  FPR(l) = 1 - F-(l)   TPR(l) = 1 - F+(l)   TNR(l) = F-(l)   FNR(l) = F+(l)
  R+(pi,l) = P(Y=0 | S > l)  = (1-pi)FPR / [(1-pi)FPR + pi TPR]      (= 1 - VPP)
  R-(pi,l) = P(Y=1 | S <= l) = pi FNR / [pi FNR + (1-pi)TNR]          (= 1 - VPN)
  R+ es creciente en FPR y decreciente en TPR; R- es creciente en FNR y decreciente en TNR.
  Si el denominador de la cota es 0, la cota es 1 (no certificable).
  Un lado solo se considera si es informativo: alpha < 1-pi (positivo), alpha < pi
  (negativo); si no, "todos positivos" o "todos negativos" se certificaría sin datos.

SUPUESTOS
  (S1) Puntajes por paciente de calibración i.i.d. dentro de cada clase (F- o F+),
       independientes del ajuste del modelo (pacientes disjuntos).
  (S2) Transportabilidad: un paciente nuevo, condicionado a su clase, sigue F- o F+;
       la prevalencia pi la fija el usuario. (S2) no la verifica el software.
  Bajo (S1) los casos y controles pueden muestrearse con cualquier proporción.

GARANTÍAS DEMOSTRADAS (no dependen de las simulaciones)

  [os] Sea F^{-1}(u) = inf{x : F(x) >= u}. Hechos: F(F^{-1}(u)) >= u (continuidad por la
  derecha) y F(x) < u para todo x < F^{-1}(u), luego F(F^{-1}(u)-) <= u. Como F^{-1} es no
  decreciente, (S_(1),...,S_(n)) tiene la misma distribución que (F^{-1}(U_(1)),...),
  con U_i uniformes i.i.d. Sea k = #{i : S_i <= t} (0..n; con empates cuentan todos).
  Entonces S_(k) <= t < S_(k+1) (convenciones S_(0) = -inf, S_(n+1) = +inf), y en el
  evento E = {a_i <= U_(i) <= b_i, i = 1..n}:
      F(t) >= F(S_(k)) >= U_(k) >= a_k            (k >= 1; si k = 0, cota trivial 0)
      F(t) <= F(S_(k+1)-) <= U_(k+1) <= b_(k+1)   (k <= n-1; si k = n, cota trivial 1)
  Por tanto P(L(t) <= F(t) <= U(t) para todo t) >= P(E), para CUALQUIER F (continua,
  discreta o mixta, con o sin empates), con igualdad si F es continua. P(E) se calcula
  exactamente por programación dinámica: E equivale a N(b_i) >= i y N(a_i) <= i-1 para todo
  i (N(t) = #{U_j <= t}; difiere de U_(i) >= a_i en un evento de probabilidad 0), y dado
  N(t_j) = k, N(t_{j+1}) - k ~ Bin(n-k, (t_{j+1}-t_j)/(1-t_j)). El nivel local g se elige
  por bisección con la condición P(E) >= 1 - delta/2, verificada en el valor final.
  Bandas de ambas clases (muestras independientes): las dos correctas con prob.
  >= (1-delta/2)^2 >= 1-delta. En ese evento, por monotonía, UR+ >= R+ y UR- >= R- para
  TODO l y TODA pi. Por tanto el umbral y la prevalencia pueden elegirse después de ver los
  datos: la garantía es SIMULTÁNEA sobre todas las celdas (alpha, pi) y todos los umbrales.

  [dkw] Igual, con la banda DKW de constante de Massart: P(sup_t |F_n - F| > eps)
  <= 2 exp(-2 n eps^2), válida para cualquier F cuando exp(-2 n eps^2) <= 1/2 (aquí
  2 exp(-2 n eps^2) = delta/2 <= 1). Garantía simultánea, igual que [os].

  [split] Partición aleatoria A/B independiente de los datos. Fijada una celda (alpha, pi)
  y un lado (positivo): candidatos T, orden sigma y criterio de eficiencia dependen solo
  de A. Condicionado a A, cada l de T es fijo y B es independiente de A. Para H0_l:
  R+(pi,l) > alpha, la prueba "UR+_B(l) <= alpha" (Clopper-Pearson unilateral a delta/4
  para FPR y para TPR) rechaza H0_l solo si U_FPR < FPR(l) o L_TPR > TPR(l); cada evento
  tiene probabilidad <= delta/4 (CP exacta), así que el nivel es <= delta/2. Secuencia
  fija: se prueba en el orden sigma y se detiene en el primer no rechazo. Si j* es la
  primera hipótesis nula verdadera en sigma, cualquier certificación falsa exige rechazar
  j*, luego P(alguna certificación falsa) <= delta/2 (Bauer 1991; Learn Then Test).
  Lado negativo: igual. Ambos lados: <= delta. Cualquier elección entre los umbrales
  certificados conserva la garantía. ALCANCE: la garantía es POR CELDA (alpha, pi). No hay
  garantía simultánea entre celdas: con C celdas, la unión solo da <= C*delta.

  [test_metrics] Intervalos en prueba independiente, condicionados a los umbrales (fijados
  sin usar la prueba): 8 cotas CP unilaterales a (1-conf)/8 (FPR, TPR en l+; TNR, FNR en
  l-; inferior y superior). Con probabilidad >= conf todas son correctas a la vez y, por
  monotonía, los intervalos de R+, R-, fracción clasificada y abstención son
  SIMULTÁNEAMENTE válidos al nivel conf para esa celda y esa partición. No son
  simultáneos entre celdas ni entre repeticiones.

CONTROL DE ATRIBUCIÓN (diagnóstico, opción --null-ranges)
  Con las mismas particiones, se ajusta el mismo clasificador usando SOLO regiones sin bandas
  de la muestra y se certifica con los mismos métodos y celdas. Una certificación del análisis
  principal se marca 'no_atribuible_a_bandas' si el modelo nulo también certifica en esa celda.
  No es una prueba con garantía: la alerta indica que la información que permite certificar no
  está confinada a las bandas bioquímicas; su ausencia no prueba atribución química.

RESULTADOS EMPÍRICOS (validate, coverage) evalúan eficiencia y detectan errores de
implementación; no sustituyen las demostraciones.
==========================================================================================

Uso
  certispec validate [--reps 2000] [--fast] [--scenario NOMBRE ...]
  certispec report      (combina validate --scenario por partes)
  certispec external train.csv test.csv [--reps 50] [--cal 0.6]
  certispec coverage [--reps 20000]
  certispec data cohorte.csv [--reps 50] [--train 0.5 --cal 0.3]
        (--train 0.4 --cal 0.6 deja la prueba vacía y maximiza la calibración)
        [--wn-ranges 900-1800,2800-3700] [--null-ranges 2400-2700,3800-4000]

CSV: una fila por espectro; columnas 'patient_id', 'label' (0/1) y una columna por
absorbancia (encabezados = números de onda si se usa --wn-ranges).
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = [0.05, 0.10, 0.15, 0.20]
PIS = [0.05, 0.10, 0.20, 0.30, 0.50]
DELTA = 0.05
METHODS = ["os", "dkw", "split"]
MIN_CAL_PER_CLASS = 5
C_GRID = np.logspace(-4, 1, 6)


# ========================================================================================
# Banda por estadísticas de orden: cobertura exacta
# ========================================================================================
def os_exact_coverage(a, b):
    """P(a_i <= U_(i) <= b_i para todo i), U_i ~ U(0,1) i.i.d., por programación dinámica."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    if n == 0:
        return 1.0
    if np.any(a > b) or np.any(a < 0) or np.any(b > 1):
        return 0.0 if np.any(a > b) else os_exact_coverage(np.clip(a, 0, 1), np.clip(b, 0, 1))
    pts = np.concatenate([a, b])
    kind = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    idx = np.concatenate([np.arange(1, n + 1), np.arange(1, n + 1)])
    o = np.lexsort((kind, pts))
    pts, kind, idx = pts[o], kind[o], idx[o]
    lf = gammaln(np.arange(n + 2) + 1.0)
    v = np.zeros(n + 1)
    v[0] = 1.0
    lo, hi = 0, 0
    t_prev = 0.0
    for t, kd, i in zip(pts, kind, idx):
        if t > t_prev:
            p = min(max((t - t_prev) / (1.0 - t_prev), 0.0), 1.0) if t_prev < 1 else 0.0
            k = np.arange(lo, hi + 1)[:, None]
            kp = np.arange(lo, n + 1)[None, :]
            d = kp - k
            r = n - k
            ok = (d >= 0) & (d <= r)
            dd, rr = np.where(ok, d, 0), np.where(ok, r, 0)
            if p >= 1.0:
                M = (ok & (dd == rr)).astype(float)
            elif p <= 0.0:
                M = (ok & (dd == 0)).astype(float)
            else:
                logM = lf[rr] - lf[dd] - lf[rr - dd] + dd * np.log(p) + (rr - dd) * np.log1p(-p)
                M = np.where(ok, np.exp(np.where(ok, logM, 0.0)), 0.0)
            nv = np.zeros(n + 1)
            nv[lo:] = v[lo:hi + 1] @ M
            v = nv
            t_prev = t
        if kd == 0:
            v[i:] = 0.0
        else:
            v[:i] = 0.0
        nz = np.nonzero(v > 0)[0]
        if len(nz) == 0:
            return 0.0
        lo, hi = int(nz[0]), int(nz[-1])
    return float(min(max(v.sum(), 0.0), 1.0))


def beta_band(n, g):
    i = np.arange(1, n + 1)
    return stats.beta.ppf(g / 2.0, i, n - i + 1), stats.beta.ppf(1.0 - g / 2.0, i, n - i + 1)


_OS_CACHE = {}


def os_band_levels(n, eta, iters=30):
    """Mayor g (por bisección en log g) con cobertura exacta >= 1 - eta; se verifica al final."""
    key = (n, eta)
    if key in _OS_CACHE:
        return _OS_CACHE[key]
    lo, hi = np.log(1e-14), np.log(eta)
    if os_exact_coverage(*beta_band(n, np.exp(lo))) < 1.0 - eta:
        raise RuntimeError(f"No se alcanza cobertura {1 - eta} con n={n}")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if os_exact_coverage(*beta_band(n, np.exp(mid))) >= 1.0 - eta:
            lo = mid
        else:
            hi = mid
    g = np.exp(lo)
    a, b = beta_band(n, g)
    cov = os_exact_coverage(a, b)
    if cov < 1.0 - eta:
        raise RuntimeError(f"Cobertura exacta {cov:.8f} < {1 - eta} para n={n}")
    _OS_CACHE[key] = (a, b, g, cov)
    return _OS_CACHE[key]


def band_os(scores, eta):
    s = np.sort(np.asarray(scores, float))
    a, b, _, _ = os_band_levels(len(s), eta)
    return ("os", s, a, b)


def band_dkw(scores, eta):
    s = np.sort(np.asarray(scores, float))
    eps = np.sqrt(np.log(2.0 / eta) / (2.0 * len(s)))
    return ("dkw", s, eps, None)


def band_eval(band, t):
    """L(t) <= F(t) <= U(t), F(t) = P(S <= t); k = #{S_i <= t} cuenta todos los empates."""
    kind, s, p1, p2 = band
    n = len(s)
    k = np.searchsorted(s, np.asarray(t, float), side="right")
    if kind == "os":
        L = np.where(k >= 1, p1[np.clip(k - 1, 0, n - 1)], 0.0)
        U = np.where(k <= n - 1, p2[np.clip(k, 0, n - 1)], 1.0)
    else:
        fn = k / n
        L = np.clip(fn - p1, 0.0, 1.0)
        U = np.clip(fn + p1, 0.0, 1.0)
    return L, U


# ========================================================================================
# Cotas de riesgo y utilidades
# ========================================================================================
def risk_pos(pi, ufpr, ltpr):
    ufpr, ltpr = np.asarray(ufpr, float), np.asarray(ltpr, float)
    num = (1 - pi) * ufpr
    den = num + pi * ltpr
    return np.where((ltpr > 0) & (den > 0), num / np.where(den > 0, den, 1.0), 1.0)


def risk_neg(pi, ufnr, ltnr):
    ufnr, ltnr = np.asarray(ufnr, float), np.asarray(ltnr, float)
    num = pi * ufnr
    den = num + (1 - pi) * ltnr
    return np.where((ltnr > 0) & (den > 0), num / np.where(den > 0, den, 1.0), 1.0)


def cp_upper(x, n, d):
    """Cota superior CP unilateral: P(p > cota) <= d."""
    x = np.asarray(x)
    return np.where(x < n, stats.beta.ppf(1 - d, x + 1, np.maximum(n - x, 1)), 1.0)


def cp_lower(x, n, d):
    """Cota inferior CP unilateral: P(p < cota) <= d."""
    x = np.asarray(x)
    return np.where(x > 0, stats.beta.ppf(d, np.maximum(x, 1), n - x + 1), 0.0)


def cp_ci(k, n, conf=0.95):
    """IC Clopper-Pearson bilateral (inferior, superior)."""
    d = (1 - conf) / 2
    return float(cp_lower(k, n, d)), float(cp_upper(k, n, d))


def cp_upper1(k, n, conf=0.95):
    """Cota superior unilateral al nivel conf."""
    return float(cp_upper(k, n, 1 - conf))


def candidate_thresholds(s0, s1):
    p = np.unique(np.concatenate([s0, s1]))
    mid = (p[:-1] + p[1:]) / 2.0
    return np.concatenate([[p[0] - 1.0], mid, [p[-1] + 1.0]])


def empirical_F(s0, s1, t):
    F0 = np.searchsorted(np.sort(s0), t, side="right") / len(s0)
    F1 = np.searchsorted(np.sort(s1), t, side="right") / len(s1)
    return F0, F1


def informative(alpha, pi):
    return alpha < 1 - pi, alpha < pi


def choose_pair(okp, okn, frac_pos, frac_neg):
    """(j-, j+) con j- < j+ (l- < l+), ambos en su conjunto certificado, máxima fracción."""
    fp = np.where(okp, frac_pos, 0.0)
    fn = np.where(okn, frac_neg, 0.0)
    T = len(fp)
    best_before = np.zeros(T)
    arg_before = np.full(T, -1)
    cur, arg = 0.0, -1
    for j in range(T):
        best_before[j], arg_before[j] = cur, arg
        if fn[j] > cur:
            cur, arg = fn[j], j
    tot = fp + best_before
    j = int(np.argmax(tot))
    if fn.max() > tot[j]:
        return None, int(np.argmax(fn))
    jp = j if fp[j] > 0 else None
    jn = int(arg_before[j]) if best_before[j] > 0 else None
    return jp, jn


def pack(t, jp, jn, pi, F0, F1, ur_pos, ur_neg):
    """F0, F1: empíricas usadas solo para REPORTAR la fracción estimada en calibración."""
    fpos = pi * (1 - F1[jp]) + (1 - pi) * (1 - F0[jp]) if jp is not None else 0.0
    fneg = pi * F1[jn] + (1 - pi) * F0[jn] if jn is not None else 0.0
    return dict(l_pos=t[jp] if jp is not None else np.inf,
                l_neg=t[jn] if jn is not None else -np.inf,
                frac_pos=fpos, frac_neg=fneg, frac=fpos + fneg,
                ur_pos_cert=float(ur_pos[jp]) if jp is not None else np.nan,
                ur_neg_cert=float(ur_neg[jn]) if jn is not None else np.nan,
                min_ur_pos=float(np.min(ur_pos)), min_ur_neg=float(np.min(ur_neg)))


# ========================================================================================
# Certificación por bandas (garantía simultánea)
# ========================================================================================
def certify_bands(s0, s1, method, pis, alphas, delta=DELTA):
    eta = delta / 2.0
    bfun = band_os if method == "os" else band_dkw
    b0, b1 = bfun(s0, eta), bfun(s1, eta)
    t = candidate_thresholds(s0, s1)
    L0, U0 = band_eval(b0, t)
    L1, U1 = band_eval(b1, t)
    ufpr, ltpr, ufnr, ltnr = 1 - L0, 1 - U1, U1, L0
    F0, F1 = empirical_F(s0, s1, t)
    out = {}
    for pi in pis:
        urp, urn = risk_pos(pi, ufpr, ltpr), risk_neg(pi, ufnr, ltnr)
        fpos = pi * (1 - F1) + (1 - pi) * (1 - F0)
        fneg = pi * F1 + (1 - pi) * F0
        for al in alphas:
            ip, ineg = informative(al, pi)
            jp, jn = choose_pair((urp <= al) & ip, (urn <= al) & ineg, fpos, fneg)
            out[(al, pi)] = pack(t, jp, jn, pi, F0, F1, urp, urn)
    return out


# ========================================================================================
# split: Learn Then Test con secuencia fija (garantía por celda)
# ========================================================================================
def split_halves(s0, s1, rng):
    p0, p1 = rng.permutation(s0), rng.permutation(s1)
    h0, h1 = len(p0) // 2, len(p1) // 2
    return p0[:h0], p0[h0:], p1[:h1], p1[h1:]


def cp_rate_bounds(x0, x1, t, d):
    n0, n1 = len(x0), len(x1)
    k0 = np.searchsorted(np.sort(x0), t, side="right")
    k1 = np.searchsorted(np.sort(x1), t, side="right")
    return (cp_upper(n0 - k0, n0, d), cp_lower(n1 - k1, n1, d),
            cp_upper(k1, n1, d), cp_lower(k0, n0, d))


def fixed_sequence(order, ur_B, alpha):
    """Prefijo de 'order' hasta el primer umbral con UR_B > alpha (sin incluirlo)."""
    order = np.asarray(order, int)
    fails = np.nonzero(ur_B[order] > alpha)[0]
    return order[:fails[0] if len(fails) else len(order)]


def split_plan(a0, a1, pis, alphas, delta=DELTA):
    """TODO lo que depende de A: candidatos, orden por celda y lado, criterio de eficiencia."""
    d4 = delta / 4.0
    t = candidate_thresholds(a0, a1)
    ufpA, ltpA, ufnA, ltnA = cp_rate_bounds(a0, a1, t, d4)
    F0A, F1A = empirical_F(a0, a1, t)
    plan = {}
    for pi in pis:
        urpA, urnA = risk_pos(pi, ufpA, ltpA), risk_neg(pi, ufnA, ltnA)
        effp = pi * (1 - F1A) + (1 - pi) * (1 - F0A)
        effn = pi * F1A + (1 - pi) * F0A
        for al in alphas:
            ip, ineg = informative(al, pi)
            plan[(al, pi)] = dict(
                order_pos=np.argsort(urpA, kind="stable") if ip else np.array([], int),
                order_neg=np.argsort(urnA, kind="stable") if ineg else np.array([], int),
                eff_pos=effp, eff_neg=effn)
    return t, plan


def certify_split_halves(a0, b0, a1, b1, pis, alphas, delta=DELTA, report=None,
                         return_debug=False):
    """Certificación con B. 'report' = (s0, s1) para reportar la fracción estimada."""
    t, plan = split_plan(a0, a1, pis, alphas, delta)
    d4 = delta / 4.0
    ufpB, ltpB, ufnB, ltnB = cp_rate_bounds(b0, b1, t, d4)
    s0r, s1r = report if report is not None else (np.concatenate([a0, b0]), np.concatenate([a1, b1]))
    F0, F1 = empirical_F(s0r, s1r, t)
    out, dbg = {}, {}
    for pi in pis:
        urpB, urnB = risk_pos(pi, ufpB, ltpB), risk_neg(pi, ufnB, ltnB)
        for al in alphas:
            pl = plan[(al, pi)]
            cert_p = fixed_sequence(pl["order_pos"], urpB, al)
            cert_n = fixed_sequence(pl["order_neg"], urnB, al)
            okp = np.zeros(len(t), bool)
            okn = np.zeros(len(t), bool)
            okp[cert_p] = True
            okn[cert_n] = True
            jp, jn = choose_pair(okp, okn, pl["eff_pos"], pl["eff_neg"])
            out[(al, pi)] = pack(t, jp, jn, pi, F0, F1, urpB, urnB)
            dbg[(al, pi)] = dict(cert_pos=cert_p, cert_neg=cert_n, jp=jp, jn=jn)
    if return_debug:
        return out, t, plan, dbg
    return out


def certify_split(s0, s1, pis, alphas, delta=DELTA, rng=None):
    rng = rng or np.random.default_rng(0)
    a0, b0, a1, b1 = split_halves(s0, s1, rng)
    return certify_split_halves(a0, b0, a1, b1, pis, alphas, delta, report=(s0, s1))


def certify(s0, s1, method, pis, alphas, rng):
    if method == "split":
        return certify_split(s0, s1, pis, alphas, rng=rng)
    return certify_bands(s0, s1, method, pis, alphas)


# ========================================================================================
# Escenarios de simulación
# ========================================================================================
def rounded_normal(mu, step=0.5, sd=1.0):
    """Normal discretizada a una malla: empates muy frecuentes."""
    xk = np.round(np.arange(mu - 8 * sd, mu + 8 * sd + step, step), 10)
    pk = stats.norm.cdf(xk + step / 2, mu, sd) - stats.norm.cdf(xk - step / 2, mu, sd)
    return stats.rv_discrete(values=(xk, pk / pk.sum()))()  # distribución congelada


def scenarios():
    return {
        "normal_mu2": (stats.norm(0, 1), stats.norm(2, 1), 1.0),
        "normal_mu3": (stats.norm(0, 1), stats.norm(3, 1), 1.0),
        "lognormal": (stats.lognorm(s=0.6, scale=1.0), stats.lognorm(s=0.6, scale=np.exp(1.4)), 1.0),
        "t3_colas": (stats.t(3, 0, 1), stats.t(3, 2.5, 1), 1.0),
        "binomial": (stats.binom(12, 0.30), stats.binom(12, 0.65), 1.0),
        "poisson": (stats.poisson(3.0), stats.poisson(8.0), 1.0),
        "empates_malla0.5": (rounded_normal(0.0), rounded_normal(2.5), 1.0),
        "desbalance_1a3": (stats.norm(0, 1), stats.norm(2.5, 1), 1 / 3),
    }


def true_risks(d0, d1, r, pi):
    """Riesgos y fracciones VERDADEROS de los umbrales elegidos (regla S > l+, S <= l-)."""
    out = dict(R_pos=np.nan, R_neg=np.nan, f_pos=0.0, f_neg=0.0)
    if np.isfinite(r["l_pos"]):
        fpr, tpr = d0.sf(r["l_pos"]), d1.sf(r["l_pos"])
        den = (1 - pi) * fpr + pi * tpr
        out["f_pos"] = den
        out["R_pos"] = (1 - pi) * fpr / den if den > 0 else 0.0
    if np.isfinite(r["l_neg"]):
        tnr, fnr = d0.cdf(r["l_neg"]), d1.cdf(r["l_neg"])
        den = pi * fnr + (1 - pi) * tnr
        out["f_neg"] = den
        out["R_neg"] = pi * fnr / den if den > 0 else 0.0
    return out


# ========================================================================================
# validate: tasa de falsas certificaciones y comparación de eficiencia
# ========================================================================================
def validate(reps, fast=False, seed=0, only=None):
    """only: lista de escenarios (para correr por partes); cada escenario guarda sus propios
    archivos y 'report' los combina."""
    rng = np.random.default_rng(seed if not only else seed + sum(map(ord, "".join(only))))
    pis_sim = [0.10, 0.30, 0.50]
    ns = (25, 50, 100) if fast else (25, 50, 100, 250)
    cells = [(a, p) for a in ALPHAS for p in pis_sim]
    rows, joint = [], []
    for name, (d0, d1, ratio) in scenarios().items():
        if only and name not in only:
            continue
        for n in ns:
            n1 = max(int(round(n * ratio)), 3)
            for m in METHODS:
                t0 = time.time()
                acc = {c: dict(v=0, fp=[], fn=[], urp=[], urn=[], cp=0, cn=0) for c in cells}
                any_v = 0
                for _ in range(reps):
                    s0 = d0.rvs(n, random_state=rng).astype(float)
                    s1 = d1.rvs(n1, random_state=rng).astype(float)
                    res = certify(s0, s1, m, pis_sim, ALPHAS, rng)
                    hit = False
                    for c in cells:
                        al, pi = c
                        r = res[c]
                        tr = true_risks(d0, d1, r, pi)
                        v = (np.isfinite(r["l_pos"]) and tr["R_pos"] > al) or \
                            (np.isfinite(r["l_neg"]) and tr["R_neg"] > al)
                        a = acc[c]
                        a["v"] += bool(v)
                        a["fp"].append(tr["f_pos"])
                        a["fn"].append(tr["f_neg"])
                        a["cp"] += np.isfinite(r["l_pos"])
                        a["cn"] += np.isfinite(r["l_neg"])
                        a["urp"].append(r["ur_pos_cert"])
                        a["urn"].append(r["ur_neg_cert"])
                        hit |= bool(v)
                    any_v += hit
                for (al, pi), a in acc.items():
                    fp, fn = np.array(a["fp"]), np.array(a["fn"])
                    rows.append(dict(
                        escenario=name, n_neg=n, n_pos=n1, metodo=m, alpha=al, pi=pi,
                        falsas_cert=a["v"] / reps, falsas_cert_ic95_sup=cp_upper1(a["v"], reps),
                        prop_cert_pos=a["cp"] / reps, prop_cert_neg=a["cn"] / reps,
                        frac_pos_media=fp.mean(), frac_neg_media=fn.mean(),
                        frac_media=(fp + fn).mean(), frac_mediana=np.median(fp + fn),
                        abstencion_media=1 - (fp + fn).mean(),
                        riesgo_cert_pos_mediana=np.nanmedian(a["urp"]) if a["cp"] else np.nan,
                        riesgo_cert_neg_mediana=np.nanmedian(a["urn"]) if a["cn"] else np.nan))
                joint.append(dict(escenario=name, n_neg=n, n_pos=n1, metodo=m,
                                  falsas_cert_conjunta=any_v / reps,
                                  ic95_sup=cp_upper1(any_v, reps), reps=reps))
                print(f"  {name:18s} n={n:4d}/{n1:<4d} {m:5s} ({time.time() - t0:.1f}s)", flush=True)
    df, dj = pd.DataFrame(rows), pd.DataFrame(joint)
    tag = "_" + "_".join(only) if only else ""
    df.to_csv(f"validacion_por_celda{tag}.csv", index=False)
    dj.to_csv(f"validacion_conjunta{tag}.csv", index=False)
    if not only:
        report_validation(df, dj, reps)


def report_from_files():
    import glob
    fc = sorted(glob.glob("validacion_por_celda_*.csv")) or ["validacion_por_celda.csv"]
    fj = sorted(glob.glob("validacion_conjunta_*.csv")) or ["validacion_conjunta.csv"]
    df = pd.concat([pd.read_csv(f) for f in fc], ignore_index=True)
    dj = pd.concat([pd.read_csv(f) for f in fj], ignore_index=True)
    report_validation(df, dj, int(dj.reps.min()))


def report_validation(df, dj, reps):
    print(f"\n=== VALIDEZ ({reps} repeticiones por diseño) ===")
    print("Tasa de falsas certificaciones POR CELDA: máximo sobre escenarios, n, alpha y pi;")
    print(f"cota superior unilateral 95 % (Clopper-Pearson). Garantía: <= {DELTA} en os, dkw y split.")
    g = df.groupby("metodo").agg(max_tasa=("falsas_cert", "max"),
                                 max_ic95_sup=("falsas_cert_ic95_sup", "max"))
    print(g.round(4).to_string())
    print("\nTasa de falsas certificaciones CONJUNTA (alguna celda). Garantía: <= 0.05 en os y dkw;")
    print("split NO tiene garantía conjunta (la columna se reporta solo como descripción).")
    print(dj.groupby("metodo").agg(max_tasa=("falsas_cert_conjunta", "max"),
                                   max_ic95_sup=("ic95_sup", "max")).round(4).to_string())
    print("\nPor escenario (máximo sobre n) — tasa conjunta:")
    print(dj.pivot_table(index="escenario", columns="metodo", values="falsas_cert_conjunta",
                         aggfunc="max").round(4).to_string())

    print("\n=== EFICIENCIA (resultado empírico) ===")
    for al, pi in [(0.05, 0.10), (0.10, 0.30), (0.10, 0.50)]:
        sub = df[(df.alpha == al) & (df.pi == pi)]
        print(f"\nalpha={al}, pi={pi}: fracción clasificada media (verdadera) / abstención media")
        print(sub.pivot_table(index=["escenario", "n_neg"], columns="metodo",
                              values="frac_media").round(2).to_string())
    print("\nRiesgo certificado (cota) en los umbrales elegidos, mediana; alpha=0.05, pi=0.10, lado negativo:")
    sub = df[(df.alpha == 0.05) & (df.pi == 0.10)]
    print(sub.pivot_table(index=["escenario", "n_neg"], columns="metodo",
                          values="riesgo_cert_neg_mediana").round(3).to_string())
    print("\nTamaño mínimo por clase (de la malla) con fracción clasificada media >= 0.25:")
    req = []
    for (esc, m, al, pi), sub in df.groupby(["escenario", "metodo", "alpha", "pi"]):
        okn = sub[sub.frac_media >= 0.25].n_neg
        req.append(dict(escenario=esc, metodo=m, alpha=al, pi=pi,
                        n_min=int(okn.min()) if len(okn) else np.nan))
    req = pd.DataFrame(req)
    req.to_csv("validacion_tamano_requerido.csv", index=False)
    sel = req[((req.alpha == 0.05) & (req.pi == 0.10)) | ((req.alpha == 0.10) & (req.pi == 0.30))]
    print(sel.pivot_table(index=["alpha", "pi", "escenario"], columns="metodo",
                          values="n_min").to_string())
    print("\nGuardado: validacion_por_celda.csv, validacion_conjunta.csv, validacion_tamano_requerido.csv")


# ========================================================================================
# coverage: cobertura simultánea empírica de las bandas
# ========================================================================================
def band_coverage(reps, seed=1):
    rng = np.random.default_rng(seed)
    eta = DELTA / 2.0
    dists = {"normal": stats.norm(0, 1), "lognormal": stats.lognorm(s=0.6), "t3": stats.t(3),
             "binomial": stats.binom(12, 0.3), "poisson": stats.poisson(3.0),
             "empates_malla0.5": rounded_normal(0.0)}
    rows = []
    for name, d in dists.items():
        discrete = isinstance(d.dist, stats.rv_discrete)
        for n in (10, 25, 50, 100, 250):
            for m in ("os", "dkw"):
                fail = 0
                for _ in range(reps):
                    x = d.rvs(n, random_state=rng).astype(float)
                    band = band_os(x, eta) if m == "os" else band_dkw(x, eta)
                    if discrete:
                        sup = d.dist.xk if hasattr(d.dist, "xk") else np.arange(d.ppf(1e-12), d.ppf(1 - 1e-12) + 1)
                        sup = np.asarray(sup, float)
                        grid = np.concatenate([sup, sup - 1e-6, [sup.min() - 1, sup.max() + 1]])
                    else:
                        xs = np.sort(x)
                        grid = np.concatenate([np.linspace(d.ppf(1e-7), d.ppf(1 - 1e-7), 500),
                                               xs, xs - 1e-9 * (1 + np.abs(xs))])
                    L, U = band_eval(band, grid)
                    F = d.cdf(grid)
                    fail += bool(((F < L - 1e-12) | (F > U + 1e-12)).any())
                lo, hi = cp_ci(fail, reps)
                rows.append(dict(distribucion=name, discreta=discrete, n=n, banda=m,
                                 no_cobertura=fail / reps, ic95_inf=lo, ic95_sup=hi, nominal=eta))
        print(f"  {name} listo", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("cobertura_bandas.csv", index=False)
    print(f"\nNo cobertura simultánea de cada banda (garantía: <= {eta}; exacta = {eta} en os continua).")
    print(df.round(4).to_string(index=False))
    print("\nGuardado: cobertura_bandas.csv")


# ========================================================================================
# Datos reales
# ========================================================================================
def load_cohort(path, wn_ranges):
    df = pd.read_csv(path)
    for c in ("patient_id", "label"):
        if c not in df.columns:
            sys.exit(f"Falta la columna '{c}'.")
    if df["patient_id"].isna().any() or (df["patient_id"].astype(str).str.strip() == "").any():
        sys.exit("Hay espectros sin 'patient_id'.")
    if df["label"].isna().any():
        sys.exit("Hay espectros sin 'label'.")
    spec_cols = [c for c in df.columns if c not in ("patient_id", "label")]
    if wn_ranges:
        try:
            wn = np.array([float(c) for c in spec_cols])
        except ValueError:
            sys.exit("--wn-ranges requiere encabezados numéricos (números de onda).")
        keep = np.zeros(len(wn), bool)
        for lo, hi in wn_ranges:
            keep |= (wn >= lo) & (wn <= hi)
        spec_cols = [c for c, k in zip(spec_cols, keep) if k]
    if len(spec_cols) < 2:
        sys.exit(f"Solo {len(spec_cols)} variables espectrales tras seleccionar regiones.")
    X = df[spec_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    bad = ~np.isfinite(X).all(1)
    with np.errstate(invalid="ignore"):
        const = np.nanstd(X, axis=1) == 0
    drop = bad | const
    if drop.any():
        print(f"  Aviso: se descartan {bad.sum()} espectros con NaN/inf y {const.sum()} constantes.")
    df, X = df.loc[~drop].reset_index(drop=True), X[~drop]
    if not set(np.unique(df["label"])) <= {0, 1}:
        sys.exit("'label' debe ser 0/1.")
    X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    y = df["label"].to_numpy(int)
    g = df["patient_id"].astype(str).to_numpy()
    nun = pd.Series(y, index=g).groupby(level=0).nunique()
    if (nun > 1).any():
        sys.exit(f"Pacientes con etiquetas inconsistentes: {list(nun[nun > 1].index)[:10]}")
    lab = pd.Series(y, index=g).groupby(level=0).first()
    return X, y, g, lab, len(spec_cols)


def fit_patient_level(X, y, g, lab, rs):
    reps = pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()
    w = 1.0 / reps
    pl = lab.loc[np.unique(g)]
    k = int(min(5, (pl == 0).sum(), (pl == 1).sum()))
    if k < 2:
        raise ValueError("Menos de 2 pacientes por clase en entrenamiento.")
    folds = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=rs).split(X, y, g))
    best_c, best_auc = C_GRID[0], -np.inf
    for C in C_GRID:
        aucs = []
        for tr, va in folds:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
            m.fit(X[tr], y[tr], logisticregression__sample_weight=w[tr])
            sc = pd.Series(m.decision_function(X[va]), index=g[va]).groupby(level=0).mean()
            yy = lab.loc[sc.index].to_numpy()
            if 0 < yy.sum() < len(yy):
                aucs.append(roc_auc_score(yy, sc.to_numpy()))
        if aucs and np.mean(aucs) > best_auc:
            best_c, best_auc = C, np.mean(aucs)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=5000))
    m.fit(X, y, logisticregression__sample_weight=w)
    return m, best_c, best_auc


def patient_scores(model, X, g, pats):
    msk = np.isin(g, pats)
    return pd.Series(model.decision_function(X[msk]), index=g[msk]).groupby(level=0).mean()


TEST_KEYS = ["frac_pos_test", "frac_neg_test", "frac_test", "frac_test_ic_inf", "frac_test_ic_sup",
             "abstencion_test", "abstencion_test_ic_inf", "abstencion_test_ic_sup",
             "riesgo_obs_pos", "riesgo_obs_pos_ic_inf", "riesgo_obs_pos_ic_sup",
             "riesgo_obs_neg", "riesgo_obs_neg_ic_inf", "riesgo_obs_neg_ic_sup"]


def test_metrics(t0, t1, r, pi, conf=0.95):
    """Estimaciones en prueba e IC SIMULTÁNEOS al nivel conf para R+, R-, fracción y abstención
    (8 cotas CP unilaterales a (1-conf)/8; ver docstring del módulo)."""
    out = {k: np.nan for k in TEST_KEYS}
    if len(t0) == 0 or len(t1) == 0:
        return out
    n0, n1 = len(t0), len(t1)
    d = (1 - conf) / 8.0
    tt = np.array([r["l_neg"], r["l_pos"]])
    k0 = np.searchsorted(np.sort(t0), tt, side="right")
    k1 = np.searchsorted(np.sort(t1), tt, side="right")
    hp, hn = bool(np.isfinite(r["l_pos"])), bool(np.isfinite(r["l_neg"]))
    xfp, xtp, xfn, xtn = n0 - k0[1], n1 - k1[1], k1[0], k0[0]
    fpr, tpr, fnr, tnr = xfp / n0, xtp / n1, xfn / n1, xtn / n0
    fpr_l, fpr_u = float(cp_lower(xfp, n0, d)), float(cp_upper(xfp, n0, d))
    tpr_l, tpr_u = float(cp_lower(xtp, n1, d)), float(cp_upper(xtp, n1, d))
    fnr_l, fnr_u = float(cp_lower(xfn, n1, d)), float(cp_upper(xfn, n1, d))
    tnr_l, tnr_u = float(cp_lower(xtn, n0, d)), float(cp_upper(xtn, n0, d))
    fp = pi * tpr + (1 - pi) * fpr if hp else 0.0
    fn = pi * fnr + (1 - pi) * tnr if hn else 0.0
    f_l = (pi * tpr_l + (1 - pi) * fpr_l if hp else 0.0) + (pi * fnr_l + (1 - pi) * tnr_l if hn else 0.0)
    f_u = (pi * tpr_u + (1 - pi) * fpr_u if hp else 0.0) + (pi * fnr_u + (1 - pi) * tnr_u if hn else 0.0)
    f_u = min(f_u, 1.0)
    out.update(frac_pos_test=fp, frac_neg_test=fn, frac_test=fp + fn,
               frac_test_ic_inf=f_l, frac_test_ic_sup=f_u,
               abstencion_test=1 - fp - fn, abstencion_test_ic_inf=1 - f_u,
               abstencion_test_ic_sup=1 - f_l)
    if hp and fp > 0:
        out.update(riesgo_obs_pos=float(risk_pos(pi, fpr, tpr)),
                   riesgo_obs_pos_ic_inf=float((1 - pi) * fpr_l / max((1 - pi) * fpr_l + pi * tpr_u, 1e-300)),
                   riesgo_obs_pos_ic_sup=float(risk_pos(pi, fpr_u, tpr_l)))
    if hn and fn > 0:
        out.update(riesgo_obs_neg=float(risk_neg(pi, fnr, tnr)),
                   riesgo_obs_neg_ic_inf=float(pi * fnr_l / max(pi * fnr_l + (1 - pi) * tnr_u, 1e-300)),
                   riesgo_obs_neg_ic_sup=float(risk_neg(pi, fnr_u, tnr_l)))
    return out


# ========================================================================================
# Control de atribución (diagnóstico, no garantía)
# ========================================================================================
def load_null(path, null_ranges, g_main):
    """Carga la misma cohorte restringida a regiones sin bandas de la muestra y verifica que
    las filas coincidan con las del análisis principal."""
    Xn, _, gn, _, nvn = load_cohort(path, null_ranges)
    if len(gn) != len(g_main) or not np.array_equal(gn, g_main):
        sys.exit("Las regiones sin muestra descartan espectros distintos que las regiones "
                 "principales; revisa NaN o espectros constantes.")
    return Xn, nvn


def null_certification(Xn, y, g, lab, p_fit, p_cal, rs, test_pats=None, Xn_te=None, g_te=None,
                       lab_te=None):
    """Ajusta el mismo clasificador SOLO con regiones sin bandas de la muestra, con las mismas
    particiones, y certifica con los mismos métodos y celdas. Devuelve {(método, alpha, pi):
    bool certificado} y la AUC del modelo nulo en prueba (si hay prueba)."""
    m = np.isin(g, p_fit)
    model, _, _ = fit_patient_level(Xn[m], y[m], g[m], lab, rs)
    sc = patient_scores(model, Xn, g, p_cal)
    yc = lab.loc[sc.index].to_numpy()
    s0, s1 = sc[yc == 0].to_numpy(), sc[yc == 1].to_numpy()
    out = {}
    for meth in METHODS:
        res = certify(s0, s1, meth, PIS, ALPHAS, rng=np.random.default_rng(rs + 3))
        for (al, pi), v in res.items():
            out[(meth, al, pi)] = bool(np.isfinite(v["l_pos"]) or np.isfinite(v["l_neg"]))
    auc = np.nan
    if test_pats is not None and len(test_pats):
        Xt = Xn if Xn_te is None else Xn_te
        gt = g if g_te is None else g_te
        lt = lab if lab_te is None else lab_te
        st = patient_scores(model, Xt, gt, test_pats)
        yt = lt.loc[st.index].to_numpy()
        if 0 < yt.sum() < len(yt):
            auc = roc_auc_score(yt, st.to_numpy())
    return out, auc


def attribution_label(cert_main, cert_null):
    if not cert_main:
        return "sin_certificacion"
    return "no_atribuible_a_bandas" if cert_null else "sin_alerta"


def report_attribution(df):
    if "atribucion" not in df:
        return
    c = df[df.atribucion != "sin_certificacion"]
    print("\n[ATRIBUCIÓN] Certificaciones que el modelo con regiones sin bandas de la muestra "
          "también obtiene (mismo método, alpha y pi):")
    if not len(c):
        print("  No hubo certificaciones en el análisis principal.")
        return
    t = c.assign(no_atr=c.atribucion == "no_atribuible_a_bandas").groupby("metodo").no_atr.agg(["sum", "size"])
    for meth, r in t.iterrows():
        print(f"  {meth}: {int(r['sum'])} de {int(r['size'])} certificaciones marcadas como "
              f"'no atribuible a bandas'")
    if "auc_test_nulo" in df and df.auc_test_nulo.notna().any():
        print(f"  AUC en prueba del modelo con regiones sin muestra: mediana "
              f"{df.drop_duplicates('rep').auc_test_nulo.median():.3f}")
    print("  Nota: es un diagnóstico. Una alerta indica que la información que permite certificar no"
          "\n  está confinada a las bandas bioquímicas; la ausencia de alerta no prueba atribución química.")


def run_data(path, reps, fr_train, fr_cal, wn_ranges, seed=0, null_ranges=None):
    X, y, g, lab, nvar = load_cohort(path, wn_ranges)
    Xn = load_null(path, null_ranges, g)[0] if null_ranges else None
    pats, plab = lab.index.to_numpy(), lab.to_numpy()
    nrep = pd.Series(g).value_counts()
    print(f"{path}: {len(y)} espectros, {len(pats)} pacientes ({(plab == 1).sum()} pos, "
          f"{(plab == 0).sum()} neg), {nvar} variables, réplicas por paciente "
          f"{nrep.min()}–{nrep.max()}")
    if fr_train <= 0 or fr_cal <= 0 or fr_train + fr_cal > 1.0 + 1e-9:
        sys.exit("--train y --cal deben ser positivos y sumar como máximo 1.")
    rng = np.random.default_rng(seed)
    rows, skipped = [], 0
    for r in range(reps):
        rs = int(rng.integers(1_000_000_000))
        p_tr, p_rest, _, y_rest = train_test_split(pats, plab, train_size=fr_train,
                                                   stratify=plab, random_state=rs)
        rel = fr_cal / (1 - fr_train)
        if rel >= 0.999:
            p_cal, p_te = p_rest, np.array([], dtype=p_rest.dtype)
        else:
            p_cal, p_te = train_test_split(p_rest, train_size=rel, stratify=y_rest,
                                           random_state=rs + 1)
        ycal_p = lab.loc[p_cal].to_numpy()
        if min((ycal_p == 0).sum(), (ycal_p == 1).sum()) < MIN_CAL_PER_CLASS:
            print(f"  repetición {r + 1}: menos de {MIN_CAL_PER_CLASS} pacientes por clase en "
                  f"calibración; no se puede certificar. Se omite.")
            skipped += 1
            continue
        mtr = np.isin(g, p_tr)
        try:
            model, C, auc_cv = fit_patient_level(X[mtr], y[mtr], g[mtr], lab, rs)
        except ValueError as e:
            print(f"  repetición {r + 1}: {e} Se omite.")
            skipped += 1
            continue
        sc_cal = patient_scores(model, X, g, p_cal)
        ycal = lab.loc[sc_cal.index].to_numpy()
        s0, s1 = sc_cal[ycal == 0].to_numpy(), sc_cal[ycal == 1].to_numpy()
        ties = (len(s0) - len(np.unique(s0))) + (len(s1) - len(np.unique(s1)))
        if len(p_te):
            sc_te = patient_scores(model, X, g, p_te)
            yte = lab.loc[sc_te.index].to_numpy()
            t0, t1 = sc_te[yte == 0].to_numpy(), sc_te[yte == 1].to_numpy()
            auc_te = roc_auc_score(yte, sc_te.to_numpy()) if 0 < yte.sum() < len(yte) else np.nan
        else:
            t0 = t1 = np.array([])
            auc_te = np.nan
        nulo, auc_nulo = (null_certification(Xn, y, g, lab, p_tr, p_cal, rs, p_te)
                          if Xn is not None else ({}, np.nan))
        for m in METHODS:
            res = certify(s0, s1, m, PIS, ALPHAS, rng=np.random.default_rng(rs + 3))
            for (al, pi), v in res.items():
                extra = {}
                if Xn is not None:
                    cm = bool(np.isfinite(v["l_pos"]) or np.isfinite(v["l_neg"]))
                    extra = dict(cert_nulo=nulo[(m, al, pi)], auc_test_nulo=auc_nulo,
                                 atribucion=attribution_label(cm, nulo[(m, al, pi)]))
                rows.append(dict(rep=r, metodo=m, alpha=al, pi=pi, C=C, auc_cv_paciente=auc_cv, **extra,
                                 auc_test=auc_te, n_cal_neg=len(s0), n_cal_pos=len(s1),
                                 n_test=len(t0) + len(t1), empates_calibracion=ties,
                                 cert_pos=bool(np.isfinite(v["l_pos"])),
                                 cert_neg=bool(np.isfinite(v["l_neg"])),
                                 frac_cal=v["frac"], frac_pos_cal=v["frac_pos"],
                                 frac_neg_cal=v["frac_neg"],
                                 riesgo_cert_pos=v["ur_pos_cert"], riesgo_cert_neg=v["ur_neg_cert"],
                                 riesgo_min_cert_pos=v["min_ur_pos"],
                                 riesgo_min_cert_neg=v["min_ur_neg"],
                                 **test_metrics(t0, t1, v, pi)))
        print(f"  repetición {r + 1}/{reps}: C={C:g}, AUC cv={auc_cv:.3f}, AUC test={auc_te:.3f}, "
              f"calibración {len(s0)} neg / {len(s1)} pos", flush=True)
    if not rows:
        sys.exit("Ninguna repetición válida.")
    df = pd.DataFrame(rows)
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    df.to_csv(f"viabilidad_{stem}_detalle.csv", index=False)
    q = lambda p: (lambda v: v.quantile(p))
    summ = df.groupby(["metodo", "alpha", "pi"]).agg(
        frac_cal_mediana=("frac_cal", "median"), frac_cal_q25=("frac_cal", q(0.25)),
        frac_cal_q75=("frac_cal", q(0.75)),
        prop_cert_pos=("cert_pos", "mean"), prop_cert_neg=("cert_neg", "mean"),
        riesgo_cert_pos=("riesgo_cert_pos", "median"), riesgo_cert_neg=("riesgo_cert_neg", "median"),
        riesgo_min_pos=("riesgo_min_cert_pos", "median"),
        riesgo_min_neg=("riesgo_min_cert_neg", "median"),
        frac_test=("frac_test", "median"), abstencion_test=("abstencion_test", "median"),
        riesgo_obs_pos=("riesgo_obs_pos", "median"),
        riesgo_obs_neg=("riesgo_obs_neg", "median")).reset_index()
    summ.to_csv(f"viabilidad_{stem}_resumen.csv", index=False)
    print(f"\nRepeticiones válidas: {df.rep.nunique()} de {reps} (omitidas: {skipped})")
    print("\nNota: las repeticiones comparten pacientes. Medianas y cuantiles entre repeticiones"
          "\ndescriben la variabilidad de la partición; NO son intervalos de confianza. Los IC"
          "\n(simultáneos al 95 % para R+, R-, fracción y abstención, por celda y repetición)"
          "\nestán en el archivo de detalle.")
    print("\n[CALIBRACIÓN] Fracción clasificada con riesgo certificado (mediana):")
    print(summ.pivot_table(index=["alpha", "pi"], columns="metodo",
                           values="frac_cal_mediana").round(3).to_string())
    if df.frac_test.notna().any():
        print("\n[PRUEBA] Fracción clasificada y abstención observadas (mediana):")
        print(summ.pivot_table(index=["alpha", "pi"], columns="metodo",
                               values=["frac_test", "abstencion_test"]).round(3).to_string())
    print("\n[CALIBRACIÓN] Riesgo mínimo certificable con bandas (mediana):")
    print(summ[summ.metodo != "split"].pivot_table(
        index="pi", columns="metodo", values=["riesgo_min_pos", "riesgo_min_neg"],
        aggfunc="first").round(3).to_string())
    report_attribution(df)
    print(f"\nGuardado: viabilidad_{stem}_resumen.csv y viabilidad_{stem}_detalle.csv")


def run_external(train_csv, test_csv, reps, fr_cal, wn_ranges, seed=0, null_ranges=None):
    """Prueba externa FIJA: ajuste y calibración salen solo de train_csv (particiones repetidas
    por paciente); test_csv se evalúa siempre completo con los umbrales de cada repetición.
    La repetición 0 (semilla fija) es el ANÁLISIS PRIMARIO preespecificado; el resto describe
    la sensibilidad a la partición (no son estudios independientes)."""
    Xtr, ytr, gtr, labtr, nvar = load_cohort(train_csv, wn_ranges)
    Xte, yte_s, gte, labte, nvar2 = load_cohort(test_csv, wn_ranges)
    if nvar != nvar2:
        sys.exit("Entrenamiento y prueba tienen distintas variables espectrales.")
    if null_ranges:
        Xtr_n = load_null(train_csv, null_ranges, gtr)[0]
        Xte_n = load_null(test_csv, null_ranges, gte)[0]
    else:
        Xtr_n = Xte_n = None
    shared = set(labtr.index) & set(labte.index)
    if shared:
        sys.exit(f"Pacientes en entrenamiento y prueba: {sorted(shared)[:10]}")
    pats, plab = labtr.index.to_numpy(), labtr.to_numpy()
    print(f"Entrenamiento: {len(ytr)} espectros, {len(pats)} pacientes ({plab.sum()} pos, {(plab == 0).sum()} neg)")
    print(f"Prueba externa: {len(yte_s)} espectros, {len(labte)} pacientes ({labte.sum()} pos, "
          f"{(labte == 0).sum()} neg)")
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(reps):
        rs = seed if r == 0 else int(rng.integers(1_000_000_000))
        p_fit, p_cal = train_test_split(pats, test_size=fr_cal, stratify=plab, random_state=rs)
        ycal_p = labtr.loc[p_cal].to_numpy()
        if min((ycal_p == 0).sum(), (ycal_p == 1).sum()) < MIN_CAL_PER_CLASS:
            print(f"  repetición {r}: calibración insuficiente; se omite.")
            continue
        m = np.isin(gtr, p_fit)
        model, C, auc_cv = fit_patient_level(Xtr[m], ytr[m], gtr[m], labtr, rs)
        sc = patient_scores(model, Xtr, gtr, p_cal)
        yc = labtr.loc[sc.index].to_numpy()
        s0, s1 = sc[yc == 0].to_numpy(), sc[yc == 1].to_numpy()
        st = patient_scores(model, Xte, gte, labte.index.to_numpy())
        yt = labte.loc[st.index].to_numpy()
        t0, t1 = st[yt == 0].to_numpy(), st[yt == 1].to_numpy()
        auc_te = roc_auc_score(yt, st.to_numpy())
        auc_cal = roc_auc_score(yc, sc.to_numpy())
        nulo, auc_nulo = (null_certification(Xtr_n, ytr, gtr, labtr, p_fit, p_cal, rs,
                                             labte.index.to_numpy(), Xte_n, gte, labte)
                          if Xtr_n is not None else ({}, np.nan))
        for meth in METHODS:
            res = certify(s0, s1, meth, PIS, ALPHAS, rng=np.random.default_rng(rs + 3))
            for (al, pi), v in res.items():
                extra = {}
                if Xtr_n is not None:
                    cm = bool(np.isfinite(v["l_pos"]) or np.isfinite(v["l_neg"]))
                    extra = dict(cert_nulo=nulo[(meth, al, pi)], auc_test_nulo=auc_nulo,
                                 atribucion=attribution_label(cm, nulo[(meth, al, pi)]))
                rows.append(dict(rep=r, primario=(r == 0), metodo=meth, alpha=al, pi=pi, C=C, **extra,
                                 auc_cv_paciente=auc_cv, auc_calibracion=auc_cal, auc_test=auc_te,
                                 n_cal_neg=len(s0), n_cal_pos=len(s1),
                                 cert_pos=bool(np.isfinite(v["l_pos"])),
                                 cert_neg=bool(np.isfinite(v["l_neg"])),
                                 frac_cal=v["frac"], riesgo_cert_pos=v["ur_pos_cert"],
                                 riesgo_cert_neg=v["ur_neg_cert"],
                                 riesgo_min_cert_pos=v["min_ur_pos"],
                                 riesgo_min_cert_neg=v["min_ur_neg"],
                                 **test_metrics(t0, t1, v, pi)))
        print(f"  repetición {r}{' (primaria)' if r == 0 else ''}: C={C:g}, AUC cv={auc_cv:.3f}, "
              f"AUC calibración={auc_cal:.3f}, AUC prueba externa={auc_te:.3f}, "
              f"calibración {len(s0)} neg / {len(s1)} pos", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("externa_detalle.csv", index=False)
    report_attribution(df)
    return df


def parse_ranges(s):
    if not s:
        return None
    return [tuple(float(x) for x in part.split("-")) for part in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    a1 = sub.add_parser("validate")
    a1.add_argument("--reps", type=int, default=2000)
    a1.add_argument("--fast", action="store_true", help="omite n=250")
    a1.add_argument("--scenario", action="append", default=None,
                    help="correr solo este escenario (repetible); combinar luego con 'report'")
    sub.add_parser("report", help="combina los archivos de validate --scenario")
    a4 = sub.add_parser("external", help="prueba externa fija")
    a4.add_argument("train_csv")
    a4.add_argument("test_csv")
    a4.add_argument("--reps", type=int, default=50)
    a4.add_argument("--cal", type=float, default=0.6)
    a4.add_argument("--wn-ranges", type=str, default="900-1800,2800-3700")
    a4.add_argument("--null-ranges", type=str, default="",
                    help="regiones sin bandas de la muestra para el control de atribución, "
                         "p. ej. 2400-2700,3800-4000")
    a3 = sub.add_parser("coverage")
    a3.add_argument("--reps", type=int, default=20000)
    a2 = sub.add_parser("data")
    a2.add_argument("csv")
    a2.add_argument("--reps", type=int, default=50)
    a2.add_argument("--train", type=float, default=0.5)
    a2.add_argument("--cal", type=float, default=0.3)
    a2.add_argument("--wn-ranges", type=str, default="900-1800,2800-3700")
    a2.add_argument("--null-ranges", type=str, default="",
                    help="regiones sin bandas de la muestra para el control de atribución, "
                         "p. ej. 2400-2700,3800-4000")
    args = ap.parse_args()
    if args.mode == "validate":
        validate(args.reps, fast=args.fast, only=args.scenario)
    elif args.mode == "report":
        report_from_files()
    elif args.mode == "external":
        run_external(args.train_csv, args.test_csv, args.reps, args.cal,
                     parse_ranges(args.wn_ranges), null_ranges=parse_ranges(args.null_ranges))
    elif args.mode == "coverage":
        band_coverage(args.reps)
    else:
        run_data(args.csv, args.reps, args.train, args.cal, parse_ranges(args.wn_ranges),
                 null_ranges=parse_ranges(args.null_ranges))


if __name__ == "__main__":
    main()
