"""
run_pendientes.py — experimentos pendientes del artículo de certificación de VPP/VPN.

Requiere en WORKDIR:
  (paquete certispec instalado: pip install certispec o pip install -e .)
  diabetes.csv                           (label DM2/Control + espectro)
  plasma_train.csv, plasma_test_independent.csv   (salida de prepare_plasma.py)
  General_and_truncated_spectra.xlsx, Localidade_truncado.xlsx   (esquistosomiasis)

Uso en Colab:
  Celda 1:  from google.colab import drive; drive.mount('/content/drive')
  Celda 2:  %cd /content/drive/MyDrive/certificacion_ftir
            %run run_pendientes.py
Si Colab se desconecta, vuelve a ejecutar las dos celdas: las tareas terminadas se omiten.
Resultados en WORKDIR/resultados/ y resumen en resultados/RESUMEN.txt.
"""

import contextlib
import glob
import io
import multiprocessing as mp
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------------------------
WORKDIR = os.getcwd()
OUT = os.path.join(WORKDIR, "resultados")

TAREAS = {
    "regiones_nulas_diabetes": True,   # 1. control de regiones sin muestra
    "cohortes_finales": True,          # 4. análisis finales de las tres cohortes
    "cobertura_bandas": True,          # 3. cobertura de las bandas (20000 rep.)
    "validacion_simulacion": True,     # 2. validación por simulación (2000 rep.)
}
REPS_COHORTES = 50          # particiones por cohorte (análisis final preespecificado)
REPS_COBERTURA = 20000
REPS_VALIDACION = 2000
SEMILLA = 0

# Regiones (cm-1). Regiones sin bandas de la muestra: se evitan el CO2 (~2300-2400)
# y la región del diamante (~1900-2300). Ajusta si tu ATR no es de diamante.
REGIONES_DIABETES = {
    "bioquimicas": [(900, 1800), (2800, 3700)],
    "sin_muestra": [(2400, 2700), (3800, 4000)],
    "espectro_completo": [(400, 4000)],
}
REGIONES_COVID = [(900, 1800), (2800, 3700)]
REGIONES_SCH = [(900, 1800), (2800, 3050)]

from certispec import core as cf  # noqa: E402

os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "RESUMEN.txt")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def en_carpeta(sub):
    d = os.path.join(OUT, sub)
    os.makedirs(d, exist_ok=True)
    return d


@contextlib.contextmanager
def chdir(d):
    old = os.getcwd()
    os.chdir(d)
    try:
        yield
    finally:
        os.chdir(old)


def capturar(func, *args, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = func(*args, **kw)
    return r, buf.getvalue()


def requiere(*files):
    falta = [f for f in files if not os.path.exists(os.path.join(WORKDIR, f))]
    if falta:
        raise FileNotFoundError(f"Faltan en {WORKDIR}: {falta}")


# ------------------------------------------------------------------------------------
# CONVERSIÓN DE DATOS AL FORMATO DEL SOFTWARE (patient_id, label 0/1, espectro)
# ------------------------------------------------------------------------------------
def preparar_datos():
    d = os.path.join(OUT, "datos")
    os.makedirs(d, exist_ok=True)

    f = os.path.join(d, "diabetes_cf.csv")
    if not os.path.exists(f):
        requiere("diabetes.csv")
        x = pd.read_csv(os.path.join(WORKDIR, "diabetes.csv"))
        out = pd.concat([pd.Series([f"S{i}" for i in range(len(x))], name="patient_id"),
                         (x["label"] == "DM2").astype(int).rename("label"), x.iloc[:, 1:]], axis=1)
        out.to_csv(f, index=False)

    for src, dst in [("plasma_train.csv", "covid_train.csv"),
                     ("plasma_test_independent.csv", "covid_test.csv")]:
        f = os.path.join(d, dst)
        if not os.path.exists(f):
            requiere(src)
            x = pd.read_csv(os.path.join(WORKDIR, src), dtype={"subject": str})
            wn = [c for c in x.columns if _es_numero(c)]
            out = pd.concat([x["subject"].rename("patient_id"),
                             (x["label"] == "Severe").astype(int).rename("label"), x[wn]], axis=1)
            out.to_csv(f, index=False)

    f1, f2, f3 = (os.path.join(d, n) for n in ("sch.csv", "sch_januaria.csv", "sch_otros_sitios.csv"))
    if not all(os.path.exists(p) for p in (f1, f2, f3)):
        requiere("General_and_truncated_spectra.xlsx", "Localidade_truncado.xlsx")
        g = pd.read_excel(os.path.join(WORKDIR, "General_and_truncated_spectra.xlsx"),
                          sheet_name="General spectra")
        wn = [c for c in g.columns if _es_numero(c)]
        out = pd.concat([pd.Series([f"P{i}" for i in range(len(g))], name="patient_id"),
                         (g["Class"] == "SCH+").astype(int).rename("label"), g[wn]], axis=1)
        out.columns = [str(c) for c in out.columns]
        out.to_csv(f1, index=False)
        l = pd.read_excel(os.path.join(WORKDIR, "Localidade_truncado.xlsx"))
        wnl = [c for c in l.columns if _es_numero(c)]
        for name, m, pref in [(f2, l["Site"] == "Januária", "J"), (f3, l["Site"] != "Januária", "O")]:
            x = l[m].reset_index(drop=True)
            o = pd.concat([pd.Series([f"{pref}{i}" for i in range(len(x))], name="patient_id"),
                           (x["Class"] == "SCH+").astype(int).rename("label"), x[wnl]], axis=1)
            o.columns = [str(c) for c in o.columns]
            o.to_csv(name, index=False)
    return d


def _es_numero(c):
    try:
        float(c)
        return True
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------------------------
# RESÚMENES
# ------------------------------------------------------------------------------------
def resumen_cohorte(det, titulo, alphas=(0.05, 0.10), pis=(0.10, 0.20, 0.30, 0.50)):
    log(f"\n--- {titulo} ---")
    reps = det.drop_duplicates("rep")
    for col in ("auc_cv_paciente", "auc_calibracion", "auc_test"):
        if col in reps and reps[col].notna().any():
            log(f"  {col}: mediana {reps[col].median():.3f} "
                f"(mín {reps[col].min():.3f}, máx {reps[col].max():.3f})")
    det = det.assign(cert=det.cert_pos | det.cert_neg)
    t = det[det.alpha.isin(alphas) & det.pi.isin(pis)].pivot_table(
        index=["alpha", "pi"], columns="metodo", values="frac_cal", aggfunc="median")
    log("  Fracción clasificada con riesgo certificado (calibración, mediana):")
    log(t.round(3).to_string())
    t = det[det.alpha.isin(alphas) & det.pi.isin(pis)].pivot_table(
        index=["alpha", "pi"], columns="metodo", values="cert", aggfunc="mean")
    log("  Proporción de particiones con alguna certificación:")
    log(t.round(2).to_string())
    os_ = det[(det.metodo == "os") & (det.alpha == 0.05)].groupby("pi")[
        ["riesgo_min_cert_pos", "riesgo_min_cert_neg"]].median()
    log("  Riesgo mínimo certificable (os, mediana):")
    log(os_.round(3).to_string())
    if "riesgo_obs_pos" in det and det.riesgo_obs_pos.notna().any() or \
            "riesgo_obs_neg" in det and det.riesgo_obs_neg.notna().any():
        c = det[det.cert]
        vp = c.cert_pos & (c.riesgo_obs_pos > c.alpha)
        vn = c.cert_neg & (c.riesgo_obs_neg > c.alpha)
        log(f"  Certificaciones con riesgo observado en prueba > alpha: {int((vp | vn).sum())} "
            f"de {len(c)} (estimación puntual; ver IC en el detalle)")
    if "primario" in det:
        p = det[det.primario & det.cert]
        log(f"  Análisis primario (repetición 0): {len(p)} celdas-método certificadas")
        if len(p):
            log(p[["metodo", "alpha", "pi", "frac_cal", "frac_test", "riesgo_cert_pos",
                   "riesgo_cert_neg", "riesgo_obs_pos", "riesgo_obs_neg"]].round(3).to_string(index=False))


# ------------------------------------------------------------------------------------
# TAREA 1: CONTROL DE REGIONES SIN MUESTRA (DIABETES)
# ------------------------------------------------------------------------------------
def tarea_regiones_nulas(datos):
    log("\n================ 1. CONTROL DE REGIONES SIN MUESTRA (DIABETES) ================")
    fila = []
    for nombre, regiones in REGIONES_DIABETES.items():
        d = en_carpeta(f"diabetes_{nombre}")
        det_f = os.path.join(d, "viabilidad_diabetes_cf_detalle.csv")
        if not os.path.exists(det_f):
            t0 = time.time()
            with chdir(d):
                _, txt = capturar(cf.run_data, os.path.join(datos, "diabetes_cf.csv"),
                                  REPS_COHORTES, 0.5, 0.3, regiones, SEMILLA)
            open(os.path.join(d, "salida.txt"), "w").write(txt)
            log(f"  {nombre}: listo en {time.time() - t0:.0f} s")
        det = pd.read_csv(det_f)
        resumen_cohorte(det, f"Diabetes — regiones {nombre} {regiones}")
        x = det[(det.metodo == "os") & (det.alpha == 0.05) & (det.pi == 0.10)]
        fila.append(dict(regiones=nombre, auc_test_mediana=det.drop_duplicates("rep").auc_test.median(),
                         prop_cert=float((x.cert_pos | x.cert_neg).mean()),
                         frac_cal_mediana=x.frac_cal.median(),
                         riesgo_min_neg=x.riesgo_min_cert_neg.median()))
    comp = pd.DataFrame(fila)
    comp.to_csv(os.path.join(OUT, "diabetes_comparacion_regiones.csv"), index=False)
    log("\n  Comparación (os, alpha=0.05, pi=0.10):")
    log(comp.round(3).to_string(index=False))
    log("  Interpretación: si 'sin_muestra' también certifica, la certificación no puede atribuirse"
        "\n  solo a la química de la muestra (señal de adquisición).")


# ------------------------------------------------------------------------------------
# TAREA 4: ANÁLISIS FINALES DE LAS TRES COHORTES
# ------------------------------------------------------------------------------------
def tarea_cohortes(datos):
    log("\n================ 4. ANÁLISIS FINALES DE LAS COHORTES ================")
    d = en_carpeta("diabetes_bioquimicas")
    if os.path.exists(os.path.join(d, "viabilidad_diabetes_cf_detalle.csv")):
        log("  Diabetes (regiones bioquímicas): ya incluido en la tarea 1.")
    else:
        with chdir(d):
            _, txt = capturar(cf.run_data, os.path.join(datos, "diabetes_cf.csv"),
                              REPS_COHORTES, 0.5, 0.3, REGIONES_DIABETES["bioquimicas"], SEMILLA)
        resumen_cohorte(pd.read_csv(os.path.join(d, "viabilidad_diabetes_cf_detalle.csv")),
                        "Diabetes — regiones bioquímicas")

    d = en_carpeta("covid_externa")
    f = os.path.join(d, "externa_detalle.csv")
    if not os.path.exists(f):
        with chdir(d):
            _, txt = capturar(cf.run_external, os.path.join(datos, "covid_train.csv"),
                              os.path.join(datos, "covid_test.csv"), REPS_COHORTES, 0.6,
                              REGIONES_COVID, SEMILLA)
        open(os.path.join(d, "salida.txt"), "w").write(txt)
    resumen_cohorte(pd.read_csv(f), "Plasma COVID-19 — prueba ciega fija (22 pacientes)")

    d = en_carpeta("esquistosomiasis_completa")
    f = os.path.join(d, "viabilidad_sch_detalle.csv")
    if not os.path.exists(f):
        with chdir(d):
            _, txt = capturar(cf.run_data, os.path.join(datos, "sch.csv"), REPS_COHORTES,
                              0.4, 0.6, REGIONES_SCH, SEMILLA)
        open(os.path.join(d, "salida.txt"), "w").write(txt)
    resumen_cohorte(pd.read_csv(f), "Esquistosomiasis — base completa")

    d = en_carpeta("esquistosomiasis_entre_sitios")
    f = os.path.join(d, "externa_detalle.csv")
    if not os.path.exists(f):
        with chdir(d):
            _, txt = capturar(cf.run_external, os.path.join(datos, "sch_januaria.csv"),
                              os.path.join(datos, "sch_otros_sitios.csv"), REPS_COHORTES, 0.5,
                              REGIONES_SCH, SEMILLA)
        open(os.path.join(d, "salida.txt"), "w").write(txt)
    resumen_cohorte(pd.read_csv(f), "Esquistosomiasis — calibración en Januária, prueba en otros sitios")


# ------------------------------------------------------------------------------------
# TAREA 3: COBERTURA DE LAS BANDAS
# ------------------------------------------------------------------------------------
def tarea_cobertura():
    log(f"\n================ 3. COBERTURA DE LAS BANDAS ({REPS_COBERTURA} rep.) ================")
    d = en_carpeta("cobertura")
    f = os.path.join(d, "cobertura_bandas.csv")
    if not os.path.exists(f):
        t0 = time.time()
        with chdir(d):
            capturar(cf.band_coverage, REPS_COBERTURA)
        log(f"  listo en {time.time() - t0:.0f} s")
    c = pd.read_csv(f)
    eta = c.nominal.iloc[0]
    log(c.pivot_table(index=["distribucion", "n"], columns="banda",
                      values="no_cobertura").round(4).to_string())
    cont, disc = c[~c.discreta], c[c.discreta]
    log(f"  os continua: IC95 contiene {eta} en "
        f"{int(((cont.banda == 'os') & (cont.ic95_inf <= eta) & (cont.ic95_sup >= eta)).sum())} de "
        f"{int((cont.banda == 'os').sum())} casos (banda exacta)")
    log(f"  dkw continua: IC95 inferior <= {eta} en "
        f"{int(((cont.banda == 'dkw') & (cont.ic95_inf <= eta)).sum())} de {int((cont.banda == 'dkw').sum())}")
    log(f"  discretas: IC95 superior <= {eta} en {int((disc.ic95_sup <= eta).sum())} de {len(disc)}; "
        f"máximo observado {disc.no_cobertura.max():.4f}")


# ------------------------------------------------------------------------------------
# TAREA 2: VALIDACIÓN POR SIMULACIÓN (paralela por escenario, con puntos de control)
# ------------------------------------------------------------------------------------
def _un_escenario(args):
    nombre, reps, carpeta = args
    with chdir(carpeta):
        if os.path.exists(f"validacion_por_celda_{nombre}.csv"):
            return nombre, 0.0
        t0 = time.time()
        capturar(cf.validate, reps, False, 0, [nombre])
    return nombre, time.time() - t0


def tarea_validacion():
    log(f"\n================ 2. VALIDACIÓN POR SIMULACIÓN ({REPS_VALIDACION} rep.) ================")
    d = en_carpeta("validacion")
    nombres = list(cf.scenarios().keys())
    pend = [n for n in nombres if not os.path.exists(os.path.join(d, f"validacion_por_celda_{n}.csv"))]
    if pend:
        nproc = max(1, min(len(pend), os.cpu_count() or 1))
        log(f"  {len(pend)} escenarios pendientes, {nproc} procesos")
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=nproc, mp_context=ctx) as ex:
            for nombre, seg in ex.map(_un_escenario, [(n, REPS_VALIDACION, d) for n in pend]):
                log(f"  {nombre}: listo en {seg / 60:.1f} min")
    with chdir(d):
        _, txt = capturar(cf.report_from_files)
    open(os.path.join(d, "reporte_validacion.txt"), "w").write(txt)
    log(txt)


# ------------------------------------------------------------------------------------
def main():
    open(LOG, "a").write(f"\n\n######## Corrida {time.strftime('%Y-%m-%d %H:%M')} ########\n")
    datos = preparar_datos()
    if TAREAS["regiones_nulas_diabetes"]:
        tarea_regiones_nulas(datos)
    if TAREAS["cohortes_finales"]:
        tarea_cohortes(datos)
    if TAREAS["cobertura_bandas"]:
        tarea_cobertura()
    if TAREAS["validacion_simulacion"]:
        tarea_validacion()
    z = os.path.join(WORKDIR, "resultados.zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in glob.glob(os.path.join(OUT, "**", "*"), recursive=True):
            if os.path.isfile(p) and "/datos/" not in p:
                zf.write(p, os.path.relpath(p, WORKDIR))
    log(f"\nTodo listo. Resumen: {LOG}\nArchivo para descargar: {z}")


if __name__ == "__main__":
    main()
