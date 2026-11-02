"""
run_control_positivo.py — control positivo semisintético para el control de atribución.

Diseño
  1. Se toman SOLO los 500 controles de diabetes.csv (un espectro por paciente) y se
     asignan al azar a dos pseudo-clases de 250. Por construcción no hay ninguna diferencia
     de origen entre grupos (ni física, ni de adquisición, ni clínica).
  2. A cada espectro se le suma un conjunto de bandas tipo glucosa (C-O, 1000-1200 cm-1)
     con amplitud aleatoria por paciente: N(mu0, sigma) en el grupo 0 y N(mu0 + d*sigma,
     sigma) en el grupo 1 (truncada en 0). La única diferencia entre grupos es química y
     está localizada en 1000-1200 cm-1.
  3. Para cada tamaño de efecto d se corre la certificación con:
       - regiones bioquímicas (900-1800, 2800-3700): debe certificar cuando d es suficiente;
       - regiones sin bandas de la muestra (2400-2700, 3800-4000): NO debe certificar.
     Así se demuestra que el control de atribución distingue señal química (aquí) de señal
     no química (base real de diabetes).

Requiere el paquete certispec instalado y diabetes.csv en WORKDIR
Uso en Colab (después de terminar run_pendientes.py):
  Celda 1:  from google.colab import drive; drive.mount('/content/drive')
  Celda 2:  %cd /content/drive/MyDrive/certificacion_ftir
            %run run_control_positivo.py
Resultados en WORKDIR/resultados_control_positivo/ (RESUMEN_CONTROL_POSITIVO.txt).
Si Colab se desconecta, vuelve a ejecutar: lo terminado se omite.
"""

import contextlib
import io
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------------------------
WORKDIR = os.getcwd()
OUT = os.path.join(WORKDIR, "resultados_control_positivo")
REPS = 20                                  # particiones por condición
TAMANOS_EFECTO = [0.75, 1.25, 1.75, 2.5, 3.5]   # d = diferencia de medias / desviación estándar
MU0 = 0.010                                # amplitud media de la banda en el grupo 0 (absorbancia)
SIGMA = 0.004                              # desviación estándar de la amplitud entre pacientes
SEMILLA = 2026
REGIONES = {
    "bioquimicas": [(900, 1800), (2800, 3700)],
    "sin_muestra": [(2400, 2700), (3800, 4000)],
}
# Bandas tipo glucosa (centro cm-1, altura relativa, FWHM cm-1)
BANDAS = [(1033, 1.00, 22), (1080, 0.60, 20), (1108, 0.50, 20), (1150, 0.35, 18)]

from certispec import core as cf  # noqa: E402

os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "RESUMEN_CONTROL_POSITIVO.txt")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


@contextlib.contextmanager
def chdir(d):
    old = os.getcwd()
    os.chdir(d)
    try:
        yield
    finally:
        os.chdir(old)


def perfil_bandas(wn):
    p = np.zeros_like(wn, dtype=float)
    for c, h, fwhm in BANDAS:
        s = fwhm / 2.3548
        p += h * np.exp(-0.5 * ((wn - c) / s) ** 2)
    return p / p.max()


def construir_datos(d_efecto):
    f = os.path.join(OUT, "datos", f"control_positivo_d{d_efecto}.csv")
    if os.path.exists(f):
        return f
    os.makedirs(os.path.dirname(f), exist_ok=True)
    src = os.path.join(WORKDIR, "diabetes.csv")
    if not os.path.exists(src):
        raise FileNotFoundError(f"Falta {src}")
    x = pd.read_csv(src)
    x = x[x["label"] == "Control"].reset_index(drop=True)
    wn = np.array([float(c) for c in x.columns[1:]])
    X = x.iloc[:, 1:].to_numpy(float)
    rng = np.random.default_rng(SEMILLA)          # mismos números aleatorios para todo d
    n = len(x)
    grupo = np.zeros(n, int)
    grupo[rng.permutation(n)[: n // 2]] = 1
    z = rng.standard_normal(n)
    amp = np.maximum(MU0 + SIGMA * z + grupo * d_efecto * SIGMA, 0.0)
    Xs = X + amp[:, None] * perfil_bandas(wn)[None, :]
    out = pd.concat([pd.Series([f"C{i}" for i in range(n)], name="patient_id"),
                     pd.Series(grupo, name="label"),
                     pd.DataFrame(Xs, columns=x.columns[1:])], axis=1)
    out.to_csv(f, index=False)
    return f


def correr(d_efecto, nombre, regiones):
    d = os.path.join(OUT, f"d{d_efecto}_{nombre}")
    os.makedirs(d, exist_ok=True)
    stem = f"control_positivo_d{d_efecto}"
    det = os.path.join(d, f"viabilidad_{stem}_detalle.csv")
    if not os.path.exists(det):
        f = construir_datos(d_efecto)
        t0 = time.time()
        buf = io.StringIO()
        with chdir(d), contextlib.redirect_stdout(buf):
            cf.run_data(f, REPS, 0.5, 0.3, regiones, SEMILLA)
        open(os.path.join(d, "salida.txt"), "w").write(buf.getvalue())
        log(f"  d={d_efecto} {nombre}: listo en {time.time() - t0:.0f} s")
    return pd.read_csv(det)


def resumir(det, d_efecto, nombre):
    det = det.assign(cert=det.cert_pos | det.cert_neg)
    filas = []
    for al, pi in [(0.05, 0.10), (0.10, 0.20), (0.10, 0.30), (0.15, 0.50)]:
        x = det[(det.metodo == "os") & (det.alpha == al) & (det.pi == pi)]
        filas.append(dict(d=d_efecto, regiones=nombre, alpha=al, pi=pi,
                          auc_test=det.drop_duplicates("rep").auc_test.median(),
                          prop_cert=x.cert.mean(), frac_cal=x.frac_cal.median(),
                          frac_test=x.frac_test.median(),
                          riesgo_min_neg=x.riesgo_min_cert_neg.median(),
                          riesgo_min_pos=x.riesgo_min_cert_pos.median()))
    c = det[det.cert]
    viol = int(((c.cert_pos & (c.riesgo_obs_pos > c.alpha)) |
                (c.cert_neg & (c.riesgo_obs_neg > c.alpha))).sum())
    return filas, viol, len(c)


def main():
    open(LOG, "a").write(f"\n\n######## Corrida {time.strftime('%Y-%m-%d %H:%M')} ########\n")
    log("Control positivo semisintético: 500 controles reales asignados al azar a dos grupos;")
    log(f"bandas tipo glucosa (1000-1200 cm-1), amplitud N({MU0}, {SIGMA}) + d·{SIGMA} en el grupo 1.")
    filas, viols = [], []
    for d_efecto in TAMANOS_EFECTO:
        for nombre, reg in REGIONES.items():
            det = correr(d_efecto, nombre, reg)
            f, v, nc = resumir(det, d_efecto, nombre)
            filas += f
            viols.append(dict(d=d_efecto, regiones=nombre, certificaciones=nc,
                              riesgo_obs_mayor_alpha=v))
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(OUT, "control_positivo_resumen.csv"), index=False)
    log("\n=== Banda os: proporción de particiones con certificación y fracción clasificada ===")
    for (al, pi), g in t.groupby(["alpha", "pi"]):
        log(f"\nalpha={al}, pi={pi}")
        log(g.pivot_table(index="d", columns="regiones",
                          values=["auc_test", "prop_cert", "frac_cal"]).round(3).to_string())
    log("\n=== Riesgo mínimo certificable de descarte (os), pi=0.10 ===")
    g = t[(t.alpha == 0.05) & (t.pi == 0.10)]
    log(g.pivot_table(index="d", columns="regiones", values="riesgo_min_neg").round(3).to_string())
    log("\n=== Certificaciones con riesgo observado en prueba > alpha (estimación puntual) ===")
    log(pd.DataFrame(viols).to_string(index=False))
    log("\nLectura esperada: con regiones bioquímicas, la certificación aparece al crecer d;"
        "\ncon regiones sin muestra, AUC ~0.5 y ninguna certificación. Ese contraste, frente a la"
        "\nbase real de diabetes (donde ambas certifican), es la validación del control de atribución.")
    z = os.path.join(WORKDIR, "resultados_control_positivo.zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUT):
            for fn in files:
                p = os.path.join(root, fn)
                if os.sep + "datos" + os.sep not in p:
                    zf.write(p, os.path.relpath(p, WORKDIR))
    log(f"\nListo. Resumen: {LOG}\nArchivo para descargar: {z}")


if __name__ == "__main__":
    main()
