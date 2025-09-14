import pandas as pd
import numpy as np
import pymc as pm
from arviz import summary, plot_trace
import statsmodels.api as sm
from collections import Counter, defaultdict
from sklearn import linear_model, ensemble, svm, model_selection, metrics
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold, cross_val_score
from typing import List, Dict, Tuple, Optional
import pdb
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime
import math


def last_interval_expand(L30, L15, L6, Lsig, uu, min_size=40):
    edges = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    # 1) Índice de bin (con recorte para incluir 6.0)
    bin_idx = np.digitize(uu, edges, right=False)
    bin_idx = max(0, min(bin_idx, len(edges)-1))

    # 2) Crear máscara para el bin principal
    arr15 = np.array(L15, dtype=float)
    if bin_idx == 0:
        left, right = -np.inf, edges[0]
    else:
        left, right = edges[bin_idx-1], edges[bin_idx]
    mask = (arr15 >= left) & (arr15 < right)

    # 3) Si es primer o último bin y faltan datos, sumar vecino
    if mask.sum() < min_size:
        if bin_idx == 0:
            # sumar el segundo tramo (3.0 ≤ x < 3.5)
            m2 = (arr15 >= edges[0]) & (arr15 < edges[1])
            mask = mask | m2
        elif bin_idx == len(edges)-1:
            # sumar el penúltimo tramo (5.5 ≤ x < 6.0)
            m2 = (arr15 >= edges[-2]) & (arr15 < edges[-1])
            mask = mask | m2

    # 4) Filtrar todas las listas
    L30a  = [v for v, m in zip(L30,  mask) if m]
    L15a  = [v for v, m in zip(L15,  mask) if m]
    L6a   = [v for v, m in zip(L6,   mask) if m]
    Lsiga = [v for v, m in zip(Lsig, mask) if m]

    return L30a, L15a, L6a, Lsiga,  bin_idx 


def last_interval(L30: List[float], L15: List[float], L6: List[float], Lsig: List[float], uu: float, n_bins: int = 9) -> Tuple[List[float], List[float], List[float], List[float]]:
    # 1) Redondear min/max de L15 a múltiplos de .5
    arr15 = np.array(L15, dtype=float)
    min0 = math.floor(arr15.min()*2) / 2
    max0 = math.ceil (arr15.max()*2) / 2
    # 2) Media y diferencia
    mean = (min0 + max0) / 2
    d    = max0 - min0
    # 3) Bordes de los intervalos
    #    Creamos n_bins+1 bordes desde mean-0.5d hasta mean+0.5d
    edges = np.linspace(mean - 0.5*d, mean + 0.5*d, num=n_bins+1)
    # 4) Encontrar intervalo del último valor
    last_val = uu
    # digitize devuelve índice de borde derecho, con shift -1 conseguimos 0-based
    bin_idx  = np.digitize(last_val, edges, right=False) - 1
    # Cap en [0, n_bins-1]
    bin_idx = max(0, min(n_bins-1, bin_idx))
    # 5) Máscara booleana: mismo intervalo
    left_edge  = edges[bin_idx]
    right_edge = edges[bin_idx + 1]
    # Incluimos el límite izquierdo y excluimos el derecho salvo que sea el último bin
    if bin_idx == n_bins-1:
        mask = (arr15 >= left_edge) & (arr15 <= right_edge)
    else:
        mask = (arr15 >= left_edge) & (arr15 <  right_edge)

    # 6) Construir las listas filtradas
    L30a  = [v for v, m in zip(L30,  mask) if m]
    L15a  = [v for v, m in zip(L15,  mask) if m]
    L6a   = [v for v, m in zip(L6,   mask) if m]
    Lsiga = [v for v, m in zip(Lsig, mask) if m]
        
    return L30a, L15a, L6a, Lsiga


def recency_pos(series: pd.Series) -> list[int]:
    window_size=50
    n = len(series)
    positions = []
    # Convertimos a array para acelerar comparaciones
    arr = series.to_numpy()
    digits = np.arange(10)

    for start in range(n - window_size):
        end = start + window_size  # índice del primer elemento fuera de la ventana
        window = arr[start:end]
        # 1) Calculamos recencia para cada dígito
        recency = {}
        for x in digits:
            # buscamos todas las posiciones de `x` en la ventana
            idxs = np.where(window == x)[0]
            if idxs.size:
                last_idx = idxs[-1]
                recency[x] = window_size - last_idx
            else:
                # si no aparece, le damos recency = window_size + 1
                recency[x] = window_size + 1
        # 2) Ordenamos los dígitos por recency (menor primero)
        ordered = sorted(digits, key=lambda x: recency[x])
        # 3) Encontramos la posición 1-based del siguiente número
        next_num = arr[end]
        #print("Next...", next_num)
        pos = ordered.index(next_num) + 1
        #print("posic...", pos)
        positions.append(pos)

    lis=[(pos - 1) // 2 + 1 for pos in positions]

    return positions, lis


def jugadas_sin_caer(zonas, zonas_posibles=range(1, 6)):
    n = len(zonas)
    ultimos = {}
    rev = zonas[::-1]
    
    for z in zonas_posibles:
        if z in zonas:
            ultimos[z] = rev.index(z) + 1
        else:
            ultimos[z] = n
    return ultimos


def Zonas_Histos(lista, t):
    print("\t\tZonas de Histogramas trabajando....")
    fun_promedios(lista, 0)
    print(lista[-20:], end="\t")
    print(f"P15: {sum(lista[-15:])/len(lista[-15:]):.3f}\tP10: {sum(lista[-10:])/len(lista[-10:]):.3f}\tP4: {sum(lista[-4:])/len(lista[-4:]):.3f}")
    # Función que asigna la zona según los umbrales
    if t==0:
        zonificar = lambda x: 1 if x < 4 else 2 if x < 8 else 3 if x < 12 else 4 if x < 17 else 5
    else:
        zonificar = lambda x: 1 if x < 3 else 2 if x < 5 else 3 if x < 7 else 4 if x < 9 else 5
    
    zonas = [zonificar(x) for x in lista]
    arr = np.array(zonas)
    conteos_np = np.bincount(arr)[1:]
    por_np = conteos_np / arr.size
    F_d=jugadas_sin_caer(zonas)
    print("\x1b[38;5;86m--    --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --\x1b[0m") 
    #print("\x1b[38;5;160m" + "\t".join(map(str, conteos_np))+ "\033[0m"  + "\t\t" + "\033[0m ")  
    r="\t".join(map("{:.2f}".format, por_np))
    print("\x1b[38;5;160m" + "\t".join(map(str, conteos_np))+ "\033[0m"  + "\t\t\t" + "\x1b[38;5;162m"+ r + "\033[0m ")    
    yY=procesar_lista_tres(zonas, 1, 0)
    print_colored_stats(yY, 0, Forma=2)
    print("\x1b[38;5;86m--    --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --\x1b[0m")
    a=procesar_Histogramas("        -- Zonas de Histogramas --", 0, 1, 20, zonas, F_d, 4, 2, 6)
    print("")
    a=procesar_Histogramas("        -- Zonas de Histogramas --", 4, 1, 15, zonas, F_d, 4, 2, 6)
    print("")
    a=procesar_Histogramas("        -- Zonas de Histogramas --", 5, 1, 15, zonas, F_d, 4, 2, 6)
    print("")
    print("\x1b[38;5;215m--    --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --\x1b[0m")
    print("")
    


def Zonas_Numeros(lista):
    print("\t\tZonas de numeros trabajando....")
    zonificar = lambda x: 1 if x < 4 else 2 if x < 7 else 3
    # Aplicamos la función a cada elemento de la lista original
    zonas = [zonificar(x) for x in lista]
    arr = np.array(zonas)
    conteos_np = np.bincount(arr)[1:]
    por_np = conteos_np / arr.size
    r="\t".join(map("{:.3f}".format, por_np))
    #print("\x1b[38;5;53m--    --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --\x1b[0m")
    #print("\x1b[38;5;160m" + "\t".join(map(str, conteos_np))+ "\033[0m"  + "\t\t" + "\x1b[38;5;162m"+ r + "\033[0m ")
    #yY=procesar_lista_tres(zonas, 0, 0)
    #print_colored_stats(yY, 0, Forma=2)
    #print("\x1b[38;5;53m--    --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --\x1b[0m")
    print("")
    Pr1, _, _, _, _ = procesar_e_imprimir_regresion("Zonas Numeros", 0, zonas, 2, 1, 4)
    print("")
    #Pr2, _, _, _, _ = procesar_e_imprimir_regresion("Zonas Numeros", 4, zonas, 2, 1, 4)
    print("\x1b[38;5;55m--    --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --\x1b[0m")
    #Pr3, _, _, _, _ = procesar_e_imprimir_regresion("Zonas Numeros", 5, zonas, 2, 1, 4)
    print("\x1b[38;5;55m--    --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --\x1b[0m")
    print("")
    #y=Sumar_listas(Pr1, Pr2, Pr3, Pr3)
    #Yy=multiplicar_lista(y, 0.2)
    Yy={}
    return Yy


def zones(serie: pd.Series, window: int = 50, max_count: int = None) -> list:
    
    if max_count is None:
        max_count = window + 1

    zones = []
    n = len(serie)
    for i in range(window, n):
        current = int(serie.iat[i])
        # 1) Ventana previa de tamaño `window`
        prev_window = serie.iloc[i-window : i].tolist()
        # 2) Conteo “jugadas sin caer” por dígito
        counts = {}
        rev = prev_window[::-1]
        for d in range(10):
            try:
                # idx_rev = 0 si cayó en la jugada justo anterior
                idx_rev = rev.index(d)
                counts[d] = idx_rev + 1
            except ValueError:
                counts[d] = max_count
        # 3) Ordenar dígitos por conteo
        ordered = sorted(counts, key=counts.get)
        # 4) Posición del dígito actual y clasificación
        pos = ordered.index(current)
        if counts[current] < 4:
            zone = 1
        elif counts[current] < 8:
            zone = 2
        elif counts[current] < 12:
            zone = 3 
        else:
            zone = 4
        # Fuerza zona 3 si nunca apareció en la ventana
        if counts[current] == max_count:
            zone = 4
        zones.append(zone)

    max_count = 50
    last_window = serie.iloc[-window:].tolist()
    counts = {}
    rev = last_window[::-1] # Revertir la ventana para encontrar la aparición más reciente
    for d in range(10):
        try:
            idx_rev = rev.index(d)
            counts[d] = idx_rev + 1
        except ValueError:
            counts[d] = max_count

    ordered = sorted(counts, key=counts.get)
    print("Zonas por Jugadas....")
    #yY=procesar_lista_tres(zones, 1, 0)
    #print_colored_stats(yY, 0, Forma=2)

    print("\033[95m" + "\t".join(map(str, ordered))+ "\033[0m")        
    Pr, _, _, _, _ = procesar_e_imprimir_regresion("Posicion Caidas", 0, zones, 2, 1, 5)
    Pr, _, _, _, _ = procesar_e_imprimir_regresion("Posicion Caidas", 4, zones, 2, 1, 5)
    Pr, _, _, _, _ = procesar_e_imprimir_regresion("Posicion Caidas", 5, zones, 2, 1, 5)

    return zones


def zones_by_freq(serie: pd.Series, window: int = 50) -> List[int]:
    zones: List[int] = []
    n = len(serie)
    
    for i in range(window, n):
        # 1. Extraer ventana y contar apariciones
        win = serie.iloc[i-window : i]
        freqs = (
            win.value_counts()
               .reindex(range(10), fill_value=0)
               .to_dict()
        )

        or_digits = sorted(freqs, key=lambda d: freqs[d])
        sorted_counts = [freqs[d] for d in or_digits]
        next_digit = int(serie.iat[i])
        rank = or_digits.index(next_digit) + 1

        if rank in (1, 2):
            zone = 1
        elif rank == 3:
            zone = 1 if (sorted_counts[2] - sorted_counts[1]) < 2 else 2
        elif 4 <= rank <= 7:
            zone = 2
        elif rank == 8:
            zone = 3 if (sorted_counts[9] - sorted_counts[8]) < 2 else 2
        else:  # rank 9 o 10
            zone = 3
        zones.append(zone)

    window = 50
    last_window_data = serie.iloc[-window:]
    freqs = (
        last_window_data.value_counts()
        .reindex(range(10), fill_value=0) # Asegura que todos los dígitos del 0-9 estén presentes
        .to_dict()
    )
    ordered = sorted(freqs, key=lambda d: freqs[d])
    print("Zonas por frecuencia....")
    #yY=procesar_lista_tres(zones, 1, 0)
    #print_colored_stats(yY, 0, Forma=0)

    print("\033[95m" + "\t".join(str(k) for k in ordered) + "\033[0m")
    Pr, _, _, _, _ = procesar_e_imprimir_regresion("Posicion Frecuencias", 0, zones, 2, 1, 4)
    Pr, _, _, _, _ = procesar_e_imprimir_regresion("Posicion Frecuencias", 4, zones, 2, 1, 4)
    Pr, _, _, _, _ = procesar_e_imprimir_regresion("Posicion Frecuencias", 5, zones, 2, 1, 4)
    return zones


def Dicc_probabilidad_ordenado(lista_numeros, I_ini=0, I_fin=10, cant=30,ventanas=(15, 20)):
    dicc1 = [sliding_global(lista_numeros, ventana, list(range(I_ini, I_fin))) 
    for ventana in ventanas]

    sums = defaultdict(float)
    for d in dicc1:
        for key, val in d.items():
            # si la clave es lista la convertimos a tupla
            hkey = tuple(key) if isinstance(key, list) else key
            sums[hkey] += val
    # 3) Calcular promedio
    n = len(dicc1)
    avg = {k: sums[k] / n for k in sums}

    return avg


def sliding_global(serie, k, dominio, last_n=20):
    """Retorna un diccionario {v: prob} con prob para cada valor en 'dominio'. """
    # 1) Lista de conteos de “siguiente” para todas las ventanas
    conteos = [serie[i : i + k].count(serie[i + k]) for i in range(len(serie) - k)]
    hist_counts    = Counter(conteos)
    total_ventanas = len(conteos)
    # 2) Tomamos solo las últimas last_n ventanas (o todas si hay menos)
    conteos_ult = conteos[-last_n:]
    # 3) Calculamos probabilidades p(c) para esos últimos conteos, Pero como queremos p(v) para cada valor, primero vemos
    # cuántas veces aparece v en la última ventana real
    p_global = [hist_counts[c] / total_ventanas for c in conteos]
    ultima_ventana = serie[-k:]
    Pr, _, _, _, _ = procesar_e_imprimir_regresion("-- Posicion segun Frec en 20 --", 4, conteos, 2, 0, 7)
    print("Ultimos:", conteos[-5:])
    yY=procesar_lista_tres(conteos, 1, 0)
    print_colored_stats(yY, 0, Forma=0)

    #    y a partir de ahí buscamos p(c_v)
    probs = {}
    for v in dominio:
        c_v      = ultima_ventana.count(v)
        # hist_counts.get(c_v, 0) → cuántas ventanas históricas tuvieron ese conteo
        p_v      = hist_counts.get(c_v, 0) / total_ventanas
        probs[v] = round(p_v, 3)
    #print(probs)

    return   probs #, conteos, p_global[-last_n:],


def filtrar_segmentos_df(errores_30, errores_15, errores_6, lista_sig, u, min_count=40):
    
    # 1) Crear DataFrame
    df = pd.DataFrame({
        "err30": errores_30,
        "err15": errores_15,
        "err6" : errores_6,
        "sig"  : lista_sig
    })
    
    # 2) Tolerancias sucesivas: 0%, 3%, 6%
    tolerancias = [0.00, 0.03, 0.6, 0.1]
    df_filtrado = pd.DataFrame()  # placeholder
    for rtol in tolerancias:
        if rtol == 0.0:
            # ciclo exacto
            mask = df["err15"] == u
            etiqueta = "exacto"
        else:
            # ciclo con tolerancia rtol
            low, high = sorted([u*(1-rtol), u*(1+rtol)])
            mask = df["err15"].between(low, high)
            etiqueta = f"±{rtol*100:.0f}%"
        df_filtrado = df[mask]

        # si ya alcanzamos el mínimo, rompemos el bucle
        if len(df_filtrado) >= min_count:
            break

    # 5) Extraer las sublistas resultantes
    errores_30a = df_filtrado["err30"].tolist()
    errores_15a = df_filtrado["err15"].tolist()
    errores_6a  = df_filtrado["err6"] .tolist()
    lista_siga  = df_filtrado["sig"] .tolist()

    return errores_30a, errores_15a, errores_6a, lista_siga, df_filtrado



def calcular_jerarquia_histo(lista: List[int], start: int = 30) -> List[int]:
    n = len(lista)
    # Convertimos start a índice 0-based
    i0 = start - 1
    # Diccionario con la última ocurrencia (0-based) de cada dígito
    last_occ = {d: -1 for d in range(10)}
    resultado: List[int] = []
    
    for i, val in enumerate(lista):
        last_occ[val] = i
        # Solo a partir de i0 y sin procesar el último elemento
        if i >= i0 and i < n - 1:
            # Construir orden de caída:
            # 1) Dígitos presentes, ordenados por última aparición descendente
            presentes = [d for d in range(10) if last_occ[d] != -1]
            presentes.sort(key=lambda d: -last_occ[d])
            # 2) Dígitos ausentes, ordenados ascendentemente
            ausentes = [d for d in range(10) if last_occ[d] == -1]
            ausentes.sort()
            orden_caida = presentes + ausentes
            # Buscar la posición 1-based del siguiente valor
            siguiente = lista[i + 1]
            posicion = orden_caida.index(siguiente) + 1
            resultado.append(posicion)
    return resultado


def porcentaje_coincidencias(F_d: dict, datos: list) -> dict:
    n = len(datos)
    limite = int(n * 0.95)        # 90% de n, redondeado hacia abajo
    primeros = datos[:limite]
    ultimos  = datos[limite:]
    c1, c2 = Counter(primeros), Counter(ultimos)
    n1, n2 = len(primeros), len(ultimos)

    errores = {}
    for k, valor in F_d.items():
        freq1 = c1[valor] / n1 if n1 else 0
        freq2 = c2[valor] / n2 if n2 else 0
        errores[k] = abs(freq1 - freq2)/freq1 if freq1 else 0
    return errores


def aplicar_svr(lista_30, lista_15, lista_6, lista_sig):
    X = np.column_stack((lista_30, lista_15, lista_6))
    y = np.array(lista_sig)
    # Definir el modelo SVR con kernel RBF (muy usado para relaciones no lineales)
    svr_model = svm.SVR(kernel='rbf')
    # Definir una rejilla de hiperparámetros para ajustar el parámetro de penalización C y el gamma del kernel.
    param_grid = {
        "C": [ 118.509189,118.509191, 118.509192],
        "gamma": ['scale', 'auto', 0.0175003, 0.0175004]  
    }
    # Usar GridSearchCV para buscar los mejores hiperparámetros usando validación cruzada
    grid_search = model_selection.GridSearchCV(svr_model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    # Extraemos el mejor modelo encontrado, la mejor puntuación, y los parámetros óptimos.
    best_svr = grid_search.best_estimator_
    cv_score = grid_search.best_score_
    best_params = grid_search.best_params_
    preds_all = best_svr.predict(X)
    errors_pct_all = (preds_all - y) / (y + 1e-15)
    mpe_all = np.mean(errors_pct_all)
    scores = cross_val_score(svr_model, X, y, cv=6, scoring='neg_mean_squared_error')  # métrica usada
    mean_score = scores.mean()    # promedio de los 5 folds
    std_score  = scores.std()     # desviación estándar, para ver variabilidad
    #print(f"CV : {mean_score:.5f}   Desv: {std_score:.5f}") #   CV  ==> {cv_score:.5f}")
    
    return best_svr, cv_score, mpe_all, errors_pct_all


def prediccion_bayesiana(lista_30, lista_15, lista_6, lista_sig):
    # Convertir listas a arrays
    np_lista_30 = np.array(lista_30)
    np_lista_15 = np.array(lista_15)
    np_lista_6  = np.array(lista_6)
    np_lista_sig = np.array(lista_sig)
    
    # Preparar datos de entrenamiento (todos menos el último)
    X_train = np.column_stack((np_lista_30[:-1], np_lista_15[:-1], np_lista_6[:-1]))
    y_train = np_lista_sig[:-1]
    
    # x_new: el último registro para predecir
    x_new = np.array([np_lista_30[-1], np_lista_15[-1], np_lista_6[-1]])
    
    # Definir el modelo bayesiano
    with pm.Model() as modelo:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=3)
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = alpha + pm.math.dot(X_train, beta)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)
        trace = pm.sample(2000, tune=2000, chains=4, target_accept=0.98, return_inferencedata=True)
    
    # Usar sample_posterior_predictive y obtener un diccionario simple
    with modelo:
        ppc = pm.sample_posterior_predictive(
            trace, 
            var_names=["alpha", "beta"], 
            random_seed=42, 
            return_inferencedata=False
        )
    
    # Extraer muestras usando el diccionario
    alpha_samples = ppc["alpha"]
    beta_samples = ppc["beta"]
    # Calcular predicciones para cada muestra
    predicciones = alpha_samples + np.dot(beta_samples, x_new)
    # Calcular la predicción final y el intervalo del 95%
    prediccion_media = np.mean(predicciones)
    pred_int2_5 = np.percentile(predicciones, 1)
    pred_int97_5 = np.percentile(predicciones, 99)
    
    return {
        "prediccion_media": prediccion_media,
        "int_95": (pred_int2_5, pred_int97_5),
        "trace": trace
    }


def leer_datos_excel(file_path):
    df = pd.read_excel(file_path)
    columna = pd.to_numeric(df['A'], errors='coerce').dropna()
    return columna


def obtener_siguiente_numero(columna):
    ultima_caida = columna.iloc[-1]
    return [columna[i + 1] for i in range(len(columna) - 1) if columna[i] == ultima_caida]


def Siguientes_lista(lista):
    ultima = lista[-1]
    return [siguiente for actual, siguiente in zip(lista, lista[1:]) 
            if actual == ultima]


def obtener_historial_caidas(columnas):
    caidas_columna = []
    ultimas_posiciones = [-1] * 10
    for i, idx in enumerate(columnas):
        valor=int(idx)
        if ultimas_posiciones[valor] == -1:
            jugadas = i + 1
        else:
            jugadas = i - ultimas_posiciones[valor]
        if jugadas > 50:
            jugadas = 50 if jugadas % 2 == 0 else 49
        caidas_columna.append(jugadas)
        ultimas_posiciones[valor] = i
    return caidas_columna


def obtener_siguiente_caidas(columnas):
    siguiente_caidas = []
    caidas = obtener_historial_caidas(columnas)
    ultima_caida = caidas[-1]
    for i in range(len(caidas) - 1):
        if caidas[i] == ultima_caida:
            siguiente = min(caidas[i + 1], 50)
            siguiente_caidas.append(siguiente)
    return siguiente_caidas


def Semanas(columna):
    grupo = columna.tail(50)
     # Calcular cuántas jugadas han pasado desde la última aparición de cada número
    apariciones = {}
    for num in range(10):
        if num in grupo.tolist():
            # Encontrar la posición de la última aparición y calcular la distancia desde el final
            ultima_posicion = len(grupo) - 1 - grupo[::-1].tolist().index(num)
            distancia = len(grupo)  - ultima_posicion
        else:
            # Si el número no aparece en el grupo, asignar 40 como default
            if num % 2 == 0:
                distancia = 50
            else :
                distancia = 49
        apariciones[num] = distancia
    return apariciones


def ultima_jerarquia(columna):
    grupo = columna.tail(50)
    frecuencias = {num: grupo.tolist().count(num) for num in range(10)}  # Usa .count() en la lista
    #print(frecuencias)  # Ahora imprimirá las frecuencias correctamente
    return frecuencias


def ultima_jerarquia_Lista(columna):
    grupo = columna[-50:]  # Extrae los últimos 50 elementos de la lista
    frecuencias = {num: grupo.count(num) for num in range(10)}  # Cuenta ocurrencias
    #print(frecuencias)  # Muestra las frecuencias en consola
    return frecuencias


def calcular_jerarquias(columna):
    jerarquia = []
    posiciones = []
    for i in range(len(columna) - 2, 51,-1 ):
        grupo = columna[max(0, i - 49):i + 1]
        frecuencias = {num: grupo.value_counts().get(num, 0) for num in range(10)}
        primeras_apariciones = {num: (len(grupo) - 1 - grupo[::-1].tolist().index(num) if num in grupo.tolist() else 60) for num in range(10)}
        ordenados = sorted(frecuencias.items(), key=lambda x: (x[1], primeras_apariciones[x[0]]))
        jerarquia.append(ordenados)
        #print(ordenados)
        if i < len(columna) - 1:
            siguiente_dato = columna[i + 1]
            for pos, (num, _) in enumerate(ordenados):
                if num == siguiente_dato:
                    posiciones.append(pos + 1)
                    #print(pos)
                    break
    jerarquia.reverse()
    return jerarquia, posiciones


def calcular_alpha_prior(columna):
    limite = int(len(columna) * 0.9)  # Define el 90% del total
    grupo = columna.iloc[:limite]  # Obtiene la parte inicial de la columna
    alpha_prior = {num: grupo.tolist().count(num) for num in range(10)}  # Cuenta ocurrencias
    return alpha_prior


def calcular_alpha_prior_Lista(columna):
    limite = int(len(columna) * 0.9)  # Define el 90% del total
    grupo = columna[:limite]  # Extrae los últimos 50 elementos de la lista
    frecuencias = {num: grupo.count(num) for num in range(10)}  # Cuenta ocurrencias
    return frecuencias


def calcular_mayores_pares(columna):
    mayores = [num for num in columna if num > 4]
    pares = [num for num in columna if num % 2 == 0]
    return mayores, pares


def aplicar_regresion_logistica_mayor_menor(columna):
    if len(columna) < 70:
        print("Error: Insuficientes datos para regresión logística (mayor/menor).")
        return None
    X = np.array(columna[:-1]).reshape(-1, 1)
    y = np.array([1 if num > 4 else 0 for num in columna[1:]])
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = linear_model.LogisticRegression().fit(X_train, y_train)
    ultimo_numero = np.array(columna.iloc[-1]).reshape(1, 1)
    return modelo.predict_proba(ultimo_numero)[0][1]


def aplicar_regresion_logistica_par_impar(columna):
    if len(columna) < 70:
        print("Error: Insuficientes datos para regresión logística (par/impar).")
        return None
    X = np.array(columna[:-1]).reshape(-1, 1)
    y = np.array([1 if num % 2 == 0 else 0 for num in columna[1:]])
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = linear_model.LogisticRegression().fit(X_train, y_train)
    ultimo_numero = np.array(columna.iloc[-1]).reshape(1, 1)
    return modelo.predict_proba(ultimo_numero)[0][1]


def aplicar_regresion_ponderada(lista_30, lista_15, lista_6, lista_sig):
    if not all([lista_30, lista_15, lista_6, lista_sig]) or not len(lista_30) == len(lista_15) == len(lista_6) == len(lista_sig):
        print("Error en los datos para regresión ponderada.")
        return None
    # Preparación de datos
    X = np.array([lista_30, lista_15, lista_6]).T
    y = np.array(lista_sig)
    X_ols = sm.add_constant(X)
    # Regresión OLS inicial
    modelo_ols = sm.OLS(y, X_ols).fit()
    residuos = modelo_ols.resid
    ### 🟢 MÉTODO 1: Pesos inversos al error (el que ya tenías)
    pesos_1 = 1 / (residuos ** 2 + 1e-6)
    modelo_wls_1 = sm.WLS(y, X_ols, weights=pesos_1).fit(cov_type='HC0')
    
    ### 🔵 MÉTODO 2: Pesos con raíz cuadrada del error
    pesos_2 = 1 / np.sqrt(residuos ** 2 + 1e-6)
    modelo_wls_2 = sm.WLS(y, X_ols, weights=pesos_2).fit(cov_type='HC0')
    
    # Extraer coeficientes para ambas versiones
    coeficientes_1 = modelo_wls_1.params
    coeficientes_2 = modelo_wls_2.params
    # Calcular métricas para ambas versiones
    resultados = {
        "Método 1: Pesos inversos al error": {
            "Intercepto": coeficientes_1[0],
            "Coef. lista_30": coeficientes_1[1],
            "Coef. lista_15": coeficientes_1[2],
            "Coef. lista_6": coeficientes_1[3],
            "Porcentaje de variabilidad explicada por el modelo (cuanto mayor, mejor)": modelo_wls_1.rsquared,
            "Promedio del error al cuadrado entre lo predicho y lo real (menores valores son mejores)": metrics.mean_squared_error(y, modelo_wls_1.predict(X_ols)),
            "Promedio absoluto de error entre lo predicho y lo real (indica desviación en las mismas unidades)": metrics.mean_absolute_error(y, modelo_wls_1.predict(X_ols))
        },
        "Método 2: Pesos con raíz cuadrada del error": {
            "Intercepto": coeficientes_2[0],
            "Coef. lista_30": coeficientes_2[1],
            "Coef. lista_15": coeficientes_2[2],
            "Coef. lista_6": coeficientes_2[3],
            "Porcentaje de variabilidad explicada por el modelo (cuanto mayor, mejor)": modelo_wls_2.rsquared,
            "Promedio del error al cuadrado entre lo predicho y lo real (menores valores son mejores)": metrics.mean_squared_error(y, modelo_wls_2.predict(X_ols)),
            "Promedio absoluto de error entre lo predicho y lo real (indica desviación en las mismas unidades)": metrics.mean_absolute_error(y, modelo_wls_2.predict(X_ols))
        }
    }
    return resultados


def aplicar_regresion_elasticnet(Yy, lista_30, lista_15, lista_6, lista_sig):
    #alpha=0.86, l1_ratio=0.14, max_iter=8000, tol=0.000001
    
    if not all([lista_30, lista_15, lista_6, lista_sig]) or not len(lista_30) == len(lista_15) == len(lista_6) == len(lista_sig):
        print("Error en los datos para Elastic Net.")
        return None
    X = np.array([lista_30, lista_15, lista_6]).T
    y = np.array(lista_sig)
    m_iter=6000
    tole=0.00001

    if Yy==0:
        modelo_enet = linear_model.ElasticNet(alpha=0.95, l1_ratio=0.0001, max_iter=m_iter, tol=tole).fit(X, y)
    elif Yy==1:
        modelo_enet = linear_model.ElasticNet(alpha=0.0035, l1_ratio=0.7, max_iter=m_iter, tol=tole).fit(X, y)
    elif Yy==2:
        modelo_enet = linear_model.ElasticNet(alpha=0.004, l1_ratio=0.5, max_iter=m_iter, tol=tole).fit(X, y)
    else:
        modelo_enet = linear_model.ElasticNet(alpha=0.0001, l1_ratio=0.75, max_iter=m_iter, tol=tole).fit(X, y)
    
    return modelo_enet.intercept_, modelo_enet.coef_


def aplicar_regresion_robusta(lista_30, lista_15, lista_6, lista_sig):
    if not all([lista_30, lista_15, lista_6, lista_sig]) or not len(lista_30) == len(lista_15) == len(lista_6) == len(lista_sig):
        print("Error en los datos para regresión robusta.")
        return None

    # Preparación de datos
    X = np.array([lista_30, lista_15, lista_6]).T
    y = np.array(lista_sig)
    X_ols = sm.add_constant(X)  # Agregar intercepto
    # Aplicar regresión robusta con método HuberT
    modelo_rlm = sm.RLM(y, X_ols, M=sm.robust.norms.HuberT()).fit()
    # Calcular un pseudo R² manualmente:
    ss_res = np.sum(modelo_rlm.resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_pseudo = 1 - ss_res / ss_tot if ss_tot != 0 else None
    coeficientes = modelo_rlm.params

    return {
        "Intercepto": coeficientes[0],
        "Coef. lista_30": coeficientes[1],
        "Coef. lista_15": coeficientes[2],
        "Coef. lista_6": coeficientes[3],
        "Estimación de variabilidad explicada por el modelo (Pseudo R²)": r2_pseudo
    }


def analizar_siguientes_numeros_para_probabilidades(siguiente_numeros):
    if not siguiente_numeros:
        return {i: 0 for i in range(10)}
    frecuencia = {i: siguiente_numeros.count(i) for i in range(10)}
    total = len(siguiente_numeros)
    
    return {num: freq / total if total > 0 else 0 for num, freq in frecuencia.items()}


def calcular_probabilidades_regresion(params_wls, params_enet, lista_30, lista_15, lista_6):
    if params_wls is None or params_enet is None or not lista_30 or not lista_15 or not lista_6:
        return {"WLS": None, "ElasticNet": None}
    if len(lista_30) != len(lista_15) or len(lista_30) != len(lista_6):
        print("Error: Las listas de promedios tienen longitudes inconsistentes.")
        return {"WLS": None, "ElasticNet": None}

    ultima_media_30 = lista_30[-1] if lista_30 else 0
    ultima_media_15 = lista_15[-1] if lista_15 else 0
    ultima_media_6 = lista_6[-1] if lista_6 else 0
    prediccion_wls = params_wls[0] + params_wls[1] * ultima_media_30 + params_wls[2] * ultima_media_15 + params_wls[3] * ultima_media_6
    intercepto_enet, coefs_enet = params_enet
    prediccion_enet = intercepto_enet + coefs_enet[0] * ultima_media_30 + coefs_enet[1] * ultima_media_15 + coefs_enet[2] * ultima_media_6

    return {"WLS": prediccion_wls, "ElasticNet": prediccion_enet}


def calcular_regresion_wls_metodo1(lista_30, lista_15, lista_6, lista_sig):
     
    params = aplicar_regresion_ponderada(lista_30, lista_15, lista_6, lista_sig)
    if params:
        return {
            "Intercepto": params["Método 1: Pesos inversos al error"]["Intercepto"],
            "Coef_lista_30": params["Método 1: Pesos inversos al error"]["Coef. lista_30"],
            "Coef_lista_15": params["Método 1: Pesos inversos al error"]["Coef. lista_15"],
            "Coef_lista_6": params["Método 1: Pesos inversos al error"]["Coef. lista_6"]
        }
    else:
        return None


def calcular_regresion_wls_metodo2(lista_30, lista_15, lista_6, lista_sig):
      
    params = aplicar_regresion_ponderada(lista_30, lista_15, lista_6, lista_sig)
    if params:
        return {
            "Intercepto": params["Método 2: Pesos con raíz cuadrada del error"]["Intercepto"],
            "Coef_lista_30": params["Método 2: Pesos con raíz cuadrada del error"]["Coef. lista_30"],
            "Coef_lista_15": params["Método 2: Pesos con raíz cuadrada del error"]["Coef. lista_15"],
            "Coef_lista_6": params["Método 2: Pesos con raíz cuadrada del error"]["Coef. lista_6"]
        }
    else:
        return None


def calcular_regresion_robusta(lista_30, lista_15, lista_6, lista_sig):
        
    params = aplicar_regresion_robusta(lista_30, lista_15, lista_6, lista_sig)
    if params:
        return {
            "Intercepto": params["Intercepto"],
            "Coef_lista_30": params["Coef. lista_30"],
            "Coef_lista_15": params["Coef. lista_15"],
            "Coef_lista_6": params["Coef. lista_6"],
            "R2": params["Estimación de variabilidad explicada por el modelo (Pseudo R²)"]
        }
    else:
        return None


def predecir_con_regresion(parametros, params_enet, lista_30, lista_15, lista_6):
    # Primero, convierte el diccionario 'parametros' en una lista en el orden esperado:
    if parametros is None:
        return None
    parametros_lista = [
        parametros["Intercepto"],
        parametros["Coef_lista_30"],
        parametros["Coef_lista_15"],
        parametros["Coef_lista_6"]
    ]
    return calcular_probabilidades_regresion(parametros_lista, params_enet, lista_30, lista_15, lista_6)


def procesar_regresiones(datos, Xx):
    lista_30, lista_15, lista_6, lista_sig = datos
    # Aseguramos que se usan segmentos consistentes:
    datos_regresion = (
        lista_30[-len(lista_sig):],
        lista_15[-len(lista_sig):],
        lista_6[-len(lista_sig):],
        lista_sig
    )
    
    # Extraer parámetros para cada método
    params_wls1 = calcular_regresion_wls_metodo1(*datos_regresion)
    params_wls2 = calcular_regresion_wls_metodo2(*datos_regresion)
    params_rlm  = calcular_regresion_robusta(*datos_regresion)
    params_enet = aplicar_regresion_elasticnet(Xx, lista_30, lista_15, lista_6, lista_sig)  # Suponiendo que esta función ya existe
    # Calcular predicciones
    resultados_m1 = predecir_con_regresion(params_wls1, params_enet, lista_30, lista_15, lista_6)
    resultados_m2 = predecir_con_regresion(params_wls2, params_enet, lista_30, lista_15, lista_6)
    resultados_rlm = predecir_con_regresion(params_rlm, params_enet, lista_30, lista_15, lista_6)
    
    return {
        "WLS_Metodo1": resultados_m1,
        "WLS_Metodo2": resultados_m2,
        "RLM": resultados_rlm,
    }


def inferir_probabilidades_bayesianas(frecuencias, alpha_prior):
    # Usamos un prior uniforme si no se proporciona
    if alpha_prior is None:
        alpha_prior = {i: 1 for i in range(10)}
    
    # Calculamos la suma total del prior (en este caso, 10, ya que 1 para cada dígito)
    total_alpha = sum(alpha_prior.values())
    # Total de observaciones (suma de todas las frecuencias)
    total_frecuencias = sum(frecuencias.get(i, 0) for i in range(10))
    # Suma total del posterior
    total_posterior = total_alpha + total_frecuencias
    # Calculamos la media del posterior para cada dígito:
    probabilidades_posterior = {}
    for i in range(10):
        # alpha posterior: prior + observación
        alpha_post = alpha_prior.get(i, 1) + frecuencias.get(i, 0)
        probabilidades_posterior[i] = alpha_post / total_posterior
    
    return probabilidades_posterior


def ultimos_promedios_list(data: List[float], Va, xx) -> List[float]: 
    return [(Va + v) / xx for v in data.values()]


def ultimos_promedios_series(data: pd.Series) -> List[float]:
    if len(data) < 15:
        # no hay ni un solo promedio completo
        return pd.Series(dtype=float)
    medias = data.rolling(15).mean().dropna()
    return medias.iloc[-10:].tolist()


def calcular_promedios_de_errores(columna):
    PTotal = np.mean(columna)
    # Calcular lista_30 y errores_30
    lista_30 = [np.mean(columna[i - 30:i]) for i in range(80, len(columna))]
    errores_30 = [(p - PTotal) / PTotal for p in lista_30]
    # Calcular lista_15 y errores_15 con índice correcto
    lista_10 = [np.mean(columna[i - 15:i]) for i in range(80, len(columna))]
    errores_10 = [(p - PTotal) / PTotal for p in lista_10]
    # Calcular lista_4 y errores_4
    lista_4 = [np.mean(columna[i - 12:i]) for i in range(80, len(columna))]
    errores_4 = [(p - PTotal) / PTotal for p in lista_4]
    # Calcular lista_sig
    lista_sig = [np.mean(columna[i - 16:i + 1]) for i in range(80, len(columna) + 1)]
    lista_sig.pop()
    Tot14=sum(columna[-16:]) 
    u30=sum(columna[-30:])/len(columna[-30:])
    u10=sum(columna[-15:])/len(columna[-15:])
    u4=sum(columna[-12:])/len(columna[-12:])
    u30=(u30-PTotal)/PTotal
    u10=(u10-PTotal)/PTotal
    u4=(u4-PTotal)/PTotal

    return errores_30, errores_10, errores_4, lista_sig, Tot14, PTotal, u30, u10, u4


def calcular_promedios_y_errores(columna, Pos):
    PTotal = np.mean(columna)
    if Pos<3:
        print("Datos muy altos") 
        # Calcular lista_30 y errores_30
        lista_30 = [np.mean(columna[i - 30:i]) for i in range(40, len(columna))]
        errores_30 = [(p - PTotal) / PTotal for p in lista_30]
        # Calcular lista_15 y errores_15 con índice correcto
        lista_10 = [np.mean(columna[i - 15:i]) for i in range(40, len(columna))]
        errores_10 = [(p - PTotal) / PTotal for p in lista_10]
        # Calcular lista_4 y errores_4
        lista_4 = [np.mean(columna[i - 12:i]) for i in range(40, len(columna))]
        errores_4 = [(p - PTotal) / PTotal for p in lista_4]
        # Calcular lista_sig
        lista_sig = [np.mean(columna[i - 16:i + 1]) for i in range(40, len(columna) + 1)]
        lista_sig.pop()
        Tot14=sum(columna[-16:])
        u30=sum(columna[-30:])/len(columna[-30:])
        u10=sum(columna[-15:])/len(columna[-15:])
        u4=sum(columna[-12:])/len(columna[-12:])
        xxx=u10
        L_3, L_1, L_4, L_sig, bn = last_interval_expand(lista_30, lista_10, lista_4, lista_sig, xxx)
        best_svr, cv_score, PromT, ETo = aplicar_svr(L_3, L_1, L_4, L_sig)
        nuevo_dato = np.array([[u30, u10, u4]])
        prediccion = best_svr.predict(nuevo_dato)
        Pprom=prediccion[0]
        print(aviso_ansi(f"Nuevo dato: {Pprom:.3f}   {bn}",(225, 225, 225), (224, 112, 10)))
        u30=(u30-PTotal)/PTotal
        u10=(u10-PTotal)/PTotal
        u4=(u4-PTotal)/PTotal

    elif Pos<5:
        # Calcular lista_30 y errores_30
        print("Datos Bajos")
        lista_30 = [np.mean(columna[i - 28:i]) for i in range(40, len(columna))]
        errores_30 = [(p - PTotal) / PTotal for p in lista_30]
        # Calcular lista_15 y errores_15 con índice correcto
        lista_10 = [np.mean(columna[i - 20:i]) for i in range(40, len(columna))]
        errores_10 = [(p - PTotal) / PTotal for p in lista_10]
        # Calcular lista_4 y errores_4
        lista_4 = [np.mean(columna[i - 10:i]) for i in range(40, len(columna))]
        errores_4 = [(p - PTotal) / PTotal for p in lista_4]
        # Calcular lista_sig
        lista_sig = [np.mean(columna[i - 11:i + 1]) for i in range(40, len(columna) + 1)]
        lista_sig.pop()    
        Tot14=sum(columna[-11:])
        u30=sum(columna[-28:])/len(columna[-28:])
        u10=sum(columna[-20:])/len(columna[-20:])
        u4=sum(columna[-10:])/len(columna[-10:])
        xxx=u10
        L_3, L_1, L_4, L_sig, bn = last_interval_expand(lista_30, lista_10, lista_4, lista_sig, xxx)
        best_svr, cv_score, PromT, ETo = aplicar_svr(L_3, L_1, L_4, L_sig)
        nuevo_dato = np.array([[u30, u10, u4]])
        prediccion = best_svr.predict(nuevo_dato)
        Pprom=prediccion[0]
        print(aviso_ansi(f"Nuevo dato: {Pprom:.3f}   {bn}",(225, 225, 225), (117, 174, 90)))
        u30=(u30-PTotal)/PTotal
        u10=(u10-PTotal)/PTotal
        u4=(u4-PTotal)/PTotal

    else:
        # Calcular lista_30 y errores_30
        print("Datos muy Bajos")
        lista_30 = [np.mean(columna[i - 25:i]) for i in range(40, len(columna))]
        errores_30 = [(p - PTotal) / PTotal for p in lista_30]
        # Calcular lista_15 y errores_15 con índice correcto
        lista_10 = [np.mean(columna[i - 20:i]) for i in range(40, len(columna))]
        errores_10 = [(p - PTotal) / PTotal for p in lista_10]
        # Calcular lista_4 y errores_4
        lista_4 = [np.mean(columna[i - 15:i]) for i in range(40, len(columna))]
        errores_4 = [(p - PTotal) / PTotal for p in lista_4]
        # Calcular lista_sig
        lista_sig = [np.mean(columna[i - 16:i + 1]) for i in range(40, len(columna) + 1)]
        lista_sig.pop()    
        Tot14=sum(columna[-16:])
        u30=sum(columna[-25:])/len(columna[-25:])
        u10=sum(columna[-20:])/len(columna[-20:])
        u4=sum(columna[-15:])/len(columna[-15:])
        xxx=u10
        L_3, L_1, L_4, L_sig, bn = last_interval_expand(lista_30, lista_10, lista_4, lista_sig, xxx)
        best_svr, cv_score, PromT, ETo = aplicar_svr(L_3, L_1, L_4, L_sig)
        nuevo_dato = np.array([[u30, u10, u4]])
        prediccion = best_svr.predict(nuevo_dato)
        Pprom=prediccion[0]
        print(aviso_ansi(f"Nuevo dato: {Pprom:.3f}   {bn}",(225, 225, 225), (224, 230, 40)))
        u30=(u30-PTotal)/PTotal
        u10=(u10-PTotal)/PTotal
        u4=(u4-PTotal)/PTotal
    
    print(f"Ultimos datos de 6: {u30:.3f} {u10:.3f} {u4:.3f}")
    return errores_30, errores_10, errores_4, lista_sig, Tot14, PTotal, u30, u10, u4, xxx,Pprom 

 
def procesar_e_imprimir_regresion(titulo, Pos, Lista, Nn, start=0,stop=10):
    # Códigos ANSI
    BG_BLUE = '\033[46m'
    RED_ON_YELLOW = "\033[33;46m"
    RESET        = "\033[0m"
    
    print(aviso_ansi(f"Resultados para {titulo}:",(118, 5, 30), (240, 220, 90)))
    print("\033[31m" + "\t".join(str(k) for k in Lista[-10:]) + "\033[0m") 
    print("--  --  --  --  --  --  --  --")
    L_30, L_15, L_6, L_sig, Sum14, PROM, u3, u1, u4, ud, p = calcular_promedios_y_errores(Lista, Pos)
    liss=[]
    liss.append(p)
    best_svr, cv_score, PromT, ETo = aplicar_svr(L_30, L_15, L_6, L_sig)
    nuevo_dato = np.array([[u3, u1, u4]])
    prediccion = best_svr.predict(nuevo_dato)
    Pprom=prediccion[0]
    liss.append(Pprom)
    ye1=sum(ETo[-9:-5])/len(ETo[-9:-5])
    ye2=sum(ETo[-3:])/3

    Predm=0
    er_30a, er_15a, er_6a, lis_siga, df_debug = filtrar_segmentos_df(L_30, L_15, L_6, L_sig, u1)
    if len(df_debug)>40:
        best_svr1, cv_score1, mpe_all1, errors_all1 = aplicar_svr(er_30a, er_15a, er_6a, lis_siga)
        #nuevo_datom = np.array([[u3, u1, u4]])
        pred = best_svr.predict(nuevo_dato)
        Predm= pred[0]

    liss.append(Predm)
    l3, l1, l4, ls=last_interval(L_30, L_15, L_6, L_sig, u1)
    best_svr4, cv_score4, PromT4, ETo4 = aplicar_svr(l3, l1, l4, ls)
    nuevo_dato4 = np.array([[u3, u1, u4]])
    prediccion4 = best_svr4.predict(nuevo_dato4)
    Pprom4=prediccion4[0]
    liss.append(Pprom4)
    Le_30, Le_15, Le_6, Le_sig, Sume14, PROMe, u30, u10, u4 = calcular_promedios_de_errores(ETo)
    best_svr, cv_score, PromTe, EToe = aplicar_svr(Le_30[:-1], Le_15[:-1], Le_6[:-1], Le_sig[:-1])
    nuevo_datoe = np.array([[u30, u10, u4]])
    pred = best_svr.predict(nuevo_datoe)
    print(f"PROMES 31: {pred[0]:.3f}  y1: {ye1:.3f}  y2: {ye2:.3f} ")

    fun_promedios(ETo, 2)
    #print("\nParámetros para SVR:", best_params)
    print(f"\033[31mGeneral :\033[0m", end="\t")
    print(f"\033[1;31;47m{PROM:.3f}\033[0m", end="\t\t") 

    
    # Verificar que todas las listas contengan datos, tengan la misma longitud y mayor a 50.
    if (L_30 and L_15 and L_6 and L_sig and len(L_30) == len(L_15) == len(L_6) == len(L_sig) and len(L_30) > 60):
        datos_regresion = (L_30, L_15, L_6, L_sig)
        resultados = procesar_regresiones(datos_regresion, Nn)
        # Extraer el valor de ElasticNet (se asume que es consistente en todos los métodos)
        primer_metodo = next(iter(resultados))
        elasticnet1 = resultados["RLM"]["ElasticNet"]
        P_gral=Pprom
        if Pos<3:
            print(f"\x1b[48;5;202mRegresion : {Pprom:.3f}\033[0m", end="\t")
        elif Pos<5:
            print(f"\x1b[48;5;157mRegresion : {Pprom:.3f}\033[0m", end="\t")
        else:
            print(f"\x1b[43;5;223mRegresion : {Pprom:.3f}\033[0m", end="\t")
        print(f"'ElasticNet': {elasticnet1:.3f}   Nuevo:{Predm:.3f}   Otro: {Pprom4:.3f}")
        print(" -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --")
        n_valores, errores_ajustados, xx = imprimir_valores_y_errores(Sum14, Pos, P_gral,start,stop)
        
        minY=min(n_valores)
        maxY=max(n_valores)
        
        if P_gral<minY:  
            P_gral=minY*1.005
            
        if P_gral>maxY:
            P_gral=maxY*0.995
            
        n_valores, errores_ajustados, xx = imprimir_valores_y_errores(Sum14, Pos, P_gral, start, stop, col_val="\033[92m", col_err="\033[92m")
        print(f"\tNuevo Prom: {P_gral:.3f}")
        #print("-----------------------          ------------------------------")
        #Prom_Gral = solicitar_nuevo_prom(minY, maxY)
        #nuevos_valores, errores_ajustados, xx = imprimir_valores_y_errores(Sum14, Pos, Prom_Gral, start, stop, col_val="\033[92m", col_err="\033[92m")
        #print("")
        return  errores_ajustados, n_valores, Sum14, P_gral, xx

    else:
        print(f"No hay datos suficientes o las listas no cumplen las condiciones para {titulo}.")
        return None, None, None

def print_recencia(rec: Dict[int, int]) -> None:
    keys_l = sorted(rec.keys())
    vals_l = [rec[k] for k in keys_l]

    sorted_items = sorted(rec.items(), key=lambda kv: kv[1])
    keys_r, vals_r = zip(*sorted_items)

    # 2) Formatea con ancho fijo de columna
    col_w = 4
    sep_left = " │"
    left_idx_line   = "".join(f"{k:>{col_w}}" for k in keys_l) + sep_left
    left_vals_line  = "".join(f"{v:>{col_w}}" for v in vals_l) + sep_left
    right_idx_line  = "".join(f"{k:>{col_w}}" for k in keys_r)
    right_vals_line = "".join(f"{v:>{col_w}}" for v in vals_r)

    # 3) Genera border dinámico
    border_l = "-" * len(left_idx_line)
    border_r = "-" * len(right_idx_line)

    # 4) Imprime las 5 líneas: border / índices / border / valores / border
    spacer = "   "
    for left, right in [
        (border_l,      border_r),
        (left_idx_line, right_idx_line),
        (border_l,      border_r),
        (left_vals_line, right_vals_line),
        (border_l,      border_r),
    ]:
        print(f"{left}{spacer}{right}")



def procesar_lista_tres(data: List[float], tipo: int, Pos: int) -> Dict[str, Optional[float]]:
    print("         -- Procesamiento de tres en linea --")
    min_count =30
    max_iters = 2
    last1 = data[-1]
    last2 = data[-2]
    last3 = data[-3]
    
    # 1. Validación de longitud mínima
    if len(data) < 500:
        return {"N": 0, "Ptot": 0, "Ppos": 0, "Pneg": 0}
    
    # 2. Configurar paso según Pos
    step = 1.01 #if Pos == 0 else 0.002
    lo2 = hi2 = last2
    lo3 = hi3 = last3
    collected: List[float] = []

    # 4. Búsqueda iterativa hasta reunir min_count datos
    for j in range(max_iters):
        # Expande ambas ventanas
        for i in range(1, len(data) - 1):
            primer = data[i - 2]
            anterior = data[i - 1]
            actual   = data[i]
            # Ambas ventanas deben cumplirse simultáneamente
            if last1 == actual and lo2 < anterior < hi2 and lo3 < primer < hi3:
                # Añade el siguiente dato tras 'actual'
                collected.append(data[i + 1])

        if len(collected) >= min_count:
            break

        lo2, hi2 = lo2 - step, hi2 + step
        lo3, hi3 = lo3 - step, hi3 + step

    #xy=analizar_collected(collected, last1, tipo, Pos, 20)
    count = len(collected)
    mean_total = sum(collected) / count if count else None
    # 5. Clasificación en positivos y negativos

    pos = [x for x in collected if x > last1]
    igu = [x for x in collected if x == last1]
    neg = [x for x in collected if x < last1]

    # 6. Top-4 valores más comunes
    c = Counter(collected)
    for idx, (valor, cnt) in enumerate(c.most_common(3), start=1):
        print(f".. {valor:.0f} → {cnt} ", end="\t")
    print("")
    # 8. Medias finales y últimos datos
    mean_pos = sum(pos) / len(pos) if pos else 0
    mean_igu = sum(igu) / len(igu) if igu else 0
    mean_neg = sum(neg) / len(neg) if neg else 0
    ultimos = data[-20:]
    sig_2 = collected[-20:]

    return {"N": count, "Ptot": mean_total, "Ppos": mean_pos, "Tpos": len(pos), "Pigu": mean_igu, "Tigu": len(igu),"Pneg": mean_neg, "Tneg": len(neg),
        "Ult": last1, "Ant":ultimos, "Sig":sig_2 }


def procesar_lista_dos(
    data: List[float],
    tipo: int,
    Pos: int
) -> Dict[str, Optional[float]]:
    print("    Lista dos EN LINEA .......")

    last1, last2 = data[-1], data[-2]
    # 1. Validación de longitud mínima
    if len(data) < 100:
        return {"N": 0, "Ptot": 0, "Ppos": 0, "Pigu": 0, "Pneg": 0}
    # 2. Configurar paso según Pos
    step = 1.01
    #lo1 = hi1 = last1
    lo2 = hi2 = last2
    collected: List[float] = []

    lo2, hi2 = lo2 - step, hi2 + step

    for i in range(1, len(data) - 1):
        anterior = data[i - 1]
        actual   = data[i]
        # Ambas ventanas deben cumplirse simultáneamente
        if last1 == actual and lo2 < anterior < hi2:
            # Añade el siguiente dato tras 'actual'
            collected.append(data[i + 1])

    count = len(collected)
    mean_total = sum(collected) / count if count else None

    pos = [x for x in collected if x > last1]
    igu = [x for x in collected if x == last1]
    neg = [x for x in collected if x < last1]

    # 6. Top-4 valores más comunes
    c = Counter(collected)
    for idx, (valor, cnt) in enumerate(c.most_common(3), start=1):
        print(f"{idx}. {valor:.0f} → {cnt} ", end="\t")

    # 7. Caídas por rango (suponiendo que devuelve List[(rango, cnt)], _)
    x, y = Caidas_por_rango(collected, last1, Pos)
    num = sum(((li + ls) / 2) * cnt for ((li, ls), cnt) in x)
    den = sum(cnt for _, cnt in x)
    prom_pon = num / den if den else 0
    #for (li, ls), cnt in x:
    #    print(f"{li:.3f}-{ls:.3f}→ {cnt}", end="\t")
    print(f"\tProm pond: {prom_pon:.3f}")

    # 8. Medias finales y últimos datos
    mean_pos = sum(pos) / len(pos) if pos else 0
    mean_igu = sum(igu) / len(igu) if igu else 0
    mean_neg = sum(neg) / len(neg) if neg else 0
    ultimos_20 = data[-25:]
    sig_20 = collected[-25:]

    return {"N": count, "Ptot": mean_total, "Ppos": mean_pos, "Tpos": len(pos), "Pigu": mean_igu, "Tigu": len(igu),"Pneg": mean_neg, "Tneg": len(neg),
        "Ult": last1, "Ant": ultimos_20, "Sig": sig_20 }


def procesar_lista(data: List[float], tipo: int, Pos: int, y=1, Forma=0 ) -> Dict[str, Optional[float]]:
    min_count = 50
    max_iters = 400
    step = 0.0045 if Pos == 0 else 0.002
    
    if len(data) < 80:
        return {"N": 0, "Ptot": 0, "Ppos": 0, "Pneg": 0}
    
    last = data[-1]
    collected: List[float] = []
    lo=last
    hi=last
    for j in range(max_iters):
        lo, hi = lo - step, hi + step
        collected = [
            data[i + 1]
            for i in range(len(data) - 1)
            if lo < data[i] < hi
        ]
        if len(collected) >= min_count:
            break
    
    count = len(collected)
    mean_total = sum(collected) / count if count > 0 else None

    if tipo == 0:
        pos = [x for x in collected if x >= 0]
        neg = [x for x in collected if x < 0]
    else:
        pos = [x for x in collected if x >= last]
        neg = [x for x in collected if x < last]

    c = Counter(collected)
    # Obtenemos los dos elementos más comunes:
    if y==1:
        top2 = c.most_common(4)
        parts = []
        nume = sum(valor * cuenta for valor, cuenta in top2)
        deno = sum(cuenta for _, cuenta in top2)
        P_parcial=nume / deno if deno != 0 else 0
        print(f"\t\t\tlo {lo:.3f}  hi {hi:.3f} j:{j}\tParcial:{P_parcial:.3f} ", end="\t")
        
        x,y=Caidas_por_rango(collected, last, Pos)
        num   = sum(((li + ls) / 2) * cnt for (li, ls), cnt in x)
        den = sum(cnt for _, cnt in x)
        prom_pon = num / den if den else 0
        print(f"Prom pond: {prom_pon:.2f}")    
        #if Forma==0:
        #    print("  " + "| ".join(f"{li:.2f}-{ls:.2f}→ {cnt}" for (li, ls), cnt in x)+ "\t\t\t" + f"Prom pond: {prom_pon:.2f}")
        #else:
        #    print("  " + "| ".join(f"{li:.3f}-{ls:.3f}→ {cnt}" for (li, ls), cnt in x)+ "\t\t" + f"Prom pond: {prom_pon:.3f}")

    mean_pos = sum(pos) / len(pos) if pos else None
    mean_neg = sum(neg) / len(neg) if neg else None
    ul5 = data[-40:]
    u5 = collected[-40:]
    return {"N": count, "Ptot": mean_total, "Ppos": mean_pos, "Tpos": len(pos), "Pneg": mean_neg, "Tneg": len(neg),
        "Ult": last, "Ant":ul5, "Sig":u5 }


def Caidas_por_rango(valores, v_base,  P=0,porcentajes=None):
    # 1. Definir porcentajes por defecto si no se pasan
    if P == 0:
        # Rango absoluto en pasos de 0.01
        bordes = np.round(np.arange(v_base - 0.25, v_base + 0.251, 0.02), 4)
    elif P==1:
        porcentajes = [-0.24, -0.22, -0.2, -0.18, -0.16, -0.14, -0.12, -0.1, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24 ]
        factores = 1 + np.array(porcentajes)
        bordes   = np.round(v_base * factores, 2)
        bordes.sort()
    else :
        porcentajes = [-0.15, -0.14, -0.13, -0.12, -0.11, -0.1, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
        factores = 1 + np.array(porcentajes)
        bordes   = np.round(v_base * factores, 2)
        bordes.sort()
    
    # 3. Generar histograma
    conteos, edges = np.histogram(valores, bins=bordes)
    # 4. Obtener índices de los dos rangos con más caídas
    idx_ordenados = np.argsort(conteos)
    top2_idxs     = idx_ordenados[-5:][::-1]
    # 5. Construir estructuras de salida
    top2 = [((edges[i], edges[i+1]), int(conteos[i])) for i in top2_idxs]
    histo = [(edges[i], edges[i+1], int(conteos[i])) for i in range(len(conteos))]
    
    return top2, histo


def Imprimir_datos_listas(stats: dict, mode: int = 0):
    # ANSI settings
    BG_GRAY = "\x1b[48;2;220;220;220m"
    RESET   = "\x1b[0m"
    # Variables que deben ir sin decimales
    int_vars = {"N", "Tpos", "Tigu", "Tneg","Sig", "Ant"}

    def fmt_scalar(name: str, val: float) -> str:
        """ Construye 'name=value' con fondo gris y texto rojo/azul o gris si es None.
        Enteros sin decimales, floats con 3 decimales.
        """
        # 1) Caso None
        if val is None:
            r, g, b = (128, 128, 128)   # gris para valores no disponibles
            txt = "N/A"
        else:
            # 2) decide color según signo
            r, g, b = (255, 0, 0) if val < 0 else (0, 0, 255)
            if name in int_vars:
                txt = f"{int(val)}"
            else:
                txt = f"{val:.3f}"

        return (
            f"{BG_GRAY}"
            f"\x1b[38;2;{r};{g};{b}m"
            f" {name}={txt} "
            f"{RESET}"
        )
    
    def fmt_number(val: float) -> str:
        """ Sólo número (3 decimales), con fondo gris y texto rojo/azul. """
        r, g, b = (255, 0, 0) if val < 0 else (0, 0, 255)
        txt = f"{val:.2f}"

        return (
            f"{BG_GRAY}"
            f"\x1b[38;2;{r};{g};{b}m"
            f" {txt} "
            f"{RESET}"
        )

    # 1) Línea de escalares
    order = ["Ult", "N", "Tpos", "Tigu", "Tneg",
             "Ptot", "Ppos", "Pigu", "Pneg"]
    line1 = "  ".join(
        fmt_scalar(name, stats[name])
        for name in order
        if name in stats and not isinstance(stats[name], list)
    )
    print(line1)
    print("")
    # si sólo se pide la primera línea, cortamos
    if mode == 1:
        return

    # 2) Línea Siguiente (sig)
    sigs = stats.get("Sig", [])
    if sigs:
        y1=sum(sigs[-15:])/len(sigs[-15:]) if len(sigs[-15:])>0 else 0
        y2=sum(sigs[-10:])/len(sigs[-10:]) if len(sigs[-10:])>0 else 0
        y3=sum(sigs[-10:-5])/len(sigs[-10:-5]) if len(sigs[-10:])>0 else 0
        y4=sum(sigs[-5:])/len(sigs[-5:]) if len(sigs[-5:])>0 else 0
        
        line2_vals = " ".join(fmt_number(v) for v in sigs[-10:])
        print(f"Sig. :\t{line2_vals}\t\tP15: {y1:.2f}   P10: {y2:.2f}   PM: {y3:.2f}   P5: {y4:.2f}")

    # 3) Línea Anterior (ant)
    ants = stats.get("Ant", [])
    if len(ants)>0:
        y1=sum(ants[-15:])/len(ants[-15:])
        y2=sum(ants[-10:])/len(ants[-10:])
        y3=sum(ants[-10:-5])/len(ants[-10:-5])
        y4=sum(ants[-5:])/len(ants[-5:])
    
        line2_vals = " ".join(fmt_number(v) for v in sigs[-10:])
        print(f"Sig. :\t{line2_vals}\t\tP15: {y1:.2f}   P10: {y2:.2f}   PM: {y3:.2f}   P5: {y4:.2f}")
        print("")


def print_colored_stats(stats: dict, mode: int = 0, Forma=1):
    # ANSI settings
    BG_GRAY = "\x1b[48;2;220;220;220m"
    RESET   = "\x1b[0m"
    # Variables que deben ir sin decimales
    int_vars = {"N", "Tpos", "Tneg"}

    def fmt_scalar(name: str, val: float) -> str:
        """ Construye 'name=value' con fondo gris y texto rojo/azul o gris si es None.
        Enteros sin decimales, floats con 3 decimales.
        """
        # 1) Caso None
        if val is None:
            r, g, b = (128, 128, 128)   # gris para valores no disponibles
            txt = "N/A"
        else:
            # 2) decide color según signo
            r, g, b = (255, 0, 0) if val < 0 else (0, 0, 255)
            if name in int_vars:
                txt = f"{int(val)}"
            else:
                txt = f"{val:.3f}"

        return (
            f"{BG_GRAY}"
            f"\x1b[38;2;{r};{g};{b}m"
            f" {name}={txt} "
            f"{RESET}"
        )

    def fmt_number(val: float) -> str:
        """ Sólo número (3 decimales), con fondo gris y texto rojo/azul. """
        r, g, b = (255, 0, 0) if val < 0 else (0, 0, 255)
        if Forma==0:
            txt = f"{val:.2f}"
        elif Forma==1:
            txt = f"{val:.3f}"
        else:
            txt = f"{val:.0f}"

        return (
            f"{BG_GRAY}"
            f"\x1b[38;2;{r};{g};{b}m"
            f" {txt} "
            f"{RESET}"
        )

    # 1) Línea de escalares
    order = ["Ult", "N", "Tpos", "Tneg",
             "Ptot", "Ppos", "Pneg"]
    line1 = "  ".join(
        fmt_scalar(name, stats[name])
        for name in order
        if name in stats and not isinstance(stats[name], list)
    )
    print(line1)
    print("")
    # si sólo se pide la primera línea, cortamos
    if mode == 1:
        return

    # 2) Línea Siguiente (sig)
    sigs = stats.get("Sig", [])
    if sigs:
        if len(sigs[-15:])!=0:
            y1=sum(sigs[-15:])/len(sigs[-15:])  
        else:
            y1=0
        
        if len(sigs[-10:])!=0:    
            y2=sum(sigs[-10:])/len(sigs[-10:]) 
        else:
            y2=0
        
        if len(sigs[-10:-5])!=0:
            y3=sum(sigs[-10:-5])/len(sigs[-10:-5]) 
        else:
            y3=0
        if len(sigs[-5:])!=0:
            y4=sum(sigs[-5:])/len(sigs[-5:])
        else:
            y4=0
    
        line2_vals = " ".join(fmt_number(v) for v in sigs[-10:])
        print(f"Sig. :\t{line2_vals}\t\tP15: {y1:.2f}   P10: {y2:.2f}   PM: {y3:.2f}   P5: {y4:.2f}")

    # 3) Línea Anterior (ant)
    ants = stats.get("Ant", [])
    if len(ants)>0:
        if len(ants[-15:])!=0:
            y1=sum(ants[-15:])/len(ants[-15:])  
        else:
            y1=0
        
        if len(ants[-10:])!=0:    
            y2=sum(ants[-10:])/len(ants[-10:]) 
        else:
            y2=0
        
        if len(ants[-10:-5])!=0:
            y3=sum(ants[-10:-5])/len(ants[-10:-5]) 
        else:
            y3=0
        if len(ants[-5:])!=0:
            y4=sum(ants[-5:])/len(ants[-5:])
        else:
            y4=0
    
        line2_vals = " ".join(fmt_number(v) for v in ants[-10:])
        print(f"Act. :\t{line2_vals}\t\tP15: {y1:.2f}   P10: {y2:.2f}   PM: {y3:.2f}   P5: {y4:.2f}")
        print("")


def aviso_ansi(texto: str, fg: tuple = (64, 34, 28), bg: tuple = (253, 226, 228)) -> str:
    fg_code = f"\033[38;2;{fg[0]};{fg[1]};{fg[2]}m"
    bg_code = f"\033[48;2;{bg[0]};{bg[1]};{bg[2]}m"
    reset   = "\033[0m"
    return f"{fg_code}{bg_code}{texto}{reset}"


def solicitar_nuevo_prom(min_val, max_val):
    prompt  = aviso_ansi(
        f"⚠️  Introduce nuevo Prom_Gral "
        f"(entre {min_val:.3f} y {max_val:.3f}): ",(151,79,68),(191,183,182)
    )
    while True:
        try:
            nuevo = float(input(prompt))
        except ValueError:
            print(aviso_ansi("→ Entrada no válida. Sólo números."))
            continue

        if min_val <= nuevo <= max_val:
            return nuevo
        print(aviso_ansi(
            f"→ Fuera de rango. Debe ser entre {min_val:.3f} y {max_val:.3f}."
        ))


def colorear(valor, etiqueta):
    # amarillo brillante para la etiqueta
    azul  = "\033[34m"
    rojo  = "\033[31m"
    amarillo = "\033[92m"
    reset = "\033[0m"

    num_color = azul if valor >= 0 else rojo
    return f"{amarillo}{etiqueta}{reset} = {num_color}{valor:.3f}{reset}"

def colorear2(valor):
    fondo = "\033[104m"  # Fondo gris claro
    if valor >= 0:
        texto = "\033[34m" if valor > 0 else "\033[30m"  # Azul si >0, negro si ==0
    else:
        texto = "\033[31m"  # Rojo si negativo
    return f"{fondo}{texto}{valor:.3f}\033[0m"


def imprimir_valores_y_errores(s14, Po, p_gral, start=0, stop=10, col_val="\033[93m", col_err="\033[96m"):
    formateados = []
    errores    = []
    nuevos     = []
    ajustados  = []
    x=0
    
    for i in range(start, stop):
        if Po<3:
            val   = (s14 + i) / 17
            x=19
        elif Po<5:
            val   = (s14 + i) / 12
            x=10
        else:
            val   = (s14 + i) / 17
            x=14
            
        err   = (val - p_gral) / p_gral
        formateados.append(f"{val:.3f}\t")
        errores.append(f"{err:.3f}")
        ajustados.append(err * -0.999 if err < 0 else err)
        nuevos.append(val)

    print(f"{col_val}{' '.join(formateados)}\033[0m")
    print("\t".join(colorear2(float(e)) for e in errores))
    return nuevos, ajustados, x


def inferir_probabilidades_bayesianas1(orden_digitos, historial_posiciones):
    num_posiciones = len(orden_digitos)  # Número total de posiciones
    print("Cantidad Posiciones",num_posiciones)
    # Calculamos los totales de prior y evidencia
    total_evidence = sum(orden_digitos.values())
    print("Total Prior",total_prior)
    total_prior = sum(historial_posiciones.values())
    print("Total Evidence",total_evidence)
    total_combined = total_prior + total_evidence

    # Calculamos probabilidades posteriores
    posterior_probs = {}
    for digito in orden_digitos:
        posterior_probs[digito] = (orden_digitos[digito] + historial_posiciones[digito]) / total_combined
        print(f"Dígito {digito} => orden: {orden_digitos[digito]}, historial: {historial_posiciones[digito]}")
    
    # Devolvemos el diccionario con las mismas claves (0, 1, 2, …)
    return posterior_probs


def imprimir_tabla(Titulo, data, es_decimal=False, highlight_key=None):
    # ANSI colors
    RED = "\033[1;31m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    # Caso diccionario: se mantienen las claves en el orden de inserción
    if isinstance(data, dict):
        claves = list(data.keys())
        cabecera = [str(k) for k in claves]
        valores = [data[k] for k in claves]
    # Caso lista: se usan los índices de la lista como cabecera
    elif isinstance(data, list):
        cabecera = [str(i) for i in range(len(data))]
        valores = data
    else:
        print("Tipo de dato no soportado (se esperaba diccionario o lista).")
        return

    # Formateo de la fila de datos:
    if es_decimal:
        # Se formatean los números a 4 dígitos decimales
        fila_datos = [f"{v:.3f}" for v in valores]
    else:
        fila_datos = [str(v) for v in valores]

    # Determina el ancho mínimo que se necesita para cada celda (según la mayor longitud entre cabecera y datos)
    ancho_min = max(max(len(s) for s in cabecera), max(len(s) for s in fila_datos))
    # Se añade un pequeño padding (2 espacios adicionales)
    ancho_celda = ancho_min + 2
    num_cols = len(cabecera)
    
    # índices de las 4 columnas centrales
    mid         = num_cols // 2
    offset      = 1 if num_cols % 2 else 0
    centro_idxs = list(range(mid - 2 + offset, mid + 2))

    # índice de la clave a resaltar en rojo
    idx_hl = None
    if highlight_key is not None:
        try:
            idx_hl = cabecera.index(str(highlight_key))
        except ValueError:
            pass

    # función de formateo de cada celda
    def fmt(s, i):
        cell = f"{s:>{ancho_celda}}"
        if i == idx_hl:
            return f"{RED}{cell}{RESET}"
        if i in centro_idxs:
            return f"{YELLOW}{cell}{RESET}"
        return cell

    # línea de borde
    borde = "-" * ((ancho_celda + 1) * num_cols + 1)

    # --- 4) Impresión de la tabla ---
    print(f"\n******  {Titulo}  *****")
    print(borde)
    print(" ".join(fmt(c, i) for i, c in enumerate(cabecera)) + " │")
    print(borde)
    print(" ".join(fmt(d, i) for i, d in enumerate(fila_datos)) + " │")
    print(borde)


def imprimir_Nmedios(lista):
    #ANSI
    RED = "\033[1;31m"
    RESET = "\033[0m"
    # Formatear siempre como decimales de 4 dígitos
    vals_fmt = [f"{v:.4f}" for v in lista]
    # Encontrar índice del mínimo (numérico)
    min_idx = min(range(len(lista)), key=lambda i: lista[i])
    # Cabeceras (1,2,3,4,5)
    headers = [str(i) for i in range(1,6)]

    # Calcular ancho de celda
    ancho = max(max(len(h) for h in headers),
                max(len(v) for v in vals_fmt)) + 3

    # Borde
    borde = "-" * ((ancho + 1) * len(headers) + 1)

    # Función de formateo con color para el mínimo
    def fmt(s, idx):
        cell = f"{s:>{ancho}}"
        if idx == min_idx:
            return f"{RED}{cell}{RESET}"
        return cell

    # Imprimir
    print("\n***** Valores Numeros Medios *****")
    print(borde)
    # Headers
    print("│" + " ".join(fmt(h, i) for i, h in enumerate(headers)) + " │")
    print(borde)
    # Valores
    print("│" + "  ".join(fmt(v, i) for i, v in enumerate(vals_fmt)) + " │")
    print(borde)


def imprimir_tabla_N(titulo: str, data, es_decimal: bool = False, color_titulo: str = "default", blink: bool = False,
    light: bool = False):
    # ——— Configuración ANSI de colores/blink ———
    base = {
        "black": 30, "red": 31, "green": 32, "yellow": 33,
        "blue": 34, "magenta": 35, "cyan": 36, "white": 37,
        "default": 39
    }
    bright = light or color_titulo.startswith("light_")
    clave = color_titulo.replace("light_", "") if bright else color_titulo
    codigo_color = base.get(clave.lower(), base["default"])
    if bright:
        codigo_color += 60
    codigos = []
    if blink:
        codigos.append("5")
    codigos.append(str(codigo_color))
    seq_inicio = f"\033[{';'.join(codigos)}m"
    seq_fin    = "\033[0m"

    # ——— Extraer cabecera y valores ———
    if isinstance(data, dict):
        claves = list(data.keys())
        cabecera = [str(k) for k in claves]
        valores  = [data[k] for k in claves]
    elif isinstance(data, list):
        cabecera = [str(i) for i in range(len(data))]
        valores  = data
    else:
        print("Tipo no soportado (esperado dict o list).")
        return

    # ——— Formatear valores ———
    fila_datos = []
    for v in valores:
        if es_decimal and isinstance(v, float):
            fila_datos.append(f"{v:.4f}")
        else:
            fila_datos.append(str(v))

    # ——— Calcular anchos por columna (contenido vs. cabecera) + padding ———
    n = len(cabecera)
    anchos = []
    for i in range(n):
        ancho_max = max(len(cabecera[i]), len(fila_datos[i]))
        anchos.append(ancho_max + 2)   # +2 espacios de padding

    # ——— Construir líneas de borde ———
    # Ej: +----+------+----+
    partes = ["+" + "-" * a for a in anchos]
    borde = "".join(partes) + "+"

    # ——— Construir filas de texto ———
    # Cabecera: | key0 | key1 | ...
    cab = "|"
    datos = "|"
    for i in range(n):
        cab += f" {cabecera[i].center(anchos[i]-2)} |"
        datos += f" {fila_datos[i].center(anchos[i]-2)} |"

    # ——— Impresión final ———
    print()
    # título centrado sobre la tabla
    ancho_tabla = len(borde)
    print(seq_inicio + titulo.center(ancho_tabla) + seq_fin)
    print(borde)
    print(cab)
    print(borde)
    print(datos)
    print(borde)


def calcular_probabilidades_desde_historial(orden_digitos, historial_posiciones):
    # 2. Inicializamos un diccionario para contar apariciones, para cada dígito
    conteos = {digito: 0 for digito in orden_digitos}
    
    # 3. Recorrer el historial.
    #    Se asume que los números del historial son posiciones 1-indexadas.
    for pos in historial_posiciones:
        index = pos - 1  # Convertir a índice 0-indexado
        if 0 <= index < len(orden_digitos):
            digito = orden_digitos[index]
            conteos[digito] += 1
        else:
            print(f"Advertencia: posición {pos} fuera de rango en el historial.")
       
    # 4. Normalizamos los conteos para obtener probabilidades.
    total = sum(conteos.values())
    if total > 0:
        probabilidades = {digito: conteos[digito] / total for digito in conteos}
    else:
        # Si no hay registros en el historial, puede hacerse una distribución uniforme u otra política
        probabilidades = {digito: 0.005 for digito in conteos}
    return probabilidades


def inferir_probabilidades_bayesianas(orden_digitos, historial_posiciones):
    evidence_history = historial_posiciones[-40:]
    prior_history = historial_posiciones[:-40]
    print()
    # Inicializamos conteos para cada posición (usaremos posiciones 1 a N, donde N = len(orden_digitos))
    num_posiciones = len(orden_digitos)  # normalmente 10
    prior_counts = {pos: 0 for pos in range(1, num_posiciones + 1)}
    evidence_counts = {pos: 0 for pos in range(1, num_posiciones + 1)}
    
    # Contar ocurrencias en el prior
    for pos in prior_history:
        if 1 <= pos <= num_posiciones:
            prior_counts[pos] += 1
        else:
            print(f"Advertencia: posición {pos} fuera de rango en el prior.")
    # Contar ocurrencias en la evidencia
    for pos in evidence_history:
        if 1 <= pos <= num_posiciones:
            evidence_counts[pos] += 1
        else:
            print(f"Advertencia: posición {pos} fuera de rango en la evidencia.")
            
    # La distribución posterior para la posición i es:
    # posterior(prob_i) = (prior_counts[i] + evidence_counts[i]) / (total_prior + total_evidence)
    total_prior = sum(prior_counts.values())
    total_evidence = sum(evidence_counts.values())  # debería ser 40 si todo está bien
    total_combined = total_prior + total_evidence
    
    posterior_probs = {}
    for pos in range(1, num_posiciones + 1):
        posterior_probs[pos] = (prior_counts[pos] + evidence_counts[pos]) / total_combined
    
    # Mostramos información de chequeo
    print("-- Prior counts (por posición) --")
    print(prior_counts)
    print("-- Evidence counts (últimas 40 jugadas) --")
    print(evidence_counts)
    print("-- Posterior (distribución de posiciones) --")
    print(posterior_probs)
    
    # Reconversión: asignamos la probabilidad calculada para cada posición
    # al dígito correspondiente segun el orden en orden_digitos.
    # Si la posición es 1 (1-indexado), corresponde a orden_digitos[0].
    final_probabilidades = {}
    for pos in range(1, num_posiciones + 1):
        digito = orden_digitos[pos - 1]
        final_probabilidades[digito] = posterior_probs[pos]
    return final_probabilidades


def ordenar_por_valor(d, ascendente=True):
    if d is None:
        # No hay nada que ordenar; devuelvo lista vacía
        return []
        
    return dict(
        sorted(d.items(), key=lambda par: par[1], reverse=not ascendente)
    )


def mostrar_dict(d):
    for clave, valor in d.items():
        print(f"{clave}: {valor}")


def mostrar_formato(num):
    entero = int(num)
    decimal = round(num - entero, 4)
    dec_str = f"{decimal:.4f}"[2:]  # '1456', sin '0.'
    
    if num < 1:
        print(f".{dec_str}")
    else:
        print(f"{entero}.{dec_str}")


def Lista2_con_map(lista):
    """Aplica y = x//2 + 1 a cada elemento usando map+lambda."""
    return list(map(lambda x: x // 2 + 1, lista))


def Histo2_con_map(lista, T):
    """Aplica y = (x-1)//2 + 1 a cada elemento usando map+lambda."""
    return list(map(lambda x: (x-1) // T + 1, lista))


def ordenar_lista(lista: list, ascendente: bool = True) -> list:
    return sorted(lista, reverse=not ascendente)


def split_segments(H: List[int], n: int = 3) -> List[List[int]]:
    """Divide H en n trozos lo más parejos posible."""
    L = len(H)
    size = L // n
    segments = []
    for i in range(n-1):
        segments.append(H[i*size : (i+1)*size])
    segments.append(H[(n-1)*size : ])
    return segments


def compute_percentages(seg: List[int], possible: List[int]) -> Dict[int, float]:
    """ Cuenta cuántas veces aparece cada valor en 'possible' dentro de 'seg' y devuelve porcentaje (0–1).  """
    cnt = Counter(seg)
    total = len(seg) if seg else 1
    return {v: cnt.get(v, 0)/total for v in possible}


def mean_percentages(per_list: List[Dict[int, float]]) -> Dict[int, float]:
    """Dado un listado de dicts {v: pct}, devuelve su promedio por clave."""
    keys = per_list[0].keys()
    n = len(per_list)
    return {k: sum(d[k] for d in per_list)/n for k in keys}


def compute_fd_errors(
    mean_p: Dict[int,float],
    last_p: Dict[int,float],
    F_d:    Dict[int,int]
) -> Tuple[Dict[int,float], Dict[int,float]]:
    """ Devuelve dos dicts: error_abs_fd[k] = abs(last_p[x] - mean_p[x])
      error_rel_fd[k] = (last_p[x] - mean_p[x]) / mean_p[x] donde x = F_d[k]. """
    error_abs_fd = {}
    error_rel_fd = {}

    for k, x in F_d.items():
        # Si x no está en mean_p o last_p, saltamos o le ponemos 0
        m = mean_p.get(x)
        l = last_p.get(x)
        if m is None or l is None or m == 0:
            error_abs_fd[k] = None
            error_rel_fd[k] = None
        else:
            abs_err = abs(l - m)
            rel_err = (l - m) / m
            error_abs_fd[k] = abs_err
            error_rel_fd[k] = rel_err
    return error_abs_fd, error_rel_fd


def analyze_frecuencias(
    H: List[int],
    F_d: Dict[int,int],
    max_val: int,
    n_segments: int,
    last_n: int 
) -> Tuple[Dict[int,float], Dict[int,float], Dict[int,float], Dict[int,float]]:
    """Retorna:
      mean_p   = promedio de los n_segments porcentajes históricos
      last_p   = porcentaje de los últimos last_n valores
      errors   = error absoluto por valor
      error_fd = asigna a cada clave de F_d su error correspondiente 
    """
    possible = list(range(1, max_val+1))
    # 1) Segmentar y calcular porcentajes históricos
    segs = split_segments(H, n_segments)
    historical = [compute_percentages(s, possible) for s in segs]
    mean_p = mean_percentages(historical)
    # 2) Porcentaje últimos N
    last_seg = H[-last_n:]
    last_p = compute_percentages(last_seg, possible)
    # 3) Errores absolutos por valor
    error_abs_fd, error_rel_fd = compute_fd_errors(mean_p, last_p, F_d)
    # 4) Mapear errores según F_d
    error_fd = {k: error_abs_fd.get(k, None) for k in F_d.keys()}
    return mean_p, last_p, error_abs_fd, error_rel_fd, error_fd


def Sumar_diccionarios(*dicts):
    if dicts and isinstance(dicts[-1], int):
        Tipo, dicts = dicts[-1], dicts[:-1]
    else:
        Tipo, dicts = 0, dicts

    total = defaultdict(float)
    cuenta = defaultdict(int)
    # Sumar y contar apariciones
    if Tipo==0:
        for d in dicts:
            for clave, valor in d.items():
                total[clave] += valor
                cuenta[clave] += 1
    else:
            for d in dicts:
                for clave, valor in d.items():
                    total[clave] += abs(valor)
                    cuenta[clave] += 1
    
    # Calcular promedio por clave
    return {clave: total[clave] / cuenta[clave] for clave in total}
    #return ordenar_por_valor(diccionario_sumado, ascendente=False)


def remapear_por_posicion(claves_ordenadas: list, dic_posiciones: dict)-> dict:
    resultado = {}
    n = len(claves_ordenadas)
    for pos, val in dic_posiciones.items():
        i = pos - 1
        print(f"  probando pos={pos} → i={i}, rango 0–{len(claves_ordenadas)-1}")
        if 0 <= i < n:
            resultado[claves_ordenadas[i]] = val
        else:
            pass
    return resultado


def promedios_y_errores_lista(data, zz, Pos, yy, nx):
    n = len(data)
    pgral = sum(data) / n
    Predm=0
    def medias_ventana(k):
            return [ sum(data[i-k: i]) / k for i in range(55, n) ]

    if zz < 1:
        print("Datos muy altos")
        lista_30 = medias_ventana(30)
        lista_10 = medias_ventana(15)
        lista_6  = medias_ventana(12)
        lista_sig = [ sum(data[i-(nx-1): i+1]) / len(data[i-(nx-1): i+1]) for i in range(55, n+1) ]
        u30=sum(data[-29:])/len(data[-29:])
        u10=sum(data[-14:])/len(data[-14:])
        u4=sum(data[-11:])/len(data[-11:])
        lista_sig.pop()
        er_30a, er_15a, er_6a, lis_siga, df_debug = filtrar_segmentos_df(lista_30, lista_10, lista_6, lista_sig, u10)
        if len(df_debug)>30:
            best_svr1, cv_score1, mpe_all1, errors_all1 = aplicar_svr(er_30a, er_15a, er_6a, lis_siga)
            nuevo_dato = np.array([[u30, u10, u4]])
            pred = best_svr1.predict(nuevo_dato)
            Predm= pred[0]

        u30=(u30-pgral)/pgral
        u10=(u10-pgral)/pgral
        u4=(u4-pgral)/pgral
    
    elif  zz < 5:
        print("Datos Bajos")

        lista_30 = medias_ventana(25)
        lista_10 = medias_ventana(12)
        lista_6  = medias_ventana(10)
        lista_sig = [ sum(data[i-(nx-1): i+1]) / len(data[i-(nx-1): i+1]) for i in range(55, n+1) ]
        u30=sum(data[-24:])/len(data[-24:])
        u10=sum(data[-11:])/len(data[-11:])
        u4=sum(data[-9:])/len(data[-9:])
        lista_sig.pop()
        er_30a, er_15a, er_6a, lis_siga, df_debug = filtrar_segmentos_df(lista_30, lista_10, lista_6, lista_sig, u10)
        if len(df_debug)>30:
            best_svr1, cv_score1, mpe_all1, errors_all1 = aplicar_svr(er_30a, er_15a, er_6a, lis_siga)
            nuevo_dato = np.array([[u30, u10, u4]])
            pred = best_svr1.predict(nuevo_dato)
            Predm= pred[0]
        u30=(u30-pgral)/pgral
        u10=(u10-pgral)/pgral
        u4=(u4-pgral)/pgral
    else :
        print("Datos muy Bajos")
        #def medias_ventana(k):
        #    return [ sum(data[i-k: i]) / k for i in range(30, n) ]
        lista_30 = medias_ventana(35)
        lista_10 = medias_ventana(15)
        lista_6  = medias_ventana(10)
        lista_sig = [ sum(data[i-(nx-1): i+1]) / len(data[i-(nx-1): i+1]) for i in range(55, n+1) ]
        u30=sum(data[-34:])/len(data[-34:])
        u10=sum(data[-14:])/len(data[-14:])
        u4=sum(data[-9:])/len(data[-9:])
        lista_sig.pop()
        er_30a, er_15a, er_6a, lis_siga, df_debug = filtrar_segmentos_df(lista_30, lista_10, lista_6, lista_sig, u10)
        if len(df_debug)>30:
            best_svr1, cv_score1, mpe_all1, errors_all1 = aplicar_svr(er_30a, er_15a, er_6a, lis_siga)
            nuevo_dato = np.array([[u30, u10, u4]])
            pred = best_svr1.predict(nuevo_dato)
            Predm= pred[0]*0.95
        u30=(u30-pgral)/pgral
        u10=(u10-pgral)/pgral
        u4=(u4-pgral)/pgral

    def errores(lista_medias):
        return [ (m - pgral) / pgral for m in lista_medias ]

    errores_30 = errores(lista_30)
    errores_15 = errores(lista_10)
    errores_6  = errores(lista_6)
                      
    # 4) suma de los últimos 14 valores
    suma_14 = sum(data[-(nx-1):])
    
    return errores_30, errores_15, errores_6, lista_sig, suma_14, pgral, u30, u10, u4, Predm

def calcular_nuevos_y_errores(Valores: Dict[str, float], Sum14: float, Prom_Gral: float, Nx) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
    errores_por_clave = {}
    nuevos_por_clave  = {}
    formateados1       = []
    formateados2       = []

    for clave, incremento in Valores.items():
        nuevo_valor = (Sum14 + incremento) / Nx
        error = (nuevo_valor - Prom_Gral) / Prom_Gral
        nuevos_por_clave[clave]  = nuevo_valor
        errores_por_clave[clave] = error
        formateados1.append(f"{nuevo_valor:.2f}")
        formateados2.append(f"{error:.3f}")
    return nuevos_por_clave, errores_por_clave, formateados1, formateados2


def procesar_regresion_Histo(titulo, zz, P, Lista, Valores, Nn, medio=5, ante=2, fin=11, start=1, stop=15 ):
    BG_BLUE = '\033[46m'
    RED_ON_YELLOW = "\033[37;45m"
    RESET        = "\033[0m"
    
    print(aviso_ansi(f"Resultados para {titulo}:", (118, 5, 30), (240, 220, 90)))
    y = dict(sorted(Valores.items(), key=lambda item: item[1]))
    Predm=0
    L_30, L_15, L_6, L_sig, Sum14, PROM,u3, u1, u, Predm = promedios_y_errores_lista(Lista, zz, P, y, Nn)
    best_svr, cv_score, PromT, ETo = aplicar_svr(L_30, L_15, L_6, L_sig)
    nuevo_dato = np.array([[u3, u1, u]])
    prediccion = best_svr.predict(nuevo_dato)
    Pprom=prediccion[0]
    
    er_30a, er_15a, er_6a, lis_siga, df_debug = filtrar_segmentos_df(L_30, L_15, L_6, L_sig, u1)
    if len(df_debug)>30:
        best_svr1, cv_score1, mpe_all1, errors_all1 = aplicar_svr(er_30a, er_15a, er_6a, lis_siga)
        pred = best_svr1.predict(nuevo_dato)
        Predm= pred[0]

    Le_30, Le_15, Le_6, Le_sig, Sume14, PROMe, ue3, ue1, ue = calcular_promedios_de_errores(ETo)
    best_svr, cv_score, PromTe, EToe = aplicar_svr(Le_30, Le_15, Le_6, Le_sig)
    nuevo_datoe = np.array([[ue3, ue1, ue]])
    prediccion = best_svr.predict(nuevo_datoe)
    Prede= prediccion[0]
    ye1=sum(ETo[-9:])/9
    ye2=sum(ETo[-6:])/6
    ye3=sum(ETo[-3:])/3

    print(f"Predes: {Prede:.3f}   {ye1:.3f}   {ye2:.3f}   {ye3:.3f}")

    fun_promedios(ETo, 2)
    #print("Parámetros para SVR:", best_params)
    print("\033[31mPromedio :\033[0m", end="\t")
    print(f"\033[1;91;47m{PROM:.4f}\033[0m", end="\t\t")  
    
    if len(L_30) < 70:
        print(f"No hay datos suficientes en {titulo}.")
        # Ruta por defecto (cuando no hay datos suficientes)
        default = ({i: 0 for i in range(1, 10)}, None, 0)
        return default
    
    if (L_30 and L_15 and L_6 and L_sig and 
        len(L_30) == len(L_15) == len(L_6) == len(L_sig) ):
        datos_regresion = (L_30, L_15, L_6, L_sig)
        resultados = procesar_regresiones(datos_regresion, zz)
        # Extraer el valor de ElasticNet (se asume que es consistente en todos los métodos)
        primer_metodo = next(iter(resultados))
        valor_elasticnet = resultados[primer_metodo]["ElasticNet"]
        primer_metodo = next(iter(resultados))
        elasticnet1 = resultados["RLM"]["ElasticNet"]
        P_gral=Pprom
        if zz <3:
            print(f"\x1b[48;5;202mRegresion : {Pprom:.3f}\033[0m", end="\t")
        elif zz <5:
            print(f"\x1b[48;5;157mRegresion : {Pprom:.3f}\033[0m", end="\t")
        else:
            print(f"\x1b[43;5;223mRegresion : {Pprom:.3f}\033[0m", end="\t")
        print(f"'ElasticNet': {elasticnet1:.3f}\tF1: {Predm:.3f}    f2:{(Pprom + Predm + elasticnet1)/3:.3f}    f3:{(Pprom + Predm)/2:.3f}")
        print(" -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- ")
        Pp=(Pprom*(1-Prede)+Pprom*(1-ye2))/2
        nuevos, errores, textoN, textoE = calcular_nuevos_y_errores(y, Sum14, Pprom, Nn)
        c=[]
        i=[]
        
        for k, v in y.items():  
            c.append(str(k))
            i.append(str(v))   
        
        linea1 = "\t".join(c)
        linea2 = "\t".join(i)
        #print(f"\033[93m{linea1}\033[0m")
        print(f"\033[96m{linea2}\033[0m")

        linea3 = "\t".join(textoN)
        linea4 = "\t".join(textoE)
        print(f"\033[91m{linea3}\033[0m")
        print(f"\033[97m{linea4}\033[0m")
        minY = min(nuevos.values())
        maxY = max(nuevos.values())
                
        if Pp<minY:
            Pp=minY*1.004
            
        if Pp>maxY:
            Pp=maxY*0.996
        
        #nuevos, errores, texto,textoE = calcular_nuevos_y_errores(y, Sum14, Pp, Nn)
        #linea3 = "\t".join(textoN)
        #linea4 = "\t".join(textoE)
        #print(f"\033[91m{linea3}\033[0m")
        #print(f"\033[97m{linea4}\033[0m")
        #print("--------       -----------       ---------        --------")
        #Prom_Gral = solicitar_nuevo_prom(minY, maxY)
        #nuevos, errores, texto,textoE = calcular_nuevos_y_errores(y, Sum14, Prom_Gral, Nn)
        #linea = "\t".join(textoN)
        #linea1 = "\t".join(textoE)
        #print(f"\033[92m{linea}\033[0m")
        #print(f"\033[95m{linea1}\033[0m")

        return  errores, nuevos, Sum14

    else:
        print(f"Las listas no cumplen las condiciones para {titulo}.")
        return None, None, None


def escalar_dic(d, escalar):
    return {k: v * escalar for k, v in d.items()}


def procesar_Histogramas(titulo: str, Zz:int, Pos: int, Nn, h_data, f_data: dict, *proc_args):
    Y=len(h_data)
    default = ({i: 0 for i in range(1, 10)}, None, None)
        
    if Y > 120:
        Error_val, Nuevo_valor, Sum14 = procesar_regresion_Histo(titulo, Zz, Pos, h_data, f_data, Nn, *proc_args)
        # 2) Ordenar frecuencias y errores
        if Sum14 !=0:
            caidas_ordenadas = ordenar_por_valor(f_data, ascendente=True)
            prom_ordenados   = ordenar_por_valor(Error_val, ascendente=True)
            

            llave_min = min(prom_ordenados, key=lambda k: abs(prom_ordenados[k])) #min(prom_ordenados, key=prom_ordenados.get)
            #valor_min = Error_val[llave_min]
            # 4) Imprimir tablas
            imprimir_tabla("Caídas", caidas_ordenadas, es_decimal=False, highlight_key=llave_min) 
            imprimir_tabla(f"Promedio {titulo}", prom_ordenados, es_decimal=True)
            return Error_val
    else:
        print("No hay suficentes datos para evaluar")
        return default


def Llamada_Histo(columna, F_d):
    lispos, liss=recency_pos(columna)
    print("\x1b[38;5;71m---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---\x1b[0m")
    Pr1, _, _, _, _ = procesar_e_imprimir_regresion("Posiciones", 0, liss, 2, 1, 6)
    print("\x1b[38;5;71m---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---\x1b[0m")
    Pr2, _, _, _, _ = procesar_e_imprimir_regresion("Posiciones", 4, liss, 2, 1, 6)
    print("\x1b[38;5;71m---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---\x1b[0m")
    Pr3, _, _, _, _ = procesar_e_imprimir_regresion("Posiciones", 5, liss, 2, 1, 6)
    print("\x1b[38;5;71m---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---\x1b[0m")
    print("")
    z=Sumar_listas(Pr1, Pr2, Pr3)
    print(z)
    fun_promedios(liss, 0)
    Histog=obtener_historial_caidas(columna)
    Sig_Histo=obtener_siguiente_caidas(columna)
    
    H2=Histo2_con_map(Histog, 2)
    Sig_H2=Siguientes_lista(H2)
    F_datos_2 = {k: (v - 1) // 2 + 1 for k, v in F_d.items()}

    H3 = Histo2_con_map(Histog, 3)
    F_datos_3 = {k: (v - 1) // 3 + 1 for k, v in F_d.items()}
    print(aviso_ansi("Histogramas de 1/3 :", (118, 5, 30), (240, 220, 90) ))
    #Zonas_Histos(H3, 2)
    
    #yY=procesar_lista_tres(H3, 1, 0)
    #print_colored_stats(yY, 0, Forma=2)
    
    print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
    a1=procesar_Histogramas("Histograma 1/3", 0, 1, 19, H3, F_datos_3, 4, 2, 6)
    print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
    a2=procesar_Histogramas("Histograma 1/3", 3, 1, 18, H3, F_datos_3, 4, 2, 6)
    print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
    a3=procesar_Histogramas("Histograma 1/3", 5, 1, 17, H3, F_datos_3, 4, 2, 6)
    a=escalar_dic(Sumar_diccionarios(a1, a2, a3), 0.2)
    print(" ".join(f"{v:.3f} " for k, v in a.items()))
    print("\x1b[38;5;24m=============================================================================================================\x1b[0m")
    print(aviso_ansi("Histogramas de 1/2 :", (118, 5, 30), (240, 220, 90) ))
    #Zonas_Histos(H2, 2)
    #yY=procesar_lista_tres(H2, 1, 0)
    #print_colored_stats(yY, 0, Forma=2)
    print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
    b1=procesar_Histogramas("Histograma 1/2", 0, 1, 19, H2, F_datos_2, 5, 3, 7)
    print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
    b2=procesar_Histogramas("Histograma 1/2", 3, 1, 18, H2, F_datos_2, 5, 3, 7)
    print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
    b3=procesar_Histogramas("Histograma 1/2", 5, 1, 17, H2, F_datos_2, 5, 3, 7)
    b=escalar_dic(Sumar_diccionarios(b1, b2, b3), 0.2)
    print(" ".join(f"{v:.3f} " for k, v in b.items()))

    c={}
    if len(Sig_Histo)> 330:
        print(aviso_ansi("Siguientes Histogramas :", (118, 5, 30), (240, 220, 90) ))
        c=procesar_Histogramas("Siguiente Histog", 0, 2, 18, Sig_Histo, F_d, 6, 4, 8)
        print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
        print("")
    
    d={}
    if len(Sig_H2)> 330:
        print(aviso_ansi("Siguientes Histogramas de 2 :", (118, 5, 30), (240, 220, 90) ))
        d=procesar_Histogramas("Siguiente Histog 1/2 ", 0, 1, 18, Sig_H2, F_datos_2, 5, 3, 7)
        print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
        print("")
    
    print("\x1b[38;5;71m=============================================================================================================\x1b[0m")
    print(aviso_ansi("Histogramas completos :", (118, 5, 30), (240, 220, 90) ))
    Zonas_Histos(Histog, 0)
    #yY=procesar_lista_tres(Histog, 1, 0)
    #print_colored_stats(yY, 0, Forma=2)
    print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
    e1=procesar_Histogramas("Histograma con 30", 0, 2, 19, Histog, F_d, 7, 4, 11)
    print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
    e2=procesar_Histogramas("Histograma con 30", 3, 2, 18, Histog, F_d, 7, 4, 11)
    print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
    e3=procesar_Histogramas("Histograma con 30", 5, 2, 17, Histog, F_d, 7, 4, 11)
    e=escalar_dic(Sumar_diccionarios(e1, e2, e3), 0.2)
    print(" ".join(f"{v:.3f} " for k, v in e.items()))
    print("\x1b[38;5;71m=============================================================================================================\x1b[0m")
    print(aviso_ansi("Histogramas completos de 40 :", (118, 5, 30), (240, 220, 90) ))
    f=procesar_Histogramas("Histograma con 40", 0, 2, 22, Histog, F_d, 7, 4, 11)
    print()
    
    Suma_Total=escalar_dic(Sumar_diccionarios(escalar_dic(a, 2.5), escalar_dic(b, 2.5), c, d, escalar_dic(e, 4), escalar_dic(f, 2), 1),0.1)
    Ordenados=ordenar_por_valor(Suma_Total, ascendente=True)
    imprimir_tabla("Probabilidad Caidas Histograma ", Ordenados, es_decimal=True)
    print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
    print("")
    mean_p, last_p, errors, error_abs_fd, error_rel_fd = analyze_frecuencias(
        Histog, F_d, max_val=40, n_segments=4, last_n=45)

    mean_p2, last_p2, errors2, error_abs_fd2, error_rel_fd2 = analyze_frecuencias(
        H2, F_datos_2, max_val=20, n_segments=4, last_n=30)
    
    abs_fd=Sumar_diccionarios(error_abs_fd, error_abs_fd, error_abs_fd2, 1)
    imprimir_tabla("Probabilidad Semanas Caidas  ", abs_fd, es_decimal=True)
    print("                            *************          ***********")

    Bayes_Histo=porcentaje_coincidencias(F_d, Histog)
    PromOrdenados=ordenar_por_valor(Bayes_Histo, ascendente=False)
    imprimir_tabla("PORCENTAJE caidas Semanas", PromOrdenados, es_decimal=True)
    return Ordenados


def Llamar_Numeros1(columna, Lista, F_d):
    Sig_numeros = obtener_siguiente_numero(columna)
    Ultima_Jerarquia=ultima_jerarquia(columna)
    jerarquias, Posic = calcular_jerarquias(columna)
    claves_ordenadas = sorted(Ultima_Jerarquia.keys(), key=lambda k: (Ultima_Jerarquia[k], -F_d[k]))
    ranking_dict = {rank: clave for rank, clave in enumerate(claves_ordenadas[:10])}
    
    Pb_Num = analizar_siguientes_numeros_para_probabilidades(Lista)
    Jer_orde=ordenar_por_valor(Pb_Num, ascendente=False)
    imprimir_tabla("Probabilidades Numeros Bayes ", Jer_orde, es_decimal=True)
    print()

    Pb_Sig = analizar_siguientes_numeros_para_probabilidades(Sig_numeros)
    Jer_orde=ordenar_por_valor(Pb_Sig, ascendente=False)
    imprimir_tabla("Probabilidades Siguientes Bayes ", Jer_orde, es_decimal=True)
    print()
    
    Pb_jerarquia = calcular_probabilidades_desde_historial(claves_ordenadas,Posic)
    Jer_orde=ordenar_por_valor(Pb_jerarquia, ascendente=False)
    imprimir_tabla("Probabilidades Jerarquia Bayes ", Jer_orde, es_decimal=True)
    print()
    print("\tJerarquias\tNumeros\t\tSiguientes\tTotal")
    Pb_por_Numero = {}
    for num in Pb_Sig:
        # Se asume que la clave también existe en Probab_Jerar_bayes
        Pb_por_Numero[num] = (Pb_Sig[num] + 2*Pb_Num[num]+6*Pb_jerarquia[num]) / 9
        if num % 2==0:
            print("\033[33m"+f"{num}\t{Pb_jerarquia[num]:.4f}\t\t{Pb_Num[num]:.4f}\t\t{Pb_Sig[num]:.4f}\t\t{Pb_por_Numero[num]:.4f}" + "\033[0m")
        else:
            print("\033[34m"+f"{num}\t{Pb_jerarquia[num]:.4f}\t\t{Pb_Num[num]:.4f}\t\t{Pb_Sig[num]:.4f}\t\t{Pb_por_Numero[num]:.4f}" + "\033[0m")
    print()
    #imprimir_tabla("Errores Prom. Ordenados Numeros Siguientes Jerarquia ", ErrorOrdenado, es_decimal=True)
    print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
    print("")
    ProNUm=ordenar_por_valor(Pb_por_Numero, ascendente=False)
    imprimir_tabla("Prob. Bayes Ordenadas Numeros Siguientes Jerarquia  ", ProNUm, es_decimal=True)
    print()

def multiplicar_lista(datos, escalar):
    return [x * escalar for x in datos]


def Sumar_listas(*listas: List[float]) -> List[float]:
    
    if not listas:
        return []

    # Verificar que todas las listas tengan la misma longitud
    longitud = len(listas[0])
    for idx, lst in enumerate(listas, start=1):
        if len(lst) != longitud:
            raise ValueError(
                f"Lista #{idx} tiene longitud {len(lst)}, esperaba {longitud}"
            )

    # Sumar elemento a elemento
    return [sum(vals) for vals in zip(*listas)]
    


def Prom_Num2(Nume):
    
    print("\x1b[38;5;71m---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---\x1b[0m")
    Pr_Num21, _, _, _, _ = procesar_e_imprimir_regresion("Numeros Medios", 0, Nume, 2, 1, 6)
    print("\x1b[38;5;71m---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---\x1b[0m")
    Pr_Num22, _, _, _, _ = procesar_e_imprimir_regresion("Numeros Medios", 4, Nume, 2, 1, 6)
    print("\x1b[38;5;71m---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---\x1b[0m")
    Pr_Num23, _, _, _, _ = procesar_e_imprimir_regresion("Numeros Medios", 5, Nume, 2, 1, 6)
    print("\x1b[38;5;71m---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---\x1b[0m")
    print("")
    y=Sumar_listas(Pr_Num21, Pr_Num22, Pr_Num23)

    return y
    

def Prom_numeros(Nume):
    #yY=procesar_lista_tres(Nume, 0, 0)
    #print_colored_stats(yY, 0, Forma=2)
    print("\x1b[38;5;71m---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---\x1b[0m")
    Pr_Num1, _, _, _, _ = procesar_e_imprimir_regresion("Numeros", 0, Nume, 0)
    print("\x1b[38;5;71m---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---\x1b[0m")
    Pr_Num2, _, _, _, _ = procesar_e_imprimir_regresion("Numeros", 4, Nume, 0)
    print("\x1b[38;5;71m---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---\x1b[0m")
    Pr_Num3, _, _, _, _ = procesar_e_imprimir_regresion("Numeros", 5, Nume, 0)
    print("\x1b[38;5;71m---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---\x1b[0m")
    y=Sumar_listas(Pr_Num1, Pr_Num2, Pr_Num3)
    return y



def Llamar_Numeros(columna, Lista, F_d):
    # Códigos ANSI para rojo y reset
    RED   = "\033[31m"
    RESET = "\033[0m"

    Nume2=Lista2_con_map(Lista)
    Sig_numeros = obtener_siguiente_numero(columna)
    #print(Sig_numeros)
    Ultima_Jerarquia=ultima_jerarquia(columna)
    jerarquias, Posic = calcular_jerarquias(columna)
    #print(aviso_ansi(f"\nTotal Numeros Siguientes : {len(Sig_numeros)}", (118, 5, 30), (240, 220, 100) ))
    
    Ultima_J_Sig=ultima_jerarquia_Lista(Sig_numeros)
    Jer_Sig=ordenar_por_valor(Ultima_J_Sig, ascendente=False)
    
    #Procesamiento probabilidades ultimas
    #DiccNu = Dicc_probabilidad_ordenado(Lista)
    #imprimir_tabla("Probabilidades por Jerarquia caidas en 20", DiccNu, es_decimal=True)
    
    claves_ordenadas = sorted(Ultima_Jerarquia.keys(), key=lambda k: (Ultima_Jerarquia[k], -F_d[k]))
    print(aviso_ansi("\nOrden de jerarquías :", (118, 5, 30), (240, 220, 90) ))
    print("Id\t\t" + "\t".join(str(k) for k in claves_ordenadas))
    print("Repet\t\t" + "\t".join(str(Ultima_Jerarquia[k]) for k in claves_ordenadas))
    print("Apar\t\t" + "\t".join(str(F_d[k]) for k in claves_ordenadas))
    print("\x1b[38;5;71m======================================================================================\x1b[0m")
    print("")
    print(aviso_ansi("Empezando con Números :", (118, 5, 30), (240, 220, 90) ))
    yy=Zonas_Numeros(Lista)
    print(" ".join(f"{v:.3f} " for v in yy))

    print("                 Numeros Medios ")
    fun_promedios(Nume2, 0)
    y=Prom_Num2(Nume2)
    Prom_NUme2=multiplicar_lista(y, 0.3)
    print(" ".join(f"{v:.4f} " for v in Prom_NUme2))

    print("\n                Numeros Completos 0-9 ")
    fun_promedios(Lista, 0)
    yz=Prom_numeros(columna)
    Pr_Num=multiplicar_lista(yz, 0.3)
    print(" ".join(f"{v:.4f} " for  v in Pr_Num))
    
    Resu=[]
    for i, valor in enumerate(Pr_Num):
        Resu.append((2*valor + Prom_NUme2[i // 2]) / 3)

    #print("Siguientes de Numeros Completos ")    
    #Zonas_Numeros(Sig_numeros)
    #Pr_Sig, _, _,  _ = procesar_e_imprimir_regresion("Siguientes", 0, Sig_numeros, 2)
    #Pr_Sig, _, _,  _ = procesar_e_imprimir_regresion("Siguientes", 4, Sig_numeros, 2)
    
    print("Frecuencias en 50 jugadas ")

    fun_promedios(Posic, 0)
    fg=zones_by_freq(columna)
    #print("\x1b[38;5;71m---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---\x1b[0m")
    Pr_Pos_val, Pr_Pos_err, Sum14, PromGral, xX = procesar_e_imprimir_regresion("Jerarquía", 0, Posic, 0, 1, 11)
    print("\x1b[38;5;71m---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---\x1b[0m")
    Pr_Pos_val, Pr_Pos_err, Sum14, PromGral, xX = procesar_e_imprimir_regresion("Jerarquía", 4, Posic, 0, 1, 11)
    print("\x1b[38;5;71m---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---\x1b[0m")
    Pr_Pos_val, Pr_Pos_err, Sum14, PromGral, xX = procesar_e_imprimir_regresion("Jerarquía", 5, Posic, 0, 1, 11)
    
    ranking_dict = {rank: clave for rank, clave in enumerate(claves_ordenadas[:10])}
    nuevos_valores_dict = {}
    errores_dict = {}
    
    for rank, clave in ranking_dict.items():
        nuevo_valor = (Sum14 + rank) / xX
        error = (nuevo_valor - PromGral) / PromGral
        if error < 0:
            error *= -0.999
        nuevos_valores_dict[clave] = nuevo_valor
        errores_dict[clave] = error
    print()
    sorted_keys = sorted(ranking_dict.values())  # esto da [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Pr_Pos_err_ordered = [errores_dict[k] for k in sorted_keys]
    
    min_jer = min(errores_dict.values())
    min_num = min(Resu)
    
    #min_sig = min(Pr_Sig)
    print("\tJerarquías\tNumeros\t\tSiguientes\tError Num")
    print("*******************************************************************")
    
    ErrorNUm={}
    for k in sorted_keys:
        jer = errores_dict[k]
        num = Resu[k]
        #sig = Pr_Sig[k]
        ErrorNUm[k] = ( abs(3*num) + abs(jer)) / 4 
        
        # Formatea cada celda, poniendo rojo si coincide con el mínimo
        s_jer = f"{jer:.3f}"
        if jer == min_jer:
            s_jer = f"{RED}{s_jer}{RESET}"
        
        s_num = f"{num:.3f}"
        if num == min_num:
            s_num = f"{RED}{s_num}{RESET}"
        
        #s_sig = f"{sig:.3f}"
        #if sig == min_sig:
        #    s_sig = f"{RED}{s_sig}{RESET}"
             
        print(f"{k}\t{s_jer}\t\t{s_num}\t\t{ErrorNUm[k]:.3f}")
    
    ErrorOrdenado=ordenar_por_valor(ErrorNUm, ascendente=True)
    #print()
    imprimir_tabla("\nErrores Promedios Numeros ", ErrorOrdenado, es_decimal=True)
    
    # Imprimir resultados ajustados según el ranking (en orden de ranking)
    print("\nResultados ajustados de 'Jerarquía' reordenados mediante ranking_dict:")
    for rank in sorted(ranking_dict.keys()):
        clave = ranking_dict[rank]
        print(f"Id {rank}:\tN. {clave},\t Prom: {nuevos_valores_dict[clave]:.3f},\t Er: {errores_dict[clave]:.3f}")
    print()

    #Prior1=calcular_alpha_prior(columna)
    #Prior=ordenar_por_valor(Prior1, ascendente=False)
    #PriorSig=calcular_alpha_prior_Lista(Sig_numeros)
    return ErrorOrdenado


def fun_promedios(lista, modo):    #si 1, mayores o iguales a 0, si 0 puede tener numeros negativos
    RESET     = "\033[0m"
    FG_RED    = "\033[31m"
    FG_BLUE   = "\033[34m"
    BG_GREEN  = "\033[107m"
    FG_DEFAULT= "\033[39m"

    ultimos=lista[-12:]
    salida = ""
    for i, val in enumerate(ultimos):
        text = str(val)

        if modo == 1:
            # fondo verde + texto por defecto, cubre: [número][espacio]
            bloque = f"{BG_GREEN}{FG_DEFAULT}{text} {RESET}"
        elif modo == 0:
            text = f"{val:.0f}"
            bloque = text
        else:
            # solo cambio de color de texto según signo, sin fondo
            text = f"{val:.3f}"
            if val < 0:
                bloque = f"{FG_RED}{text}{RESET}"
            elif val > 0:
                bloque = f"{FG_BLUE}{text}{RESET}"
            else:
                bloque = text

        # Después del bloque coloreado, añadimos un espacio limpio
        # para separar de la siguiente cifra (solo si no es el último)
        if i < len(ultimos) - 1:
            salida += bloque + "  "
        else:
            salida += bloque  # último elemento, no agregamos espacio extra

    print(salida)

    PromT=sum(lista)/len(lista)
    Prom15 = sum(lista[-12:])/len(lista[-12:])
    Prom10 = sum(lista[-12:-6])/len(lista[-12:-6])
    Prom6 = sum(lista[-5:])/len(lista[-5:])
    print(colorear(PromT, "P To "), end="\t")
    print(colorear(Prom15, "P. 12 "), end="\t")
    print(colorear(Prom10, "P. M "), end="\t")
    print(colorear(Prom6, "P. 5 "))



def Llamar_Caidas(Colu):
    Columna=pd.Series(Colu)
    #ff=zones(Columna)
    print("Posibilidades de Siguiente Puesto Histog")
    Pos_Histo=calcular_jerarquia_histo(Colu)
    print(len(Pos_Histo))
    Pb_His1 = analizar_siguientes_numeros_para_probabilidades(Pos_Histo)
    #print("\x1b[38;5;71m---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---\x1b[0m")
    Elo=procesar_lista(Pos_Histo, 1, 1)
    print_colored_stats(Elo, 0, 0)
    print("-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- ")
    
    fun_promedios(Pos_Histo, 0)
    Pb_His, _, _, _, _ = procesar_e_imprimir_regresion("Orden Caida", 0, Pos_Histo, 0, 1, 11)
    Pb_His, _, _, _, _ = procesar_e_imprimir_regresion("Orden Caida", 4, Pos_Histo, 0, 1, 11)
    Pb_His, _, _, _, _ = procesar_e_imprimir_regresion("Orden Caida", 5, Pos_Histo, 0, 1, 11)
    print("\x1b[38;5;71m---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---\x1b[0m")
    errores = {}
    for idx, err in enumerate(Pb_His, start=1):
        errores[idx] = err
    
    return errores


def LLama_Sigui(Colu):
    Sig_numeros = obtener_siguiente_numero(Colu)
    Ss=pd.Series(Sig_numeros)
    SHis=obtener_historial_caidas(Ss)
    F_dsig=Semanas(Ss)
    print(len(SHis))
    if len(SHis)> 100:
        Zonas_Histos(Sig_numeros, 0)
        c=procesar_Histogramas("Histograma de Siguientes", 2, 2, 30, Ss, F_dsig, 7, 4, 11)
    return SHis


def Lotery(Re, Res):
    banner = [
        # L        OOO      TTTTT    EEEEE   RRRR    Y   Y
        "\n",
        " \t\t\t**          *******      ********     *******     *******      **      ** ",
        " \t\t\t**         **     **        **        **          **   **       **    **  ",
        " \t\t\t**         **     **        **        **          **   **         ****   ",
        " \t\t\t**         **     **        **        ******      ******           **   ",
        " \t\t\t**         **     **        **        **          **   **          **   ",
        " \t\t\t**         **     **        **        **          **    **         **   ",
        " \t\t\t*******     *******         **        *******     **     **        **   ",
        "\n",
    ]
    for line in banner:
        print(f"{Re}{line}{Res}")


def main(file_path):
    # Códigos ANSI para rojo y reset
    RED   = "\033[31m"
    RESET = "\033[0m"
    Lotery(RED, RESET)
    Inic = datetime.datetime.now().strftime("%H:%M:%S")
    
    Numeros = leer_datos_excel(file_path)
    Nume=Numeros.tolist()
    print("                          ***   Recencia de Semanas   ***")
    F_datos=Semanas(Numeros)
    print_recencia(F_datos)
    print("")
    E_Num=Llamar_Numeros(Numeros, Nume, F_datos)
    #Llamar_Numeros1(Numeros, Nume, F_datos)
    print(aviso_ansi("\nTERMINAMOS NUMEROS...", (220, 110, 10), (120, 220, 200) ))
    
    print("  --  Posicion de Caidas  --  ")
    print("")
    Cai=Llamar_Caidas(Nume)
    print("\x1b[38;5;18m=============================================================================================================\x1b[0m")
    print("       -- -- Probabilidades Recencia (Histograma) --  -- ")
    print("")
    E_Histo=Llamada_Histo(Numeros, F_datos) 
    print(aviso_ansi("\nTERMINAMOS HISTOGRAMAS \n", (220, 110, 10), (120, 220, 200) ))
    print("  --  Probabilidades recencia números  siguientes --  ")
    print("")
    #HIs_Sig=LLama_Sigui(Numeros)
    print("  --  Pseudo probabilidad jerárquica  --  ")
    print("")
    Prior1=calcular_alpha_prior(Numeros)
    Prior=ordenar_por_valor(Prior1, ascendente=False)

    Probab_mayor = aplicar_regresion_logistica_mayor_menor(Numeros)
    if Probab_mayor is not None:
        print(f"\nProbabilidad (Reg. Logística) de que el siguiente número sea mayor que 4: {Probab_mayor:.4f}")

    Probab_par = aplicar_regresion_logistica_par_impar(Numeros)
    if Probab_par is not None:
        print(f"Probabilidad (Reg. Logística) de que el siguiente número sea par: {Probab_par:.4f}")
    print()

    imprimir_tabla("Probabilidad de Numeros ", E_Num, es_decimal=True)
    print("")
    imprimir_tabla("Posicion de Caida ", Cai, es_decimal=True)
    imprimir_tabla("Probabilidad de Histograma ", E_Histo, es_decimal=True)
    print(aviso_ansi(f"\nTERMINAMOS AQUI : ", (118, 5, 30), (240, 220, 100) ))
    print(Inic)
    print(datetime.datetime.now().strftime("%H:%M:%S"))
    

if __name__ == "__main__":
    print("Hello World")
    file_path = 'D:/loter.xlsx'
    main(file_path)




    #imprimir_Nmedios(Pr_Num2)
    #print()
    #imprimir_tabla("Errores Prom. Ordenados Numeros Siguientes Jerarquia ", ErrorOrdenado, es_decimal=True)
    #print()
    #ProNUm=ordenar_por_valor(Pb_por_Numero, ascendente=False)
    #imprimir_tabla("Prob. Bayes Ordenadas Numeros Siguientes Jerarquia  ", ProNUm, es_decimal=True)
    #print()

       
    #DiccSig = Dicc_probabilidad_ordenado(Sig_numeros)
    #DiccJer = Dicc_probabilidad_ordenado(Posic,(15, 20, 25), intervalo_inicial=1, intervalo_final=10, cantidad=30)
    #imprimir_tabla("Probabilidad Numeros con 15, 20 y 25 ", DiccNu, es_decimal=True)
    #imprimir_tabla("Probabilidad Siguientes Numeros con 15, 20 y 25 ", DiccSig, es_decimal=True)
    #DicJ=remapear_por_posicion(claves_ordenadas, DiccJer)
    #imprimir_tabla("Probabilidad Jerarquia ", DicJ, es_decimal=True)
    #DicT=sumar_diccionarios(DiccSig, DiccNu, DiccNu, DiccNu, DiccNu, DicJ, Divisor=6)
    #DicT=sumar_diccionarios(DiccSig, DiccNu, DiccNu, DiccNu, DiccNu, Divisor=5)


    #DiccHS15={}
    #DiccH15=Probabilidad_Caidas(Histog,1,41,60,40, True)
    #imprimir_tabla("Probabilidad Numeros con 15 ", DiccS15, es_decimal=True)
    #DiccH20={}
    #DiccH20=Probabilidad_Caidas(Histog,1,41,50,30, True)
    #imprimir_tabla("Probabilidad Numeros con 20 ", DiccS20, es_decimal=True)
    #DiccH25={}
    #DiccH25=Probabilidad_Caidas(Histog,1,41,40,25, True)
    #imprimir_tabla("Probabilidad Numeros con 25 ", DiccS25, es_decimal=True)
    #SDicH=sumar_diccionarios(DiccH15, DiccH20, DiccH25)
    #DicHi=ordenar_por_valor(SDicH, ascendente=False)
    #imprimir_tabla("Probabilidad Numeros con 15, 20 y 25 ", DicHi, es_decimal=True)


    #modelo_ajustado = validate_residuals(lista_30, lista_15, lista_6, lista_sig)
    #graficar_residuos(lista_30, lista_15, lista_6, lista_sig)
    #reporte = reporte_regresiones(lista_30, lista_15, lista_6, lista_sig)
 
    #print("Procesando regresión bayesiana con PyMC...")
    #resultado_bayes = prediccion_bayesiana(lista_30, lista_15, lista_6, lista_sig)
    #print("Predicción media (modelo bayesiano):", resultado_bayes["prediccion_media"])
    #print("Intervalo 98% de credibilidad:", resultado_bayes["int_95"])






    #def analizar_collected(
#    collected: List[float],
#    last: float,
#    tipo: int,
#    Pos: int,
#    y: int
#) -> Dict[str, Optional[float]]:
    #count = len(collected)
    #mean_total = sum(collected) / count if count else None

    #if tipo == 0:
    #    pos = [x for x in collected if x >= 0]
    #    neg = [x for x in collected if x < 0]
    #else:
    #    pos = [x for x in collected if x >= last]
    #    neg = [x for x in collected if x < last]

    #mean_pos = sum(pos) / len(pos) if pos else None
    #mean_neg = sum(neg) / len(neg) if neg else None

    # 6. Top-4 valores más comunes
    #c = Counter(collected)
    #for idx, (valor, cnt) in enumerate(c.most_common(4), start=1):
    #    print(f"{idx}. {valor:.4f} → {cnt} ", end="\t")
    #print(f"\t\t\t\tj:{j}  lo {lo1:.3f}    hi {hi1:.3f}")
    # 7. Caídas por rango (suponiendo que devuelve List[(rango, cnt)], _)
    #x, y = Caidas_por_rango(collected, last, Pos)
    #num = sum(((li + ls) / 2) * cnt for ((li, ls), cnt) in x)
    #den = sum(cnt for _, cnt in x)
    #prom_pon = num / den if den else 0
    #for (li, ls), cnt in x:
    #    print(f"{li:.3f}-{ls:.3f}→ {cnt}", end="\t")
    #print(f"\tProm pond: {prom_pon:.3f}")

    # 8. Medias finales y últimos datos
    #mean_pos = sum(pos) / len(pos) if pos else None
    #mean_neg = sum(neg) / len(neg) if neg else None
    #ultimos = data[-y:]
    #sig_2 = collected[-y:]
    # (‘data’ original no está aquí; si la necesitas, pásala también)
    #return {
    #    "N": count,
    #    "Ptot": mean_total,
    #    "Ppos": mean_pos,
    #    "Tpos": len(pos),
    #    "Pneg": mean_neg,
    #    "Tneg": len(neg),
    #    "Ult": last,
    #    "Ant":ultimos,
    #    "Sig":sig_2
    #}