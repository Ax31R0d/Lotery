import pandas as pd
import numpy as np
import pymc as pm
from arviz import summary, plot_trace
import statsmodels.api as sm
from collections import Counter, defaultdict
from sklearn import linear_model, ensemble, svm, model_selection, metrics
from typing import List, Dict, Tuple, Optional
import pdb
import matplotlib.pyplot as plt
import scipy.stats as stats


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

    #    y a partir de ahí buscamos p(c_v)
    probs = {}
    for v in dominio:
        c_v      = ultima_ventana.count(v)
        # hist_counts.get(c_v, 0) → cuántas ventanas históricas tuvieron ese conteo
        p_v      = hist_counts.get(c_v, 0) / total_ventanas
        probs[v] = round(p_v, 3)
    print(probs)

    return   probs #, conteos, p_global[-last_n:],


def sumar_diccionarios(*dicts, Divisor=10):
    """
    Recibe N diccionarios con las mismas llaves y devuelve uno
    donde cada llave = suma de sus valores en los dicts recibidos.
    """
    # Asumo que hay al menos un dict y todos comparten exactamente las mismas llaves.
    claves = dicts[0].keys()
    return {k: sum(d[k]/Divisor for d in dicts) for k in claves}


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
    errors_pct_all = (preds_all - y) / (y + 1e-8)
    mpe_all = np.mean(errors_pct_all)
    preds_prev = preds_all[-20:]
    reales_prev = y[-20:]
	# Cálculo del error porcentual CON signo
    errores_pct20 = (preds_prev - reales_prev) / reales_prev
    mpe_last10  = np.mean(errores_pct20)

    return best_svr, cv_score, best_params, mpe_all, errores_pct20, errors_pct_all


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
        if jugadas > 40:
            jugadas = 40 if jugadas % 2 == 0 else 39
        caidas_columna.append(jugadas)
        ultimas_posiciones[valor] = i
    return caidas_columna


def obtener_siguiente_caidas(columnas):
    siguiente_caidas = []
    caidas = obtener_historial_caidas(columnas)
    ultima_caida = caidas[-1]
    for i in range(len(caidas) - 1):
        if caidas[i] == ultima_caida:
            siguiente = min(caidas[i + 1], 40)
            siguiente_caidas.append(siguiente)
    return siguiente_caidas


def Semanas(columna):
    grupo = columna.tail(40)
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
                distancia = 40
            else :
                distancia = 39
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


def calcular_promedios_y_errores(columna, Pos):
    promedio_general = np.mean(columna)
    # Calcular lista_30 y errores_30
    lista_40 = [np.mean(columna[i - 40:i]) for i in range(45, len(columna))]
    errores_40 = [(p - promedio_general) / promedio_general for p in lista_40]
    
    # Calcular lista_15 y errores_15 con índice correcto
    lista_15 = [np.mean(columna[i - 8:i]) for i in range(45, len(columna))]
    errores_15 = [(p - promedio_general) / promedio_general for p in lista_15]
    
    # Calcular lista_6 y errores_6
    lista_4 = [np.mean(columna[i - 3:i]) for i in range(45, len(columna))]
    errores_4 = [(p - promedio_general) / promedio_general for p in lista_4]

    lista_20 = [np.mean(columna[i - 20:i]) for i in range(45, len(columna)+1)]
    l20=sum(columna[-19:])
       
    # Calcular lista_sig
    lista_sig = [np.mean(columna[i - 19:i + 1]) for i in range(45, len(columna) + 1)]
    lista_sig.pop()
    Ele=procesar_lista(lista_sig, 1, Pos, 1, Forma=1) 
    print_colored_stats(Ele, 0, Forma=0)
    lista_14=sum(columna[-19:])  

    return errores_40, errores_15, errores_4, lista_sig, lista_14, l20, promedio_general


def promedios_y_errores_lista(data, Pos, yy):
    n = len(data)
    if n == 0:
        raise ValueError("La lista no puede estar vacía.")
    
    # 1) Promedio general
    promedio_general = sum(data) / n
    # 2) Ventanas deslizantes y errores relativos Para i en [40 .. n-1], media de los últimos k elementos

    def medias_ventana(k):
        return [ sum(data[i-k: i]) / k for i in range(45, n) ]

    lista_30 = medias_ventana(40)
    lista_15 = medias_ventana(15)
    lista_6  = medias_ventana(4)
    
    def errores(lista_medias):
        return [ (m - promedio_general) / promedio_general for m in lista_medias ]

    errores_30 = errores(lista_30)
    errores_15 = errores(lista_15)
    errores_6  = errores(lista_6)
    
    # 3) lista_sig: media de ventana de tamaño 15, pero alineada como en tu código original
    #    Para i en [40 .. n], media de data[i-14 : i+1], luego descartamos el último
    lista_4 = [ sum(data[i-19: i+1]) / 20 for i in range(45, n+1) ]
    lista_4.pop()
    lista_sig = [ sum(data[i-19: i+1]) / 20 for i in range(45, n+1) ]
    lista_sig.pop()                  
    # 4) suma de los últimos 14 valores
    suma_14 = sum(data[-19:])
    l5=sum(data[-19:])
    Bloque= ultimos_promedios_list(yy, l5)
    bloque_fmt = [f"{x:.2f}" for x in Bloque]
    all_values = [f"\033[32m{v}\033[0m" for v in bloque_fmt] + ["\t"]
    print(*all_values, sep="  ")
    
    if n>100:
        Elp=procesar_lista(lista_sig, 1, Pos, Forma=1) 
        print_colored_stats(Elp, 0, Forma=0)
    
    return errores_30, errores_15, errores_6, lista_sig, suma_14, promedio_general,l5


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


def aplicar_regresion_elasticnet(lista_30, lista_15, lista_6, lista_sig, alpha=0.6, l1_ratio=0.4, max_iter=3000, tol=0.00001,):
    if not all([lista_30, lista_15, lista_6, lista_sig]) or not len(lista_30) == len(lista_15) == len(lista_6) == len(lista_sig):
        print("Error en los datos para Elastic Net.")
        return None
    X = np.array([lista_30, lista_15, lista_6]).T
    y = np.array(lista_sig)
    modelo_enet = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol).fit(X, y)
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
    """ Aplica la regresión ponderada y extrae los coeficientes del Método 1. """
    
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
    """ Aplica la regresión ponderada y extrae los coeficientes del Método 2. """
    
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
    #Aplica la regresión robusta y extrae los coeficientes (se puede usar el Método 1 como referencia). 
    
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


def procesar_regresiones(datos):
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
    params_enet = aplicar_regresion_elasticnet(*datos_regresion)  # Suponiendo que esta función ya existe

    # Calcular predicciones
    resultados_m1 = predecir_con_regresion(params_wls1, params_enet, lista_30, lista_15, lista_6)
    resultados_m2 = predecir_con_regresion(params_wls2, params_enet, lista_30, lista_15, lista_6)
    resultados_rlm = predecir_con_regresion(params_rlm, params_enet, lista_30, lista_15, lista_6)
    
    return {
        "WLS_Metodo1": resultados_m1,
        "WLS_Metodo2": resultados_m2,
        "RLM": resultados_rlm
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


def ultimos_promedios_list(data: List[float], Va) -> List[float]: 
    return [(Va + v) / 20 for v in data.values()]


def ultimos_promedios_series(data: pd.Series) -> List[float]:
    if len(data) < 15:
        # no hay ni un solo promedio completo
        return pd.Series(dtype=float)

    medias = data.rolling(15).mean().dropna()
    return medias.iloc[-10:].tolist()

 
def procesar_e_imprimir_regresion(titulo, Pos, Lista, start=0,stop=10, medio=5, ante=2, fin=7):
    print(aviso_ansi("                      * - * - * - * - * - * - * - * - * - * - * - * - *", (44, 62, 80), (189, 195, 199)))
    print(aviso_ansi(f"Resultados para {titulo}:",(118, 5, 30), (240, 220, 90)))
    L_30, L_15, L_6, L_sig, Sum14, S5, PROM = calcular_promedios_y_errores(Lista, Pos)
    best_svr, cv_score, best_params,PromT,Err6, ETo = aplicar_svr(L_30, L_15, L_6, L_sig)
    #print("")
    Ele=procesar_lista(ETo, 0, 0)
    print_colored_stats(Ele, 0)
    Elp=procesar_lista(ETo, 1, 0, 0) 
    print_colored_stats(Elp, 1)
    print("Probando....")
    Ele=procesar_lista_dos(ETo, 0, 0)
    print_colored_stats(Ele, 1)
    
    Prom10 = Err6[-12:].mean()
    Prom6 = Err6[-4:].mean()
    print(colorear(PromT, "\tProm Error "), end="\t")
    print(colorear(Prom10, "P. 10 "), end="\t\t")
    print(colorear(Prom6, "P. 4 "))
    #print("\nParámetros para SVR:", best_params)
    print(f"\033[31mGeneral :\033[0m", end="\t")
    print(f"\033[1;31;47m{PROM:.3f}\033[0m", end="\t\t") 
    # Se toma el último valor de cada lista para formar una nueva observación.
    nuevo_dato = np.array([[L_30[-1], L_15[-1], L_6[-1]]])
    prediccion = best_svr.predict(nuevo_dato)
    Pprom=prediccion[0]
    datos_regresion = (L_30, L_15, L_6, L_sig)
    
    # Códigos ANSI
    BG_BLUE = '\033[46m'
    RED_ON_YELLOW = "\033[33;46m"
    RESET        = "\033[0m"
    # Verificar que todas las listas contengan datos, tengan la misma longitud y mayor a 50.
    if (L_30 and L_15 and L_6 and L_sig and len(L_30) == len(L_15) == len(L_6) == len(L_sig) and len(L_30) > 50):
        resultados = procesar_regresiones(datos_regresion)
        # Extraer el valor de ElasticNet (se asume que es consistente en todos los métodos)
        primer_metodo = next(iter(resultados))
        valor_elasticnet = resultados[primer_metodo]["ElasticNet"]
        print(f"\033[1;94;43mRegresion : {Pprom:.2f}\033[0m", end="\t")
        print(f"'ElasticNet': {valor_elasticnet:.2f}", end="\t")
    
        Bb=Pprom*(1+Ele['Ptot'])
        print(f"\033[1;34;43m{Bb:.2f}\033[0m", end="\t\t")

        resultados = [Pprom * factor for factor in (0.9, 0.95, 1, 1.05, 1.1)]
        #Unimos con dos espacios como separador y lo imprimimos en la misma línea
        print("  ".join(f"\033[31;45m{valor:.2f}{RESET}" for valor in resultados))

        print(
            f"\033[34m\t\tProm + {ante}: {((S5 + ante) / 20):.2f}\033[0m\t"
            f"\033[31mProm Medio: {((S5 + medio) / 20):.2f}\033[0m\t"
            f"\033[34mProm + {fin}: {((S5 + fin) / 20):.2f}\033[0m"
        )
        print(" -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --")
        nuevos_valores, errores_ajustados = imprimir_valores_y_errores(Sum14, Pprom,start,stop)
        #print("")
        minY=min(nuevos_valores)
        maxY=max(nuevos_valores)
        S14=(Sum14+medio)/20
        Prom_Gral = solicitar_nuevo_prom(minY, maxY)
        nuevos_valores, errores_ajustados = imprimir_valores_y_errores(Sum14, Prom_Gral, start, stop, col_val="\033[92m", col_err="\033[92m")
        print("")
        return  errores_ajustados, nuevos_valores, Sum14, Prom_Gral

    else:
        print(f"No hay datos suficientes o las listas no cumplen las condiciones para {titulo}.")
        return None, None, None, None


def procesar_lista_dos(
    data: List[float],
    tipo: int,
    Pos: int
) -> Dict[str, Optional[float]]:
    min_count = 40
    max_iters = 500
    last1, last2 = data[-1], data[-2]
    # 1. Validación de longitud mínima
    if len(data) < 100:
        return {"N": 0, "Ptot": 0, "Ppos": 0, "Pneg": 0}
    # 2. Configurar paso según Pos
    step = 0.0045 if Pos == 0 else 0.002
    lo1 = hi1 = last1
    lo2 = hi2 = last2
    collected: List[float] = []

    # 4. Búsqueda iterativa hasta reunir min_count datos
    for j in range(max_iters):
        # Expande ambas ventanas
        lo1, hi1 = lo1 - step, hi1 + step
        lo2, hi2 = lo2 - step, hi2 + step

        for i in range(1, len(data) - 1):
            anterior = data[i - 1]
            actual   = data[i]
            # Ambas ventanas deben cumplirse simultáneamente
            if lo1 < actual < hi1 and lo2 < anterior < hi2:
                # Añade el siguiente dato tras 'actual'
                collected.append(data[i + 1])

        if len(collected) >= min_count:
            break
    
    count = len(collected)
    mean_total = sum(collected) / count if count else None

    # 5. Clasificación en positivos y negativos
    if tipo == 0:
        pos = [x for x in collected if x >= 0]
        neg = [x for x in collected if x < 0]
    else:
        pos = [x for x in collected if x >= last1]
        neg = [x for x in collected if x < last1]

    # 6. Top-4 valores más comunes
    c = Counter(collected)
    for idx, (valor, cnt) in enumerate(c.most_common(4), start=1):
        print(f"{idx}. {valor:.4f} → {cnt} ", end="\t")
    print(f"\t\t\t\tj:{j}  lo {lo1:.3f}    hi {hi1:.3f}")
    # 7. Caídas por rango (suponiendo que devuelve List[(rango, cnt)], _)
    x, y = Caidas_por_rango(collected, last1, Pos)
    num = sum(((li + ls) / 2) * cnt for ((li, ls), cnt) in x)
    den = sum(cnt for _, cnt in x)
    prom_pon = num / den if den else 0
    for (li, ls), cnt in x:
        print(f"{li:.3f}-{ls:.3f}→ {cnt}", end="\t")
    print(f"\tProm pond: {prom_pon:.3f}")

    # 8. Medias finales y últimos datos
    mean_pos = sum(pos) / len(pos) if pos else None
    mean_neg = sum(neg) / len(neg) if neg else None
    ultimos_20 = data[-25:]
    sig_20 = collected[-25:]

    return {
        "N": count,
        "Ptot": mean_total,
        "Ppos": mean_pos,
        "Tpos": len(pos),
        "Pneg": mean_neg,
        "Tneg": len(neg),
        "Ult": last1,
        "Ant": ultimos_20,
        "Sig": sig_20
    }


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
        
        if Forma==0:
            for idx, (valor, yy) in enumerate(top2, start=1):
                parts.append(f"{idx}. {valor:.3f}->{yy} ")
            print("  ".join(parts),end=" ")
            print(f"\t\t\tlo {lo:.3f}  hi {hi:.3f} j:{j}\tParcial:{P_parcial:.2f} ")
        else:
            for idx, (valor, yy) in enumerate(top2, start=1):
                parts.append(f"{idx}. {valor:.2f}->{yy} ")
            print("  ".join(parts),end=" ")
            print(f"\t\t\t\tlo {lo:.3f}  hi {hi:.3f} j:{j}\tParcial:{P_parcial:.3f} ")
     
        x,y=Caidas_por_rango(collected, last, Pos)
        num   = sum(((li + ls) / 2) * cnt for (li, ls), cnt in x)
        den = sum(cnt for _, cnt in x)
        prom_pon = num / den if den else 0
            
        if Forma==0:
            print("  " + "| ".join(f"{li:.2f}-{ls:.2f}→ {cnt}" for (li, ls), cnt in x)+ "\t\t\t" + f"Prom pond: {prom_pon:.2f}")
        else:
            print("  " + "| ".join(f"{li:.3f}-{ls:.3f}→ {cnt}" for (li, ls), cnt in x)+ "\t\t" + f"Prom pond: {prom_pon:.3f}")

    mean_pos = sum(pos) / len(pos) if pos else None
    mean_neg = sum(neg) / len(neg) if neg else None
    ul5 = data[-40:]
    u5 = collected[-40:]
    return {
        "N": count,
        "Ptot": mean_total,
        "Ppos": mean_pos,
        "Tpos": len(pos),
        "Pneg": mean_neg,
        "Tneg": len(neg),
        "Ult": last,
        "Ant":ul5,
        "Sig":u5
    }


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
        else:
            txt = f"{val:.3f}"
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
        y1=sum(sigs[-10:])/len(sigs[-10:])
        y2=sum(sigs[-7:])/len(sigs[-7:])
        y3=sum(sigs[-3:])/len(sigs[-3:])
        if Forma==0:
            line2_vals = "\t".join(fmt_number(v) for v in sigs[-8:])
            print(f"Sig. :\t{line2_vals}\t\t\tP10: {y1:.2f}   P7: {y2:.2f}   P3: {y3:.2f}")
        else:
            line2_vals = " ".join(fmt_number(v) for v in sigs[-8:])
            print(f"Sig. :\t{line2_vals} \t\tP10: {y1:.3f}   P7: {y2:.3f}   P3: {y3:.3f}")

    # 3) Línea Anterior (ant)
    ants = stats.get("Ant", [])
    if len(ants)>0:
        y4=sum(ants[-10:])/len(ants[-10:])
        y5=sum(ants[-7:])/len(ants[-7:])
        y6=sum(ants[-3:])/len(ants[-3:])
        
        if Forma==0:
            line3_vals = "\t".join(fmt_number(v) for v in ants[-8:])
            print(f"Ant. :\t{line3_vals}\t\t\tP10: {y4:.2f}   P7: {y5:.2f}   P3: {y6:.2f}")
        else:
            line3_vals = " ".join(fmt_number(v) for v in ants[-8:])
            print(f"Ant. :\t{line3_vals} \t\tP10: {y4:.3f}   P7: {y5:.3f}   P3: {y6:.3f}")
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
        texto = "\033[30m" if valor > 0 else "\033[30m"  # Azul si >0, negro si ==0
    else:
        texto = "\033[31m"  # Rojo si negativo
    return f"{fondo}{texto}{valor:.3f}\033[0m"


def imprimir_valores_y_errores(s14, p_gral, start=0, stop=10, col_val="\033[93m", col_err="\033[96m"):
    formateados = []
    errores    = []
    nuevos     = []
    ajustados  = []

    for i in range(start, stop):
        val   = (s14 + i) / 20
        err   = (val - p_gral) / p_gral
        formateados.append(f"{val:.3f}\t")
        errores.append(f"{err:.3f}")
        ajustados.append(err * -0.999 if err < 0 else err)
        nuevos.append(val)

    print(f"{col_val}{' '.join(formateados)}\033[0m")
    print("\t".join(colorear2(float(e)) for e in errores))
    return nuevos, ajustados


def procesar_regresion_Histo(titulo, P, Lista, Valores,  medio=5, ante=2, fin=11, start=1, stop=15):
    print(aviso_ansi("                      * - * - * - * - * - * - * - * - * - * - * - * - *", (166, 216, 210), (38, 87, 83)))
    print(aviso_ansi(f"Resultados para {titulo}:", (118, 5, 30), (240, 220, 90)))
    y = dict(sorted(Valores.items(), key=lambda item: item[1]))
    L_30, L_15, L_6, L_sig, Sum14, PROM, S5 = promedios_y_errores_lista(Lista, P, y)
    best_svr, cv_score, best_params,PromT,Err6, ETo = aplicar_svr(L_30, L_15, L_6, L_sig)
    
    Ele=procesar_lista(ETo, 0, 0)
    print_colored_stats(Ele, 0)
    Elp=procesar_lista(ETo, 1, 0, 0) 
    print_colored_stats(Elp, 1)

    Prom10 = Err6[-12:].mean()
    Prom6 = Err6[-4:].mean()
    print(colorear(PromT, "\tPromedio "), end="\t")
    print(colorear(Prom10, "P. 10 "), end="\t\t")
    print(colorear(Prom6, "P. 4 "))

    #print("Parámetros para SVR:", best_params)
    print("\033[31mPromedio :\033[0m", end="\t")
    print(f"\033[1;91;47m{PROM:.4f}\033[0m", end="\t\t")  
    
    # Se toma el último valor de cada lista para formar una nueva observación.
    nuevo_dato = np.array([[L_30[-1], L_15[-1], L_6[-1]]])
    prediccion = best_svr.predict(nuevo_dato)
    Pprom=prediccion[0]
    datos_regresion = (L_30, L_15, L_6, L_sig)
    # Códigos ANSI
    BG_BLUE = '\033[46m'
    RED_ON_YELLOW = "\033[37;45m"
    RESET        = "\033[0m"
    # Verificar que todas las listas contengan datos, tengan la misma longitud y mayor a 50.
    if (L_30 and L_15 and L_6 and L_sig and 
        len(L_30) == len(L_15) == len(L_6) == len(L_sig) and len(L_30) > 80):
        resultados = procesar_regresiones(datos_regresion)
        # Extraer el valor de ElasticNet (se asume que es consistente en todos los métodos)
        primer_metodo = next(iter(resultados))
        valor_elasticnet = resultados[primer_metodo]["ElasticNet"]
        valores_wls = [float(datos["WLS"]) for datos in resultados.values()]
        Prom_wls = sum(valores_wls) / len(valores_wls)
        print(f"\033[1;94;43mRegresion : {Pprom:.2f}\033[0m", end="\t")
        print(f"'ElasticNet': {valor_elasticnet:.2f}", end="\t\t")
        
        if Ele['Ptot'] is None:
            Bb=Pprom*(1+Ele['Ptot'])
        else:
            Bb=Pprom

        print(f"\033[1;34;43m{Bb:.2f}\033[0m", end="\t")
        resultados = [Pprom * factor for factor in (0.9, 0.95, 1, 1.05, 1.1)]
        # Unimos con dos espacios como separador y lo imprimimos en la misma línea
        print("  ".join(f"{RED_ON_YELLOW}{valor:.2f}{RESET}" for valor in resultados))

        print(
            f"\033[34m\t\tProm + {ante}: {((S5 + ante) / 20):.2f}\033[0m\t\t"
            f"\033[31mProm Bajo: {((S5 + medio) / 20):.2f}\033[0m\t"
            f"\033[31mProm Medio: {((S5 + medio+3) / 20):.2f}\033[0m\t"
            f"\033[34mProm + {fin}: {((S5 + fin) / 20):.2f}\033[0m"
        )
        print(" -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- ")
        nuevos, errores, textoN, textoE = calcular_nuevos_y_errores(y, Sum14, Pprom)
        c=[]
        i=[]
        
        for k, v in y.items():  
            c.append(str(k))
            i.append(str(v))   
        
        linea1 = "\t".join(c)
        linea2 = "\t".join(i)
        print(f"\033[93m{linea1}\033[0m")
        print(f"\033[96m{linea2}\033[0m")

        linea3 = "\t".join(textoN)
        linea4 = "\t".join(textoE)
        print(f"\033[91m{linea3}\033[0m")
        print(f"\033[97m{linea4}\033[0m")
        v_min = min(nuevos.values())
        v_max = max(nuevos.values())
        Prom_Gral = solicitar_nuevo_prom(v_min, v_max)
        nuevos, errores, texto,textoE = calcular_nuevos_y_errores(y, Sum14, Prom_Gral)
        linea = "\t".join(textoN)
        linea1 = "\t".join(textoE)
        print(f"\033[92m{linea}\033[0m")
        print(f"\033[95m{linea1}\033[0m")

        print()
        # Retornamos ambas listas para que el main pueda reordenarlas si es necesario.
        return  errores, nuevos, Sum14

    else:
        print(f"No hay datos suficientes o las listas no cumplen las condiciones para {titulo}.")
        return None, None, None, None


def calcular_nuevos_y_errores(Valores: Dict[str, float], Sum14: float, Prom_Gral: float) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
    errores_por_clave = {}
    nuevos_por_clave  = {}
    formateados1       = []
    formateados2       = []

    for clave, incremento in Valores.items():
        nuevo_valor = (Sum14 + incremento) / 20
        error = (nuevo_valor - Prom_Gral) / Prom_Gral
        nuevos_por_clave[clave]  = nuevo_valor
        errores_por_clave[clave] = error
        formateados1.append(f"{nuevo_valor:.2f}")
        formateados2.append(f"{error:.3f}")
    return nuevos_por_clave, errores_por_clave, formateados1, formateados2


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
    n_segments: int =3,
    last_n: int =45
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
    error_abs_fd, error_rel_fd = compute_fd_errors(mean_p, last_p,F_d)
    # 4) Mapear errores según F_d
    error_fd = {k: error_abs_fd.get(k, None) for k in F_d.keys()}
    return mean_p, last_p, error_abs_fd, error_rel_fd, error_fd


def sumar_diccionarios(*dicts):
    total = defaultdict(float)
    cuenta = defaultdict(int)
    # Sumar y contar apariciones
    for d in dicts:
        for clave, valor in d.items():
            total[clave] += valor
            cuenta[clave] += 1
    # Calcular promedio por clave
    return {clave: total[clave] / cuenta[clave] for clave in total}
    #return ordenar_por_valor(diccionario_sumado, ascendente=False)


def remapear_por_posicion(claves_ordenadas: list, dic_posiciones: dict)-> dict:
    print("Hi 5")
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


def procesar_Histogramas(titulo: str, Pos: int, h_data, f_data: dict, *proc_args ):
    Y=len(h_data)
    if Y > 100:
        Error_val, Nuevo_valor, Sum14 = procesar_regresion_Histo(titulo, Pos, h_data, f_data, *proc_args)
        # 2) Ordenar frecuencias y errores
        caidas_ordenadas = ordenar_por_valor(f_data, ascendente=True)
        prom_ordenados   = ordenar_por_valor(Error_val, ascendente=True)
        # 3) Encontrar la clave con error mínimo
        llave_min = min(prom_ordenados, key=lambda k: abs(prom_ordenados[k])) #min(prom_ordenados, key=prom_ordenados.get)
        # 4) Imprimir tablas
        imprimir_tabla("Caídas", caidas_ordenadas, es_decimal=False, highlight_key=llave_min) 
        imprimir_tabla(f"Promedio {titulo}", prom_ordenados, es_decimal=True)
        return Error_val
    else:
        print("No hay suficentes datos para evaluar")


def Dicc_probabilidad_ordenado(lista_numeros, I_ini=0, I_fin=10, cant=30,ventanas=(15, 20, 20, 25)):
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


def escalar_dic(d, escalar):
    return {k: v * escalar for k, v in d.items()}


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

    ProNUm=ordenar_por_valor(Pb_por_Numero, ascendente=False)
    imprimir_tabla("Prob. Bayes Ordenadas Numeros Siguientes Jerarquia  ", ProNUm, es_decimal=True)
    print()
    

def Llamar_Numeros(columna, Lista, F_d):
    # Códigos ANSI para rojo y reset
    RED   = "\033[31m"
    RESET = "\033[0m"
    Nume2=Lista2_con_map(Lista)
    Sig_numeros = obtener_siguiente_numero(columna)
    #print(Sig_numeros)
    Ultima_Jerarquia=ultima_jerarquia(columna)
    jerarquias, Posic = calcular_jerarquias(columna)
    print(aviso_ansi(f"\nTotal Numeros Siguientes : {len(Sig_numeros)}", (118, 5, 30), (240, 220, 100) ))
    
    Ultima_J_Sig=ultima_jerarquia_Lista(Sig_numeros)
    Jer_Sig=ordenar_por_valor(Ultima_J_Sig, ascendente=False)
    
    #Procesamiento probabilidades ultimas
    DiccNu = Dicc_probabilidad_ordenado(Lista)
    imprimir_tabla("Probabilidades por Jerarquia ", DiccNu, es_decimal=True)

    if len(Sig_numeros) > 90:
        Sig_pd=pd.Series(Sig_numeros)
        jerarSig, PosSig = calcular_jerarquias(Sig_pd)





    print(Ultima_Jerarquia)
    claves_ordenadas = sorted(Ultima_Jerarquia.keys(), key=lambda k: (Ultima_Jerarquia[k], -F_d[k]))

    # Imprimir el resultado ordenado
    print(aviso_ansi("\nOrden de jerarquías :", (118, 5, 30), (240, 220, 90) ))
    for k in claves_ordenadas:
        print(f"Id: {k}\t Repet: {Ultima_Jerarquia[k]}\t Aparición: {F_d[k]}")
    print()
    ranking_dict = {rank: clave for rank, clave in enumerate(claves_ordenadas[:10])}

    Pr_Num2, _, _,  _ = procesar_e_imprimir_regresion("Numeros Medios", 1, Nume2, 1, 6,3,2,4)
    print()
    Pr_Num, _, _,  _ = procesar_e_imprimir_regresion("Numeros", 1, columna)
    Pr_Sig, _, _,  _ = procesar_e_imprimir_regresion("Siguientes", 1,Sig_numeros)
    Pr_Pos_val, Pr_Pos_err, Sum14, PromGral = procesar_e_imprimir_regresion("Jerarquía", 1, Posic,1,11,5,2,7)
    nuevos_valores_dict = {}
    errores_dict = {}
    
    for rank, clave in ranking_dict.items():
        nuevo_valor = (Sum14 + rank) / 20
        error = (nuevo_valor - PromGral) / PromGral
        if error < 0:
            error *= -0.999
        nuevos_valores_dict[clave] = nuevo_valor
        errores_dict[clave] = error
    print()

    sorted_keys = sorted(ranking_dict.values())  # esto da [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Pr_Pos_err_ordered = [errores_dict[k] for k in sorted_keys]
    
    # Calcula los mínimos
    min_jer = min(errores_dict.values())
    min_num = min(Pr_Num)
    min_num2 = min(Pr_Num2)
    min_sig = min(Pr_Sig)
    print("\tJerarquías\tNumeros\t\tSiguientes\tError Num")
    print("*******************************************************************")
    
    ErrorNUm={}
    for k in sorted_keys:
        jer = errores_dict[k]
        num = Pr_Num[k]
        sig = Pr_Sig[k]
        ErrorNUm[k] = (sig + 4*num + 3*jer) / 8 
        
        # Formatea cada celda, poniendo rojo si coincide con el mínimo
        s_jer = f"{jer:.4f}"
        if jer == min_jer:
            s_jer = f"{RED}{s_jer}{RESET}"
        
        s_num = f"{num:.4f}"
        if num == min_num:
            s_num = f"{RED}{s_num}{RESET}"
        
        s_sig = f"{sig:.4f}"
        if sig == min_sig:
            s_sig = f"{RED}{s_sig}{RESET}"
             
        print(f"{k}\t{s_jer}\t\t{s_num}\t\t{s_sig}\t\t{ErrorNUm[k]:.4f}")
    
    ErrorOrdenado=ordenar_por_valor(ErrorNUm, ascendente=True)
    print()
    imprimir_tabla("Errores Promedios Numeros ", ErrorOrdenado, es_decimal=True)
    
    # Imprimir resultados ajustados según el ranking (en orden de ranking)
    print("\nResultados ajustados de 'Jerarquía' reordenados mediante ranking_dict:")
    for rank in sorted(ranking_dict.keys()):
        clave = ranking_dict[rank]
        print(f"Id {rank}:\tNum {clave},\t Prom15: {nuevos_valores_dict[clave]:.4f},\t Er: {errores_dict[clave]:.4f}")
    print()
    
    #Prior1=calcular_alpha_prior(columna)
    #Prior=ordenar_por_valor(Prior1, ascendente=False)
    #PriorSig=calcular_alpha_prior_Lista(Sig_numeros)


def Llamada_Histo(columna, F_d):
    Histog=obtener_historial_caidas(columna)
    Sig_Histo=obtener_siguiente_caidas(columna)
    
    H2=Histo2_con_map(Histog,2)
    Sig_H2=Siguientes_lista(H2)
    F_datos_2 = {k: (v - 1) // 2 + 1 for k, v in F_d.items()}
    
    H3 = Histo2_con_map(Histog,3)
    F_datos_3 = {k: (v - 1) // 3 + 1 for k, v in F_d.items()}
    a=procesar_Histogramas("Histograma 1/3", 1, H3, F_datos_3, 4, 2, 6 )
    b=procesar_Histogramas("Histograma 1/2", 1, H2, F_datos_2, 5, 3, 7 )
    c={}
    if len(Sig_Histo)> 130:
        c=procesar_Histogramas("Siguiente Histog", 2, Sig_Histo, F_d, 6, 4, 8 )
    
    d={}
    if len(Sig_H2)> 130:
        d=procesar_Histogramas("Siguiente Histog 1/2 ", 1, Sig_H2, F_datos_2, 5, 3, 7 )
    
    e=procesar_Histogramas("Histograma ", 2, Histog, F_d, 7, 4, 11 )
    print()
    

    Suma_Total=sumar_diccionarios(escalar_dic(a, 2), escalar_dic(b, 2), c, d, escalar_dic(e, 4))
    Ordenados=ordenar_por_valor(Suma_Total, ascendente=False)
    imprimir_tabla("Probabilidad Caidas Histograma ", Ordenados, es_decimal=True)

    mean_p, last_p, errors, error_abs_fd, error_rel_fd = analyze_frecuencias(
        Histog, F_d, max_val=40, n_segments=4, last_n=45)

    mean_p2, last_p2, errors2, error_abs_fd2, error_rel_fd2 = analyze_frecuencias(
        H2, F_datos_2, max_val=20, n_segments=4, last_n=30)
    
    abs_fd=sumar_diccionarios(error_abs_fd, error_abs_fd, error_abs_fd2)
    imprimir_tabla("Probabilidad Semanas Caidas  ", abs_fd, es_decimal=True)
    print("                            *************          ***********")

    Bayes_Histo=porcentaje_coincidencias(F_d, Histog)
    PromOrdenados=ordenar_por_valor(Bayes_Histo, ascendente=False)
    imprimir_tabla("PORCENTAJE caidas Semanas", PromOrdenados, es_decimal=True)

def procesar_lista_tres(
    data: List[float],
    tipo: int,
    Pos: int
) -> Dict[str, Optional[float]]:
    min_count = 10
    max_iters = 50
    last1 = data[-1]
    last2 = data[-2]
    last3 = data[-3]
    # 1. Validación de longitud mínima
    if len(data) < 500:
        return {"N": 0, "Ptot": 0, "Ppos": 0, "Pneg": 0}
    # 2. Configurar paso según Pos
    step = 0.0045 if Pos == 0 else 0.002
    lo1, hi1 = last1 - step, last1 + step
    lo2 = hi2 = last2
    lo3 = hi3 = last3
    collected: List[float] = []

    # 4. Búsqueda iterativa hasta reunir min_count datos
    for j in range(max_iters):
        # Expande ambas ventanas
        #lo1, hi1 = lo1 - step, hi1 + step
        lo2, hi2 = lo2 - step, hi2 + step
        lo3, hi3 = lo3 - step, hi3 + step

        for i in range(1, len(data) - 1):
            primer = data[i - 2]
            anterior = data[i - 1]
            actual   = data[i]
            # Ambas ventanas deben cumplirse simultáneamente
            if lo1 < actual < hi1 and lo2 < anterior < hi2 and lo3 < primer < hi3:
                # Añade el siguiente dato tras 'actual'
                collected.append(data[i + 1])

        if len(collected) >= min_count:
            break

    #xy=analizar_collected(collected, last1, tipo, Pos, 20)
    count = len(collected)
    mean_total = sum(collected) / count if count else None
    # 5. Clasificación en positivos y negativos
    if tipo == 0:
        pos = [x for x in collected if x >= 0]
        neg = [x for x in collected if x < 0]
    else:
        pos = [x for x in collected if x >= last1]
        neg = [x for x in collected if x < last1]
    # 6. Top-4 valores más comunes
    c = Counter(collected)
    for idx, (valor, cnt) in enumerate(c.most_common(4), start=1):
        print(f"{idx}. {valor:.4f} → {cnt} ", end="\t")
    print(f"\t\t\t\tj:{j}  lo {lo1:.3f}    hi {hi1:.3f}")
    # 7. Caídas por rango (suponiendo que devuelve List[(rango, cnt)], _)
    x, y = Caidas_por_rango(collected, last1, Pos)
    num = sum(((li + ls) / 2) * cnt for ((li, ls), cnt) in x)
    den = sum(cnt for _, cnt in x)
    prom_pon = num / den if den else 0
    for (li, ls), cnt in x:
        print(f"{li:.3f}-{ls:.3f}→ {cnt}", end="\t")
    print(f"\tProm pond: {prom_pon:.3f}")
    # 8. Medias finales y últimos datos
    mean_pos = sum(pos) / len(pos) if pos else None
    mean_neg = sum(neg) / len(neg) if neg else None
    ultimos = data[-20:]
    sig_2 = collected[-20:]

    return {
        "N": count,
        "Ptot": mean_total,
        "Ppos": mean_pos,
        "Tpos": len(pos),
        "Pneg": mean_neg,
        "Tneg": len(neg),
        "Ult": last1,
        "Ant":ultimos,
        "Sig":sig_2
    }


def Lotery(Re, Res):
    banner = [
        # L        OOO      TTTTT    EEEEE   RRRR    Y   Y
        " **          *******      ********     *******     *******      **      ** ",
        " **         **  *  **        **        **          **   **       **   **  ",
        " **         **     **        **        **          **   **         ****   ",
        " **         **     **        **        ******      ******          ****   ",
        " **         **     **        **        **          **   **          **   ",
        " *******    **  *  **        **        **          **    **         **   ",
        " *******     *******         **        *******     **     **        **   ",
    ]
    for line in banner:
        print(f"{Re}{line}{Res}")


def main(file_path):
    # Códigos ANSI para rojo y reset
    RED   = "\033[31m"
    RESET = "\033[0m"
    #Lotery(RED, RESET)

    Numeros = leer_datos_excel(file_path)
    Nume=Numeros.tolist()
    F_datos=Semanas(Numeros)
    print(F_datos)
    Ultima_Jerarquia=ultima_jerarquia(Numeros)
    jerarquias, Posic = calcular_jerarquias(Numeros)
    Llamar_Numeros(Numeros, Nume, F_datos)
    Llamar_Numeros1(Numeros, Nume, F_datos)
    Llamada_Histo(Numeros, F_datos) 

    Ele=procesar_lista_tres(Nume, 0, 0)
    print_colored_stats(Ele, 0)
    
    Sig_numeros = obtener_siguiente_numero(Numeros)
    Ss=pd.Series(Sig_numeros)
    SHis=obtener_historial_caidas(Ss)
    F_dsig=Semanas(Ss)
    if len(SHis)> 100:
        c=procesar_Histogramas("Histog Siguientes", 2, Ss, F_dsig, 7, 4, 11 )
    
    Prior1=calcular_alpha_prior(Numeros)
    Prior=ordenar_por_valor(Prior1, ascendente=False)
 
    Probab_mayor = aplicar_regresion_logistica_mayor_menor(Numeros)
    if Probab_mayor is not None:
        print(f"\nProbabilidad (Reg. Logística) de que el siguiente número sea mayor que 4: {Probab_mayor:.4f}")

    Probab_par = aplicar_regresion_logistica_par_impar(Numeros)
    if Probab_par is not None:
        print(f"Probabilidad (Reg. Logística) de que el siguiente número sea par: {Probab_par:.4f}")
    print()

    Pos_Histo=calcular_jerarquia_histo(Nume)
    Pb_His, _, _,  _ = procesar_e_imprimir_regresion("Orden Caida", 1, Pos_Histo, 1,11,5,3,7)
    Pb_His = analizar_siguientes_numeros_para_probabilidades(Pos_Histo)
    Elo=procesar_lista(Pos_Histo, 1, 1)
    print_colored_stats(Elo,0)
    print("\n\nTERMINAMOS AQUI....")



if __name__ == "__main__":
    print("Hello World")
    file_path = 'D:/loter.xlsx'
    main(file_path)

