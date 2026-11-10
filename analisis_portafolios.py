import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from scipy.optimize import minimize


"""
FUNCIONES DEL LAB 1
"""

def rendimiento_anualizado(rendimientos, p=252):
    """Calcula el rendimiento geométrico anualizado (CAGR) de una serie de rendimientos.

    Args:
        rendimientos (pd.Series o np.ndarray): Serie de rendimientos simples/porcentuales por período.
        p (int, opcional): Número de períodos por año según la frecuencia temporal.
            - 252: Datos diarios de mercado bursátil (por defecto).
            - 365: Datos diarios continuos (ej. cripto).
            - 52: Datos semanales.
            - 12: Datos mensuales.
            - 4: Datos trimestrales.

    Returns:
        float: Rendimiento anualizado equivalente expresado en formato decimal (ej. 0.15 para 15%).
    """
    # número de periodos
    n_periods = rendimientos.shape[0]

    # rendimiento equivalente del periodo
    rendimiento_periodo = (rendimientos + 1).prod() ** (1 / n_periods) - 1

    # rendimiento anualizado equivalente
    rendimiento_anual = (1 + rendimiento_periodo) ** p - 1
    return rendimiento_anual


def volatilidad_anualizada(rendimientos, p=252):
    """Calcula la volatilidad (desviación estándar) anualizada de una serie de rendimientos.

    Args:
        rendimientos (pd.Series, pd.DataFrame o np.ndarray): Serie, matriz o DataFrame 
            de rendimientos por período.
        p (int, opcional): Número de períodos por año según la frecuencia temporal.
            - 252: Datos diarios de mercado bursátil (por defecto).
            - 365: Datos diarios continuos (ej. cripto).
            - 52: Datos semanales.
            - 12: Datos mensuales.
            - 4: Datos trimestrales.

    Returns:
        float o pd.Series: Volatilidad anualizada expresada en formato decimal 
        (ej. 0.20 para 20% de dispersión anual). Devuelve una Serie si la entrada es un DataFrame.
    """
    volatilidad = rendimientos.std()
    volatilidad_anualizada = volatilidad * p ** (1 / 2)
    return volatilidad_anualizada


def return_risk_ratio(rendimientos, p=252):
    """Calcula el ratio de rendimiento sobre riesgo anualizado (Return/Risk Ratio) de una serie de rendimientos.

    Args:
        rendimientos (pd.Series, pd.DataFrame o np.ndarray): Serie, matriz o DataFrame 
            de rendimientos por período.
        p (int, opcional): Número de períodos por año según la frecuencia temporal (252 por defecto).

    Returns:
        float o pd.Series: Ratio de rendimiento por unidad de volatilidad anualizada.
    """
    return rendimiento_anualizado(rendimientos, p=p) / volatilidad_anualizada(rendimientos, p=p)

def sharpe_ratio(rendimientos, tasa_libre_de_riesgo=0.0, p=252):
    """Calcula el Ratio de Sharpe anualizado de una serie de rendimientos.

    Args:
        rendimientos (pd.Series, pd.DataFrame o np.ndarray): Serie, matriz o DataFrame 
            de rendimientos por período.
        tasa_libre_de_riesgo (float, opcional): Tasa libre de riesgo anualizada en formato decimal 
            (ej. 0.05 para 5%). Por defecto es 0.0.
        p (int, opcional): Número de períodos por año según la frecuencia temporal.
            - 252: Datos diarios de mercado bursátil (por defecto).
            - 365: Datos diarios continuos (ej. cripto).
            - 52: Datos semanales.
            - 12: Datos mensuales.
            - 4: Datos trimestrales.

    Returns:
        float o pd.Series: Ratio de Sharpe anualizado. Devuelve una Serie si la entrada es un DataFrame.
    """
    exceso = rendimiento_anualizado(rendimientos, p=p) - tasa_libre_de_riesgo
    volatilidad = volatilidad_anualizada(rendimientos, p=p)
    return exceso / volatilidad


def calcular_drawdown(serie_rendimientos, base_inicial=100.0):
    """Calcula el índice de riqueza acumulada, el máximo histórico y la serie de drawdown,
    anclando el capital base inicial para reconocer caídas desde el primer período.

    Args:
        serie_rendimientos (pd.Series): Serie temporal de rendimientos porcentuales limpios.
        base_inicial (float, opcional): Capital base inicial. Por defecto 100.0.

    Returns:
        pd.DataFrame: DataFrame con 'Valor Acumulado', 'Máximo Anterior' y 'Drawdown'.
    """
    # 1. Riqueza acumulada a partir de los rendimientos
    valor_acumulado = base_inicial * (1 + serie_rendimientos).cumprod()

    # 2. El máximo histórico debe considerar el capital base inicial (100.0) como referencia mínima
    maximo_anterior = valor_acumulado.cummax().clip(lower=base_inicial)

    # 3. Drawdown medido contra el pico real
    drawdown = (valor_acumulado - maximo_anterior) / maximo_anterior

    return pd.DataFrame({
        "Valor Acumulado": valor_acumulado,
        "Máximo Anterior": maximo_anterior,
        "Drawdown": drawdown
    })


def max_drawdown(serie_rendimientos):
    """Calcula el Drawdown Máximo (MDD) y la fecha en la que se registró el piso de la caída.

    Args:
        serie_rendimientos (pd.Series): Serie temporal de rendimientos porcentuales.

    Returns:
        tuple (float, pd.Timestamp):
            - float: Valor mínimo del drawdown (en formato decimal, ej. -0.25 para -25%).
            - pd.Timestamp: Fecha exacta en la que se alcanzó el drawdown máximo.
    """
    df_dd = calcular_drawdown(serie_rendimientos)
    mdd = df_dd["Drawdown"].min()
    fecha_mdd = df_dd["Drawdown"].idxmin()
    return mdd, fecha_mdd


def plot_drawdown(serie_rendimientos, titulo="Análisis de Drawdown"):
    """Grafica en dos paneles sincronizados el índice de riqueza y el área de drawdown,
    resaltando la fecha y magnitud del Drawdown Máximo.

    Args:
        serie_rendimientos (pd.Series): Serie temporal de rendimientos porcentuales.
        titulo (str, opcional): Título principal del gráfico.

    Returns:
        tuple (matplotlib.figure.Figure, np.ndarray): Objetos Figure y Axes de Matplotlib.
    """
    df_dd = calcular_drawdown(serie_rendimientos)
    mdd, fecha_mdd = max_drawdown(serie_rendimientos)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Panel 1: Riqueza y Pico histórico
    ax1.plot(df_dd["Valor Acumulado"], label="Valor Acumulado", color="#1f77b4", lw=1.5)
    ax1.plot(df_dd["Máximo Anterior"], label="Máximo Histórico", color="#2ca02c", linestyle="--", alpha=0.7)
    ax1.set_title(titulo, fontsize=12, fontweight="bold")
    ax1.set_ylabel("Índice de Riqueza")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Panel 2: Drawdown submarino y punto de Max DD
    ax2.plot(df_dd["Drawdown"], color="#d62728", lw=1.2)
    ax2.fill_between(df_dd.index, df_dd["Drawdown"], color="#d62728", alpha=0.3)
    ax2.scatter(fecha_mdd, mdd, color="black", zorder=5, label=f"Max DD: {mdd:.2%} ({fecha_mdd.strftime('%Y-%m-%d')})")
    ax2.set_ylabel("Drawdown (%)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    return fig, (ax1, ax2)


"""
FUNCIONES LAB 2
"""

def prueba_jarque_bera(rendimientos, alpha=0.05):
    """Ejecuta la prueba de hipótesis de Jarque-Bera para evaluar la normalidad
    de una serie de rendimientos a partir de su sesgo y curtosis.

    Args:
        rendimientos (pd.Series | np.ndarray): Serie o arreglo con los rendimientos financieros.
        alpha (float, optional): Nivel de significancia para la regla de decisión.
            Por defecto es 0.05 (5%).

    Returns:
        tuple (float, float):
            - float: Valor del estadístico de prueba Jarque-Bera.
            - float: P-value asociado a la prueba.
    """
    x, p_value = stats.jarque_bera(rendimientos)
    print(f"Estadístico: {x:.3f}")
    print(f"P-value: {p_value}")
    if p_value < alpha:
        print(f"Con un nivel de significancia del {alpha*100}% se rechaza la hipótesis nula")
        print("Los rendimientos no tienen el sesgo y curtosis de una distribución normal")
    else:
        print(f"Con un nivel de significancia del {alpha*100}%")
        print("no existe evidencia suficiente para rechazar que los rendimientos tienen el sesgo y curtosis de una distribución normal")
    
    return x, p_value


def semideviation(r, objetivo=False):
    """Calcula la semidesviación (downside risk) considerando únicamente las
    desviaciones por debajo de un umbral o rendimiento objetivo.

    Args:
        r (pd.Series | pd.DataFrame): Serie temporal o DataFrame de rendimientos.
        objetivo (float | bool, optional): Rendimiento mínimo aceptable (MAR) o umbral.
            Si es False, se utiliza la media muestral de r como referencia.
            Por defecto es False.

    Returns:
        float | pd.Series: Semidesviación calculada en formato decimal. Devuelve un
            escalar (float) si r es pd.Series o una pd.Series si r es pd.DataFrame.
    """
    if objetivo is False:
        # No hay objetivo, se elige la media
        objetivo = r.mean()

    # Calcular las desviaciones negativas
    deviations = r[r < objetivo] - objetivo
    squared_deviations = deviations**2

    # Calcular la semidesviación
    semidev = np.sqrt(squared_deviations.mean())
    return semidev

def var_historic(r, level=5):
    """Calcula el Valor en Riesgo (VaR) no paramétrico (histórico) para un nivel de significancia dado.

    Args:
        r (pd.Series | pd.DataFrame | np.ndarray): Serie temporal, DataFrame o arreglo de rendimientos.
        level (float | int, optional): Nivel de significancia o probabilidad en la cola (en porcentaje, ej. 5 para 5%).
            Por defecto es 5.

    Returns:
        float | pd.Series: Valor en Riesgo histórico expresado como pérdida positiva (en formato decimal).
            Devuelve un escalar (float) si r es Series/array o una pd.Series si r es DataFrame.
    """
    return -np.percentile(r, level)


def var_gaussiano(r, nivel=5, modified=False):
    """Calcula el Valor en Riesgo (VaR) Paramétrico Gaussiano con estadísticas muestrales.
    Opcionalmente aplica la expansión de Cornish-Fisher para ajustar por sesgo y 
    curtosis muestrales (VaR modificado).

    Args:
        r (pd.Series | pd.DataFrame): Serie temporal o DataFrame de rendimientos.
        nivel (float | int, optional): Nivel de significancia o probabilidad en la cola (en porcentaje, ej. 5 para 5%).
            Por defecto es 5.
        modified (bool, optional): Indica si se aplica la aproximación de Cornish-Fisher
            para incorporar asimetría y curtosis muestrales observadas. Por defecto es False.

    Returns:
        float | pd.Series: Valor en Riesgo paramétrico expresado como pérdida positiva (en formato decimal).
            Devuelve un escalar (float) si r es pd.Series o una pd.Series si r es pd.DataFrame.
    """
    # Cuantil Z bajo normalidad estándar
    z = stats.norm.ppf(nivel / 100)

    if modified:
        # bias=False calcula los estimadores insesgados (muestrales)
        # fisher=False devuelve la curtosis de Pearson (donde la normal vale 3)
        s = stats.skew(r, bias=False)
        k = stats.kurtosis(r, fisher=False, bias=False)

        z = (z +
             (z**2 - 1) * s / 6 +
             (z**3 - 3*z) * (k - 3) / 24 -
             (2*z**3 - 5*z) * (s**2) / 36
            )

    # ddof=1 para la desviación estándar muestral (N - 1)
    return -(r.mean() + z * r.std(ddof=1))


def cvar_historic(r, level=5):
    """Calcula el Valor en Riesgo Condicional (CVaR / Expected Shortfall) histórico,
    correspondiente a la pérdida promedio esperada en los escenarios que superan el VaR histórico.

    Args:
        r (pd.Series | pd.DataFrame): Serie temporal o DataFrame de rendimientos.
        level (float | int, optional): Nivel de significancia o probabilidad en la cola (en porcentaje, ej. 5 para 5%).
            Por defecto es 5.

    Raises:
        TypeError: Si r no es una instancia de pd.Series o pd.DataFrame.

    Returns:
        float | pd.Series: CVaR histórico expresado como pérdida positiva (en formato decimal).
            Devuelve un escalar (float) si r es pd.Series o una pd.Series si r es pd.DataFrame.
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

    
"""
FUNCIONES LAB 3 y LAB 4
"""

def matriz_covarianza_anualizada(rendimientos, p=252):
    """Calcula la matriz de varianzas y covarianzas anualizada de un conjunto de activos.

    Args:
        rendimientos (pd.DataFrame): DataFrame de rendimientos por período, una columna por activo.
        p (int, opcional): Número de períodos por año según la frecuencia temporal.
            - 252: Datos diarios de mercado bursátil (por defecto).
            - 365: Datos diarios continuos (ej. cripto).
            - 52: Datos semanales.
            - 12: Datos mensuales.
            - 4: Datos trimestrales.

    Returns:
        pd.DataFrame: Matriz de covarianzas anualizada (activos x activos).
    """
    return rendimientos.cov() * p


def plot_correlacion(rendimientos, titulo="Matriz de Correlación"):
    """Grafica un mapa de calor de la matriz de correlación de los activos,
    anotando el coeficiente de correlación en cada celda.

    Args:
        rendimientos (pd.DataFrame): DataFrame de rendimientos por período, una columna por activo.
        titulo (str, opcional): Título del gráfico.

    Returns:
        tuple (matplotlib.figure.Figure, matplotlib.axes.Axes): Objetos Figure y Axes de Matplotlib.
    """
    corr = rendimientos.corr()
    n = corr.shape[0]

    fig, ax = plt.subplots(figsize=(1.2 * n + 3, 1.2 * n + 2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    # Anotamos el valor de la correlación en cada celda
    for i in range(n):
        for j in range(n):
            valor = corr.iloc[i, j]
            color = "white" if abs(valor) > 0.6 else "black"
            ax.text(j, i, f"{valor:.2f}", ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig, ax


def rendimiento_portafolio(pesos, rendimientos_esperados):
    """Calcula el rendimiento esperado de un portafolio como el promedio ponderado
    de los rendimientos esperados de sus activos.

    Args:
        pesos (np.ndarray | pd.Series): Pesos de cada activo en el portafolio (deben sumar 1).
        rendimientos_esperados (np.ndarray | pd.Series): Rendimiento esperado anualizado de cada activo.

    Returns:
        float: Rendimiento esperado del portafolio en formato decimal.
    """
    return np.asarray(pesos) @ np.asarray(rendimientos_esperados)


def volatilidad_portafolio(pesos, matriz_cov):
    """Calcula la volatilidad (desviación estándar) de un portafolio a partir de sus pesos
    y de la matriz de covarianzas de los activos.

    Args:
        pesos (np.ndarray | pd.Series): Pesos de cada activo en el portafolio (deben sumar 1).
        matriz_cov (pd.DataFrame | np.ndarray): Matriz de covarianzas (anualizada si se
            quiere la volatilidad anualizada).

    Returns:
        float: Volatilidad del portafolio en formato decimal.
    """
    w = np.asarray(pesos)
    return np.sqrt(w @ np.asarray(matriz_cov) @ w)


def rendimientos_historicos_portafolio(pesos, rendimientos):
    """Construye la serie de rendimientos por período de un portafolio con pesos fijos
    (rebalanceo en cada período). Útil para aplicar al portafolio las funciones de
    los Labs 1 y 2 (Sharpe, drawdown, VaR, CVaR).

    Args:
        pesos (np.ndarray | pd.Series): Pesos de cada activo en el portafolio (deben sumar 1).
        rendimientos (pd.DataFrame): DataFrame de rendimientos por período, una columna por activo.

    Returns:
        pd.Series: Serie temporal de rendimientos del portafolio.
    """
    return rendimientos @ np.asarray(pesos)


def frontera_2_activos(rendimientos_esperados, matriz_cov, n_puntos=50):
    """Calcula la frontera de portafolios formados por dos activos, recorriendo
    el peso del primer activo de 0 a 1 (el segundo recibe el complemento).

    Args:
        rendimientos_esperados (pd.Series | np.ndarray): Rendimientos esperados de los 2 activos.
        matriz_cov (pd.DataFrame | np.ndarray): Matriz de covarianzas 2x2.
        n_puntos (int, opcional): Número de combinaciones a evaluar. Por defecto 50.

    Raises:
        ValueError: Si se proporcionan más o menos de 2 activos.

    Returns:
        pd.DataFrame: DataFrame con columnas 'Peso Activo 1', 'Rendimiento' y 'Volatilidad'.
    """
    if len(rendimientos_esperados) != 2:
        raise ValueError("frontera_2_activos sólo acepta exactamente 2 activos")

    pesos = [np.array([w, 1 - w]) for w in np.linspace(0, 1, n_puntos)]
    return pd.DataFrame({
        "Peso Activo 1": [w[0] for w in pesos],
        "Rendimiento": [rendimiento_portafolio(w, rendimientos_esperados) for w in pesos],
        "Volatilidad": [volatilidad_portafolio(w, matriz_cov) for w in pesos],
    })


def minimizar_volatilidad(rendimiento_objetivo, rendimientos_esperados, matriz_cov):
    """Encuentra los pesos del portafolio con la menor volatilidad posible que alcanza
    un rendimiento objetivo (problema de Markowitz). No se permiten ventas en corto.

    Args:
        rendimiento_objetivo (float): Rendimiento esperado que debe alcanzar el portafolio.
        rendimientos_esperados (pd.Series | np.ndarray): Rendimiento esperado de cada activo.
        matriz_cov (pd.DataFrame | np.ndarray): Matriz de covarianzas de los activos.

    Returns:
        np.ndarray: Pesos óptimos de cada activo (suman 1, cada uno entre 0 y 1).
    """
    n = len(rendimientos_esperados)
    pesos_iniciales = np.repeat(1 / n, n)
    limites = ((0.0, 1.0),) * n  # sin ventas en corto

    restricciones = (
        # los pesos deben sumar 1
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        # el portafolio debe alcanzar el rendimiento objetivo
        {"type": "eq", "fun": lambda w: rendimiento_portafolio(w, rendimientos_esperados) - rendimiento_objetivo},
    )

    resultado = minimize(volatilidad_portafolio, pesos_iniciales,
                         args=(matriz_cov,), method="SLSQP",
                         bounds=limites, constraints=restricciones,
                         options={"disp": False})
    return resultado.x


def portafolio_minima_varianza(matriz_cov):
    """Calcula los pesos del Portafolio de Mínima Varianza Global (GMV), es decir,
    el portafolio con la menor volatilidad posible sin importar su rendimiento.
    No se permiten ventas en corto.

    Args:
        matriz_cov (pd.DataFrame | np.ndarray): Matriz de covarianzas de los activos.

    Returns:
        np.ndarray: Pesos del portafolio GMV (suman 1, cada uno entre 0 y 1).
    """
    n = matriz_cov.shape[0]
    pesos_iniciales = np.repeat(1 / n, n)
    limites = ((0.0, 1.0),) * n
    restricciones = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    resultado = minimize(volatilidad_portafolio, pesos_iniciales,
                         args=(matriz_cov,), method="SLSQP",
                         bounds=limites, constraints=restricciones,
                         options={"disp": False})
    return resultado.x


def portafolio_max_sharpe(rendimientos_esperados, matriz_cov, tasa_libre_de_riesgo=0.0):
    """Calcula los pesos del portafolio con el máximo Ratio de Sharpe (portafolio tangente).
    Como scipy sólo minimiza, se minimiza el Sharpe con signo negativo.
    No se permiten ventas en corto.

    Args:
        rendimientos_esperados (pd.Series | np.ndarray): Rendimiento esperado anualizado de cada activo.
        matriz_cov (pd.DataFrame | np.ndarray): Matriz de covarianzas anualizada de los activos.
        tasa_libre_de_riesgo (float, opcional): Tasa libre de riesgo anualizada en formato decimal
            (ej. 0.07 para 7%). Por defecto es 0.0.

    Returns:
        np.ndarray: Pesos del portafolio de máximo Sharpe (suman 1, cada uno entre 0 y 1).
    """
    n = len(rendimientos_esperados)
    pesos_iniciales = np.repeat(1 / n, n)
    limites = ((0.0, 1.0),) * n
    restricciones = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    def sharpe_negativo(w):
        exceso = rendimiento_portafolio(w, rendimientos_esperados) - tasa_libre_de_riesgo
        return -exceso / volatilidad_portafolio(w, matriz_cov)

    resultado = minimize(sharpe_negativo, pesos_iniciales,
                         method="SLSQP", bounds=limites,
                         constraints=restricciones, options={"disp": False})
    return resultado.x


def frontera_eficiente(rendimientos_esperados, matriz_cov, n_puntos=50):
    """Calcula la frontera eficiente resolviendo el problema de Markowitz para una malla
    de rendimientos objetivo que va del rendimiento del portafolio de mínima varianza (GMV)
    al mayor rendimiento esperado de los activos. Los portafolios por debajo del GMV
    se omiten porque son ineficientes. No se permiten ventas en corto.

    Args:
        rendimientos_esperados (pd.Series): Rendimiento esperado anualizado de cada activo.
        matriz_cov (pd.DataFrame): Matriz de covarianzas anualizada de los activos.
        n_puntos (int, opcional): Número de puntos de la frontera. Por defecto 50.

    Returns:
        pd.DataFrame: DataFrame con columnas 'Rendimiento', 'Volatilidad' y una columna
            con el peso de cada activo.
    """
    rendimiento_gmv = rendimiento_portafolio(portafolio_minima_varianza(matriz_cov), rendimientos_esperados)
    objetivos = np.linspace(rendimiento_gmv, np.max(rendimientos_esperados), n_puntos)
    pesos = [minimizar_volatilidad(obj, rendimientos_esperados, matriz_cov) for obj in objetivos]

    df = pd.DataFrame(pesos, columns=getattr(rendimientos_esperados, "index", None))
    df.insert(0, "Volatilidad", [volatilidad_portafolio(w, matriz_cov) for w in pesos])
    df.insert(0, "Rendimiento", [rendimiento_portafolio(w, rendimientos_esperados) for w in pesos])
    return df


def plot_frontera_eficiente(rendimientos_esperados, matriz_cov, n_puntos=50,
                            tasa_libre_de_riesgo=0.0, mostrar_cml=True,
                            mostrar_gmv=True, mostrar_ew=True,
                            titulo="Frontera Eficiente"):
    """Grafica la frontera eficiente junto con los activos individuales y, opcionalmente,
    el portafolio de mínima varianza (GMV), el portafolio igualmente ponderado (EW)
    y la Línea del Mercado de Capitales (CML) que toca al portafolio de máximo Sharpe.

    Args:
        rendimientos_esperados (pd.Series): Rendimiento esperado anualizado de cada activo.
        matriz_cov (pd.DataFrame): Matriz de covarianzas anualizada de los activos.
        n_puntos (int, opcional): Número de puntos de la frontera. Por defecto 50.
        tasa_libre_de_riesgo (float, opcional): Tasa libre de riesgo anualizada. Por defecto 0.0.
        mostrar_cml (bool, opcional): Dibuja la CML y el portafolio de máximo Sharpe. Por defecto True.
        mostrar_gmv (bool, opcional): Marca el portafolio de mínima varianza. Por defecto True.
        mostrar_ew (bool, opcional): Marca el portafolio igualmente ponderado. Por defecto True.
        titulo (str, opcional): Título del gráfico.

    Returns:
        tuple (matplotlib.figure.Figure, matplotlib.axes.Axes): Objetos Figure y Axes de Matplotlib.
    """
    frontera = frontera_eficiente(rendimientos_esperados, matriz_cov, n_puntos)
    vols_activos = np.sqrt(np.diag(np.asarray(matriz_cov)))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frontera["Volatilidad"], frontera["Rendimiento"], color="#1f77b4", lw=2, label="Frontera eficiente")

    # Activos individuales
    ax.scatter(vols_activos, rendimientos_esperados, color="gray", zorder=4)
    for nombre, x, y in zip(rendimientos_esperados.index, vols_activos, rendimientos_esperados):
        ax.annotate(nombre, (x, y), textcoords="offset points", xytext=(5, 5))

    if mostrar_ew:
        n = len(rendimientos_esperados)
        w_ew = np.repeat(1 / n, n)
        ax.scatter(volatilidad_portafolio(w_ew, matriz_cov), rendimiento_portafolio(w_ew, rendimientos_esperados),
                   color="#ff7f0e", marker="s", s=70, zorder=5, label="Igualmente ponderado")

    if mostrar_gmv:
        w_gmv = portafolio_minima_varianza(matriz_cov)
        ax.scatter(volatilidad_portafolio(w_gmv, matriz_cov), rendimiento_portafolio(w_gmv, rendimientos_esperados),
                   color="#2ca02c", marker="D", s=70, zorder=5, label="Mínima varianza (GMV)")

    if mostrar_cml:
        w_msr = portafolio_max_sharpe(rendimientos_esperados, matriz_cov, tasa_libre_de_riesgo)
        vol_msr = volatilidad_portafolio(w_msr, matriz_cov)
        rend_msr = rendimiento_portafolio(w_msr, rendimientos_esperados)
        ax.plot([0, vol_msr * 1.5],
                [tasa_libre_de_riesgo, tasa_libre_de_riesgo + 1.5 * (rend_msr - tasa_libre_de_riesgo)],
                color="#d62728", linestyle="--", lw=1.5, label="CML")
        ax.scatter(vol_msr, rend_msr, color="#d62728", marker="*", s=200, zorder=5, label="Máximo Sharpe")

    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.set_xlabel("Volatilidad anualizada")
    ax.set_ylabel("Rendimiento anualizado")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    return fig, ax


"""
FUNCIONES LAB 4 (CML, MSR y GMV)

"""

def rendimiento_cml(volatilidad, rend_msr, vol_msr, tasa_libre_de_riesgo=0.0):
    """Evalúa la ecuación de la Línea del Mercado de Capitales (CML):
    r = r_f + (r_MSR - r_f) / sigma_MSR * sigma.

    Args:
        volatilidad (float | np.ndarray): Volatilidad (o arreglo de volatilidades) en la que se evalúa la recta.
        rend_msr (float): Rendimiento esperado del portafolio de máximo Sharpe.
        vol_msr (float): Volatilidad del portafolio de máximo Sharpe.
        tasa_libre_de_riesgo (float, opcional): Tasa libre de riesgo anualizada. Por defecto 0.0.

    Returns:
        float | np.ndarray: Rendimiento esperado sobre la CML para cada volatilidad.
    """
    pendiente = (rend_msr - tasa_libre_de_riesgo) / vol_msr
    return tasa_libre_de_riesgo + pendiente * np.asarray(volatilidad)


def portafolios_cml(rend_msr, vol_msr, tasa_libre_de_riesgo=0.0, proporciones=None):
    """Construye portafolios sobre la CML combinando el activo libre de riesgo con el MSR.
    Una proporción mayor a 1 significa pedir prestado a la tasa libre de riesgo para
    invertir más del 100% en el MSR (apalancamiento).

    Args:
        rend_msr (float): Rendimiento esperado del portafolio de máximo Sharpe.
        vol_msr (float): Volatilidad del portafolio de máximo Sharpe.
        tasa_libre_de_riesgo (float, opcional): Tasa libre de riesgo anualizada. Por defecto 0.0.
        proporciones (array-like, opcional): Proporciones invertidas en el MSR.
            Por defecto de 0 a 1.5 en pasos de 0.1.

    Returns:
        pd.DataFrame: DataFrame con columnas 'Peso MSR', 'Peso Libre de Riesgo',
            'Rendimiento', 'Volatilidad' y 'Sharpe'.
    """
    if proporciones is None:
        proporciones = np.linspace(0, 1.5, 16)
    a = np.asarray(proporciones, dtype=float)

    rendimiento = (1 - a) * tasa_libre_de_riesgo + a * rend_msr
    volatilidad = a * vol_msr  # el activo libre de riesgo no tiene volatilidad ni covarianza

    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(volatilidad > 0, (rendimiento - tasa_libre_de_riesgo) / volatilidad, np.nan)

    return pd.DataFrame({
        "Peso MSR": a,
        "Peso Libre de Riesgo": 1 - a,
        "Rendimiento": rendimiento,
        "Volatilidad": volatilidad,
        "Sharpe": sharpe,
    })


def pesos_gmv_2_activos(matriz_cov):
    """Calcula con la fórmula cerrada los pesos del portafolio de mínima varianza de 2 activos:
    w_1 = (sigma_2^2 - sigma_12) / (sigma_1^2 + sigma_2^2 - 2 sigma_12),  w_2 = 1 - w_1.
    A diferencia de portafolio_minima_varianza, sí permite ventas en corto,
    por lo que los pesos pueden quedar fuera de [0, 1].

    Args:
        matriz_cov (pd.DataFrame | np.ndarray): Matriz de covarianzas 2x2.

    Raises:
        ValueError: Si la matriz no es de 2x2.

    Returns:
        np.ndarray: Pesos [w_1, w_2] del portafolio de mínima varianza.
    """
    s = np.asarray(matriz_cov)
    if s.shape != (2, 2):
        raise ValueError("pesos_gmv_2_activos sólo acepta una matriz de covarianzas de 2x2")

    w1 = (s[1, 1] - s[0, 1]) / (s[0, 0] + s[1, 1] - 2 * s[0, 1])
    return np.array([w1, 1 - w1])


def sensibilidad_pesos(rendimientos_esperados, matriz_cov, activo, valores, tasa_libre_de_riesgo=0.0):
    """Recalcula los pesos del MSR y del GMV variando el rendimiento esperado de un activo,
    manteniendo fijos los demás rendimientos y la matriz de covarianzas. Sirve para ver
    que el MSR es muy sensible a los rendimientos esperados y el GMV no depende de ellos.

    Args:
        rendimientos_esperados (pd.Series): Rendimiento esperado anualizado de cada activo.
        matriz_cov (pd.DataFrame): Matriz de covarianzas anualizada de los activos.
        activo (str): Nombre del activo cuyo rendimiento esperado se modifica.
        valores (array-like): Rendimientos esperados a probar para ese activo.
        tasa_libre_de_riesgo (float, opcional): Tasa libre de riesgo anualizada. Por defecto 0.0.

    Returns:
        tuple (pd.DataFrame, pd.DataFrame): Pesos del MSR y pesos del GMV; una fila por cada
            valor probado y una columna por activo.
    """
    filas_msr, filas_gmv = [], []
    for valor in valores:
        r = rendimientos_esperados.copy()
        r[activo] = valor
        filas_msr.append(portafolio_max_sharpe(r, matriz_cov, tasa_libre_de_riesgo))
        filas_gmv.append(portafolio_minima_varianza(matriz_cov))

    indice = pd.Index(np.round(valores, 4), name=f"Rendimiento esperado de {activo}")
    pesos_msr = pd.DataFrame(filas_msr, index=indice, columns=rendimientos_esperados.index)
    pesos_gmv = pd.DataFrame(filas_gmv, index=indice, columns=rendimientos_esperados.index)
    return pesos_msr, pesos_gmv


def plot_pesos_apilados(pesos, titulo="Pesos del portafolio", ax=None):
    """Grafica en barras apiladas los pesos de varios portafolios (una barra por fila).

    Args:
        pesos (pd.DataFrame): Una fila por portafolio y una columna por activo
            (por ejemplo, la salida de sensibilidad_pesos).
        titulo (str, opcional): Título del gráfico.
        ax (matplotlib.axes.Axes, opcional): Eje donde dibujar. Si es None se crea una figura nueva.

    Returns:
        tuple (matplotlib.figure.Figure, matplotlib.axes.Axes): Objetos Figure y Axes de Matplotlib.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    etiquetas = [f"{v:.0%}" if isinstance(v, (int, float, np.floating)) else str(v) for v in pesos.index]
    pesos.set_axis(etiquetas, axis=0).plot(kind="bar", stacked=True, ax=ax, width=0.7, alpha=0.85)

    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.set_xlabel(pesos.index.name if pesos.index.name else "")
    ax.set_ylabel("Peso")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


"""
FUNCIONES LAB 5 (CPPI y simulación)

"""

def rendimientos_mensuales(rendimientos):
    """Convierte rendimientos diarios en rendimientos mensuales componiendo los
    rendimientos de cada mes: (1 + r_1)(1 + r_2)...(1 + r_n) - 1.

    Args:
        rendimientos (pd.Series | pd.DataFrame): Rendimientos diarios con índice de fechas.

    Returns:
        pd.Series | pd.DataFrame: Rendimientos mensuales (una fila por mes, fecha de fin de mes).
    """
    try:
        agrupados = (1 + rendimientos).resample("ME")
    except ValueError:  # versiones de pandas anteriores a 2.2
        agrupados = (1 + rendimientos).resample("M")
    return agrupados.prod() - 1


def rendimiento_aritmetico_anualizado(rendimientos, p=252):
    """Calcula el rendimiento aritmético anualizado (media por período x p).
    Es el parámetro mu que usa el Movimiento Browniano Geométrico. No confundir con
    rendimiento_anualizado (Lab 1), que es geométrico (CAGR) y es menor por
    aproximadamente sigma^2 / 2.

    Args:
        rendimientos (pd.Series | pd.DataFrame): Rendimientos simples por período.
        p (int, opcional): Número de períodos por año (252 diarios, 12 mensuales). Por defecto 252.

    Returns:
        float | pd.Series: Rendimiento aritmético anualizado en formato decimal.
    """
    return rendimientos.mean() * p


def cppi(rend_riesgo, tasa_libre_de_riesgo=0.0, m=3, piso=0.8, valor_inicial=100000,
         max_drawdown=None, p=12):
    """Aplica la estrategia CPPI (Constant Proportion Portfolio Insurance) rebalanceando
    en cada período. En cada período:
        C = V - F,   E = min(M x C, V),   B = V - E
    (E no puede ser negativo: si V < F todo se invierte en el activo libre de riesgo).

    Args:
        rend_riesgo (pd.Series | pd.DataFrame): Rendimientos por período del activo con riesgo.
            Si es un DataFrame, cada columna es un escenario (por ejemplo, una simulación).
        tasa_libre_de_riesgo (float, opcional): Tasa libre de riesgo anualizada. Por defecto 0.0.
        m (float, opcional): Multiplicador M. Por defecto 3.
        piso (float, opcional): Piso de protección F como proporción del valor inicial. Por defecto 0.8.
        valor_inicial (float, opcional): Valor inicial del portafolio V. Por defecto 100,000.
        max_drawdown (float, opcional): Máximo drawdown aceptable (1 - alfa), por ejemplo 0.1.
            Si se indica, el piso se vuelve dinámico: F_t = alfa x P_t, con P_t el valor máximo
            alcanzado por el portafolio; en ese caso se ignora 'piso'. Por defecto None.
        p (int, opcional): Número de períodos por año, para convertir la tasa libre de riesgo
            (12 si se rebalancea mensualmente). Por defecto 12.

    Returns:
        dict: Diccionario de DataFrames (una columna por escenario, una fila por período, valores
            al final de cada período) con las llaves:
            - 'Valor CPPI': valor del portafolio con la estrategia CPPI.
            - 'Valor riesgo': valor si se invierte el 100% en el activo con riesgo.
            - 'Piso': piso de protección vigente en cada período.
            - 'Peso riesgo': proporción E / V invertida en el activo con riesgo durante el período.
            - 'Rendimiento CPPI': rendimiento por período de la estrategia CPPI (para usar las
              funciones de los Labs 1 y 2).
    """
    if isinstance(rend_riesgo, pd.Series):
        rend_riesgo = rend_riesgo.to_frame(name=rend_riesgo.name if rend_riesgo.name is not None else "Riesgo")

    rf_periodo = (1 + tasa_libre_de_riesgo) ** (1 / p) - 1
    n_periodos, n_escenarios = rend_riesgo.shape

    valor_cppi = pd.DataFrame(np.nan, index=rend_riesgo.index, columns=rend_riesgo.columns)
    piso_hist = valor_cppi.copy()
    peso_hist = valor_cppi.copy()

    V = np.full(n_escenarios, float(valor_inicial))  # valor del portafolio
    P = V.copy()  # valor máximo alcanzado (para el max drawdown)
    F = np.full(n_escenarios, piso * valor_inicial)  # piso fijo

    for t in range(n_periodos):
        if max_drawdown is not None:
            F = (1 - max_drawdown) * P  # piso dinámico
        C = V - F  # colchón
        peso_E = np.clip(m * C / V, 0, 1)  # E = min(M x C, V) y no negativo
        E = peso_E * V
        B = V - E

        # el activo con riesgo y el libre de riesgo crecen durante el período
        V = E * (1 + rend_riesgo.iloc[t].values) + B * (1 + rf_periodo)
        P = np.maximum(P, V)

        valor_cppi.iloc[t] = V
        piso_hist.iloc[t] = F
        peso_hist.iloc[t] = peso_E

    valor_riesgo = valor_inicial * (1 + rend_riesgo).cumprod()
    valor_anterior = valor_cppi.shift(1)
    valor_anterior.iloc[0] = valor_inicial
    rend_cppi = valor_cppi / valor_anterior - 1

    return {
        "Valor CPPI": valor_cppi,
        "Valor riesgo": valor_riesgo,
        "Piso": piso_hist,
        "Peso riesgo": peso_hist,
        "Rendimiento CPPI": rend_cppi,
    }


def resumen_cppi(resultado):
    """Resume los valores finales de la estrategia CPPI contra invertir el 100% en el
    activo con riesgo. Con varios escenarios (simulación) da la distribución del valor final.

    Args:
        resultado (dict): Salida de la función cppi.

    Returns:
        pd.DataFrame: Filas 'CPPI' y '100% riesgo'. Con un solo escenario: valor final, valor
            mínimo y si el valor terminó algún período por debajo del piso (1 = sí, 0 = no).
            Con varios escenarios: valor final promedio, mediana, percentil 5, peor caso y la
            proporción de escenarios que terminaron algún período por debajo del piso (gap risk).
    """
    piso = resultado["Piso"]
    un_escenario = piso.shape[1] == 1
    filas = {}
    for nombre, llave in [("CPPI", "Valor CPPI"), ("100% riesgo", "Valor riesgo")]:
        valores = resultado[llave]
        finales = valores.iloc[-1]
        rompe_piso = (valores < piso).any()
        if un_escenario:
            filas[nombre] = {
                "Valor final": finales.iloc[0],
                "Valor mínimo": valores.min().iloc[0],
                "Rompió el piso": float(rompe_piso.iloc[0]),
            }
        else:
            filas[nombre] = {
                "Valor final promedio": finales.mean(),
                "Valor final mediana": finales.median(),
                "Percentil 5": finales.quantile(0.05),
                "Peor caso": finales.min(),
                "Prob. romper el piso": rompe_piso.mean(),
            }
    return pd.DataFrame(filas).T


def plot_cppi(resultado, escenario=None, titulo="Estrategia CPPI"):
    """Grafica en dos paneles el valor del CPPI contra invertir el 100% en el activo con
    riesgo junto con el piso de protección, y la proporción invertida en el activo con riesgo.

    Args:
        resultado (dict): Salida de la función cppi.
        escenario (str | int, opcional): Columna (escenario) a graficar. Por defecto la primera.
        titulo (str, opcional): Título principal del gráfico.

    Returns:
        tuple (matplotlib.figure.Figure, np.ndarray): Objetos Figure y Axes de Matplotlib.
    """
    if escenario is None:
        escenario = resultado["Valor CPPI"].columns[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Panel 1: valor del portafolio y piso
    ax1.plot(resultado["Valor riesgo"][escenario], label="100% activo con riesgo", color="#1f77b4", lw=1.5)
    ax1.plot(resultado["Valor CPPI"][escenario], label="CPPI", color="#ff7f0e", lw=1.8)
    ax1.plot(resultado["Piso"][escenario], label="Piso (F)", color="#d62728", linestyle="--")
    ax1.set_title(titulo, fontsize=12, fontweight="bold")
    ax1.set_ylabel("Valor del portafolio")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Panel 2: peso en el activo con riesgo
    ax2.plot(resultado["Peso riesgo"][escenario], color="#2ca02c", lw=1.5, drawstyle="steps-post")
    ax2.set_ylabel("Peso en riesgo (E/V)")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, (ax1, ax2)


def simular_gbm(mu, sigma, n_escenarios=1000, anios=3, p=12, valor_inicial=100, semilla=None,
                precios=False):
    """Simula escenarios del Movimiento Browniano Geométrico con el esquema de Euler:
        r_t = mu x dt + sigma x sqrt(dt) x Z,   Z ~ N(0, 1),   dt = 1 / p
        S_(t+dt) = S_t x (1 + r_t)

    Args:
        mu (float): Rendimiento esperado anual (aritmético, ver rendimiento_aritmetico_anualizado).
        sigma (float): Volatilidad anual.
        n_escenarios (int, opcional): Número de escenarios a simular. Por defecto 1,000.
        anios (float, opcional): Horizonte de la simulación en años. Por defecto 3.
        p (int, opcional): Pasos por año (12 mensual, 52 semanal, 252 diario). Por defecto 12.
        valor_inicial (float, opcional): Precio inicial S_0 (sólo si precios=True). Por defecto 100.
        semilla (int, opcional): Semilla para que los resultados sean reproducibles. Por defecto None.
        precios (bool, opcional): Si es True devuelve precios (con la fila 0 = S_0);
            si es False devuelve rendimientos. Por defecto False.

    Returns:
        pd.DataFrame: Una fila por período y una columna por escenario.
    """
    dt = 1 / p
    n_pasos = int(round(anios * p))
    generador = np.random.default_rng(semilla)
    Z = generador.standard_normal((n_pasos, n_escenarios))

    rend = pd.DataFrame(mu * dt + sigma * np.sqrt(dt) * Z, index=range(1, n_pasos + 1))
    rend.index.name = "Período"
    if not precios:
        return rend

    trayectorias = valor_inicial * (1 + rend).cumprod()
    fila_inicial = pd.DataFrame([[valor_inicial] * n_escenarios], index=[0], columns=rend.columns)
    trayectorias = pd.concat([fila_inicial, trayectorias])
    trayectorias.index.name = "Período"
    return trayectorias


def plot_simulacion(trayectorias, n_mostrar=50, titulo="Simulación de Monte Carlo"):
    """Grafica algunas trayectorias simuladas y el histograma de los valores finales.

    Args:
        trayectorias (pd.DataFrame): Valores simulados, una columna por escenario
            (por ejemplo, la salida de simular_gbm con precios=True).
        n_mostrar (int, opcional): Número de trayectorias a dibujar. Por defecto 50.
        titulo (str, opcional): Título principal del gráfico.

    Returns:
        tuple (matplotlib.figure.Figure, np.ndarray): Objetos Figure y Axes de Matplotlib.
    """
    finales = trayectorias.iloc[-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True, gridspec_kw={'width_ratios': [3, 1]})

    # Panel 1: trayectorias
    ax1.plot(trayectorias.iloc[:, :n_mostrar], color="#1f77b4", alpha=0.3, lw=0.8)
    ax1.axhline(trayectorias.iloc[0, 0], color="black", linestyle="--", lw=1, label="Valor inicial")
    ax1.set_title(titulo, fontsize=12, fontweight="bold")
    ax1.set_xlabel("Período")
    ax1.set_ylabel("Valor")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Panel 2: distribución de los valores finales
    ax2.hist(finales, bins=50, orientation="horizontal", color="#1f77b4", alpha=0.6)
    ax2.axhline(finales.mean(), color="#d62728", lw=1.5, label=f"Promedio: {finales.mean():,.2f}")
    ax2.axhline(finales.median(), color="#2ca02c", lw=1.5, linestyle="--", label=f"Mediana: {finales.median():,.2f}")
    ax2.set_title("Valores finales", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Frecuencia")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    return fig, (ax1, ax2)