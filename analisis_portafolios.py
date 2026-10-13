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
FUNCIONES LAB 3
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
