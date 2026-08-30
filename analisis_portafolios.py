import pandas as pd
import matplotlib.pyplot as plt


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