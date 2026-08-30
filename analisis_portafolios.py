
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