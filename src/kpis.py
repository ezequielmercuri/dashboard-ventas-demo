"""Fórmulas de los indicadores del dashboard.

Todas son funciones "puras": reciben tablas del esquema único (las que devuelve
`src.etl.unificar.cargar_todo`) y devuelven números o tablas. No leen archivos
ni dibujan nada, así se pueden testear con casos calculados a mano.

Para filtrar un período, primero se usa `filtrar(...)` y después la fórmula.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

# Turnos en los que el local abre: lunes a viernes mañana y tarde, sábado solo mañana.
HORARIO_CAJA = {0: ["mañana", "tarde"], 1: ["mañana", "tarde"], 2: ["mañana", "tarde"],
                3: ["mañana", "tarde"], 4: ["mañana", "tarde"], 5: ["mañana"], 6: []}
UMBRAL_SOBRESTOCK_DIAS = 60


# =============================================================================
# Períodos
# =============================================================================
def filtrar(df: pd.DataFrame, columna: str, desde, hasta) -> pd.DataFrame:
    """Filas con `columna` entre `desde` y `hasta`, ambos días incluidos."""
    desde = pd.Timestamp(desde)
    hasta_exclusivo = pd.Timestamp(hasta) + pd.Timedelta(days=1)
    return df[(df[columna] >= desde) & (df[columna] < hasta_exclusivo)]


def periodo_anterior(desde: date, hasta: date) -> tuple[date, date]:
    """El período de la misma cantidad de días que termina justo antes de `desde`."""
    dias = (hasta - desde).days + 1
    return desde - timedelta(days=dias), desde - timedelta(days=1)


def variacion_pct(actual: float, anterior: float) -> float | None:
    """Variación porcentual. None si no hay base de comparación."""
    if not anterior:
        return None
    return (actual - anterior) / anterior * 100


def variacion_real_pct(actual: float, anterior: float, ipc_actual: float, ipc_anterior: float) -> float | None:
    """Variación descontando la inflación: compara las ventas en pesos del mismo poder de compra."""
    if not anterior or not ipc_actual or not ipc_anterior:
        return None
    return variacion_pct(actual / ipc_actual, anterior / ipc_anterior)


# =============================================================================
# Página 1: Resumen
# =============================================================================
def resumen_ventas(ventas: pd.DataFrame, caja: pd.DataFrame | None) -> dict:
    """Ventas brutas (MP + efectivo), neto cobrado, costo de MP y ticket promedio.

    - El efectivo no tiene comisiones, así que su neto es igual al bruto.
    - El costo de MP se mide sobre lo que cobró MP (cuánto te cobra MP por cada $100 que pasan por él).
    - El ticket promedio solo usa los turnos de caja en los que se anotó la cantidad de ventas.
    """
    bruto_mp = ventas["monto_bruto"].sum()
    comisiones = ventas["comision"].sum()
    impuestos = ventas["impuestos"].sum()

    efectivo, ops_efectivo, efectivo_con_cantidad, turnos_sin_cantidad = 0.0, 0, 0.0, 0
    if caja is not None and not caja.empty:
        efectivo = caja["efectivo_ventas"].sum()
        if "cantidad_ventas" in caja.columns:
            con_cantidad = caja[caja["cantidad_ventas"].notna()]
            ops_efectivo = int(con_cantidad["cantidad_ventas"].sum())
            efectivo_con_cantidad = con_cantidad["efectivo_ventas"].sum()
            turnos_sin_cantidad = int(caja["cantidad_ventas"].isna().sum())

    operaciones = len(ventas) + ops_efectivo
    return {
        "bruto_mp": bruto_mp,
        "efectivo": efectivo,
        "bruto_total": bruto_mp + efectivo,
        "comisiones": comisiones,
        "impuestos": impuestos,
        "neto": bruto_mp - comisiones - impuestos + efectivo,
        "costo_mp_pct": (comisiones + impuestos) / bruto_mp * 100 if bruto_mp else None,
        "operaciones": operaciones,
        "ticket_promedio": (bruto_mp + efectivo_con_cantidad) / operaciones if operaciones else None,
        "turnos_sin_cantidad": turnos_sin_cantidad,
    }


def ventas_diarias(ventas: pd.DataFrame, caja: pd.DataFrame | None) -> pd.DataFrame:
    """Una fila por día (también los días sin ventas) con MP, efectivo, total y promedio móvil de 7 días."""
    mp = ventas.groupby(ventas["fecha_hora"].dt.normalize())["monto_bruto"].sum()
    ef = caja.groupby("fecha")["efectivo_ventas"].sum() if caja is not None else pd.Series(dtype=float)
    fechas = mp.index.union(ef.index)
    dias = pd.date_range(fechas.min(), fechas.max(), freq="D")

    df = pd.DataFrame({"mp": mp.reindex(dias, fill_value=0), "efectivo": ef.reindex(dias, fill_value=0)})
    df["total"] = df["mp"] + df["efectivo"]
    # Promedio móvil: el promedio de ese día y los 6 anteriores. Los primeros 6 días quedan vacíos.
    df["promedio_7d"] = df["total"].rolling(7).mean()
    return df.rename_axis("fecha").reset_index()


# =============================================================================
# Página 2: Cuándo vendo
# =============================================================================
def mapa_calor(ventas: pd.DataFrame) -> pd.DataFrame:
    """Facturación por día de la semana (0 = lunes) × hora.

    Solo usa ventas con hora exacta (MP y Tiendanube): el efectivo se registra por turno.
    """
    return ventas.pivot_table(index=ventas["fecha_hora"].dt.day_of_week, columns=ventas["fecha_hora"].dt.hour,
                              values="monto_bruto", aggfunc="sum", fill_value=0)


def promedio_por_dia_semana(diarias: pd.DataFrame) -> pd.Series:
    """Venta promedio de cada día de la semana (0 = lunes), a partir de `ventas_diarias`."""
    return diarias.groupby(diarias["fecha"].dt.day_of_week)["total"].mean()


def promedio_inicio_vs_resto(diarias: pd.DataFrame) -> dict:
    """Venta diaria promedio del 1 al 5 de cada mes (días de cobro) contra el resto del mes."""
    inicio = diarias["fecha"].dt.day <= 5
    return {"del_1_al_5": diarias.loc[inicio, "total"].mean(), "resto": diarias.loc[~inicio, "total"].mean()}


def mezcla_medios(ventas: pd.DataFrame, caja: pd.DataFrame | None) -> pd.DataFrame:
    """Cuánto entra por cada medio de pago, qué % del total es y cuánto cuesta cobrar con él."""
    por_medio = ventas.assign(costo=ventas["comision"].fillna(0) + ventas["impuestos"].fillna(0)) \
        .groupby("medio_pago")[["monto_bruto", "costo"]].sum()
    if caja is not None and not caja.empty:
        por_medio.loc["efectivo"] = [caja["efectivo_ventas"].sum(), 0.0]
    por_medio = por_medio.rename(columns={"monto_bruto": "monto"})
    por_medio["pct"] = por_medio["monto"] / por_medio["monto"].sum() * 100
    por_medio["costo_pct"] = por_medio["costo"] / por_medio["monto"] * 100
    return por_medio.sort_values("monto", ascending=False).rename_axis("medio_pago").reset_index()


# =============================================================================
# Página 3: Productos
# =============================================================================
def ventas_por_producto(unidades: pd.DataFrame, productos: pd.DataFrame) -> pd.DataFrame:
    """Unidades, facturación y margen bruto por producto.

    Las unidades salen de los conteos de stock. La facturación se estima con el
    precio de venta ACTUAL, así que en períodos viejos queda un poco inflada
    (no cambia el orden del ranking ni el ABC).
    """
    vendidas = unidades.groupby("sku")["unidades"].sum()
    df = productos[["sku", "nombre", "categoria", "precio", "costo"]].copy()
    df["unidades"] = df["sku"].map(vendidas).fillna(0).astype(int)
    df["facturacion"] = df["unidades"] * df["precio"]
    df["margen"] = df["unidades"] * (df["precio"] - df["costo"])
    df["margen_pct"] = np.where(df["facturacion"] > 0, df["margen"] / df["facturacion"] * 100, np.nan)
    return df.sort_values("facturacion", ascending=False, ignore_index=True)


def clasificar_abc(facturacion: pd.Series, corte_a: float = 0.80, corte_b: float = 0.95) -> pd.Series:
    """Clase A, B o C de cada producto.

    Se ordenan de mayor a menor facturación. Un producto es A si lo que facturaron
    los productos ANTERIORES a él todavía no llega al 80% del total; B si no llega al 95%;
    si no, C. Así el producto que cruza el 80% queda en A.
    """
    total = facturacion.sum()
    if total == 0:
        return pd.Series("C", index=facturacion.index)
    ordenada = facturacion.sort_values(ascending=False, kind="stable")
    acumulado_previo = (ordenada.cumsum() - ordenada) / total
    clase = pd.Series(np.select([acumulado_previo < corte_a, acumulado_previo < corte_b], ["A", "B"], "C"),
                      index=ordenada.index)
    clase[ordenada == 0] = "C"
    return clase.reindex(facturacion.index)


def productos_sin_ventas(unidades: pd.DataFrame, productos: pd.DataFrame, dias: int = 30) -> pd.DataFrame:
    """Productos que no vendieron nada en los conteos de los últimos `dias` días."""
    fin = unidades["hasta"].max()
    recientes = unidades[unidades["desde"] >= fin - pd.Timedelta(days=dias)]
    vendidas = recientes.groupby("sku")["unidades"].sum()
    sin = productos[productos["sku"].map(vendidas).fillna(0) <= 0]
    return sin[["sku", "nombre", "categoria", "stock_actual"]].reset_index(drop=True)


def margen_por_categoria(por_producto: pd.DataFrame) -> pd.DataFrame:
    """Suma facturación y margen de `ventas_por_producto` por categoría."""
    df = por_producto.groupby("categoria")[["facturacion", "margen"]].sum()
    df["margen_pct"] = df["margen"] / df["facturacion"] * 100
    return df.sort_values("margen", ascending=False).reset_index()


# =============================================================================
# Página 4: Stock
# =============================================================================
def venta_diaria_promedio(unidades: pd.DataFrame, dias: int = 28) -> pd.Series:
    """Unidades por día de cada producto, con los conteos de los últimos `dias` días.

    Se divide por los días que realmente cubren esos conteos (si el último conteo
    no cae justo, no se inventan días).
    """
    fin = unidades["hasta"].max()
    recientes = unidades[unidades["desde"] >= fin - pd.Timedelta(days=dias)]
    if recientes.empty:
        return pd.Series(dtype=float)
    dias_cubiertos = (fin - recientes["desde"].min()).days
    return recientes.groupby("sku")["unidades"].sum() / dias_cubiertos


def estado_stock(productos: pd.DataFrame, unidades: pd.DataFrame, dias: int = 28,
                 umbral_sobrestock: int = UMBRAL_SOBRESTOCK_DIAS) -> pd.DataFrame:
    """Cobertura, alerta de quiebre, sobrestock y capital inmovilizado de cada producto.

    - Días de cobertura = stock actual / venta diaria promedio.
    - Quiebre: la cobertura no llega a los días que tarda en reponerse.
    - Sobrestock: cobertura de más de `umbral_sobrestock` días (o stock que no se vende).
    """
    df = productos[["sku", "nombre", "categoria", "stock_actual", "costo", "dias_reposicion"]].copy()
    df["venta_diaria"] = df["sku"].map(venta_diaria_promedio(unidades, dias)).fillna(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cobertura = df["stock_actual"] / df["venta_diaria"]
    # Sin stock → 0 días. Con stock pero sin ventas → infinito (no se termina nunca).
    df["dias_cobertura"] = np.where(df["stock_actual"] <= 0, 0.0, cobertura)
    df["quiebre"] = df["dias_cobertura"] < df["dias_reposicion"]
    df["sobrestock"] = df["dias_cobertura"] > umbral_sobrestock
    df["capital_inmovilizado"] = df["stock_actual"] * df["costo"]
    return df.sort_values("dias_cobertura", ignore_index=True)


def rotacion(unidades: pd.DataFrame, conteos: pd.DataFrame, productos: pd.DataFrame) -> float | None:
    """Rotación del inventario = costo de lo vendido / inventario promedio (valorizado al costo).

    Una rotación de 2 en un mes quiere decir que el stock "se vendió entero" dos veces.
    """
    costo = productos.set_index("sku")["costo"]
    costo_vendido = (unidades["unidades"] * unidades["sku"].map(costo)).sum()
    valor_por_conteo = (conteos["stock"] * conteos["sku"].map(costo)).groupby(conteos["fecha"]).sum()
    inventario_promedio = valor_por_conteo.mean()
    return costo_vendido / inventario_promedio if inventario_promedio else None


def sell_through(unidades: pd.DataFrame, conteos: pd.DataFrame, compras: pd.DataFrame) -> pd.Series:
    """% de la mercadería disponible que se vendió, por producto.

    sell-through = vendido / (stock al inicio + recibido) × 100

    El plan decía vendido / recibido; esa versión da más de 100% cuando se vende
    stock que ya estaba, así que se suma el stock inicial.
    """
    inicial = conteos.sort_values("fecha").groupby("sku")["stock"].first()
    recibido = compras.groupby("sku")["cantidad"].sum() if compras is not None else pd.Series(dtype=float)
    vendido = unidades.groupby("sku")["unidades"].sum()
    disponible = inicial.add(recibido, fill_value=0)
    vendido = vendido.reindex(disponible.index, fill_value=0)
    return (vendido / disponible.replace(0, np.nan) * 100).rename("sell_through")


# =============================================================================
# Control de caja
# =============================================================================
def turnos_sin_cargar(caja: pd.DataFrame, desde, hasta, horario: dict = HORARIO_CAJA) -> pd.DataFrame:
    """Turnos en los que el local abrió pero nadie cargó el cierre de caja."""
    esperados = [(dia, turno) for dia in pd.date_range(desde, hasta, freq="D")
                 for turno in horario[dia.day_of_week]]
    cargados = set(zip(caja["fecha"], caja["turno"]))
    faltan = [(d, t) for d, t in esperados if (d, t) not in cargados]
    return pd.DataFrame(faltan, columns=["fecha", "turno"])
