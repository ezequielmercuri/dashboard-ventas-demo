"""Junta todas las fuentes en el esquema único.

- Ventas: MercadoPago es la fuente principal (trae comisiones e impuestos).
  Las órdenes de Tiendanube que ya están en MP no se vuelven a sumar.
- Unidades vendidas por producto: salen de los conteos de stock semanales,
  porque MP no dice qué productos se vendieron en el local.
"""

from pathlib import Path

import pandas as pd

from src.etl.mercadopago import leer_mercadopago
from src.etl.planilla import leer_compras, leer_conteos, leer_ipc, leer_planilla_productos
from src.etl.tiendanube import leer_tiendanube_productos, leer_tiendanube_ventas


def unir_ventas(ventas_mp: pd.DataFrame | None, ventas_tn: pd.DataFrame | None) -> pd.DataFrame | None:
    """Une las ventas de MP y Tiendanube sin contar dos veces las órdenes online."""
    if ventas_mp is None:
        return ventas_tn
    if ventas_tn is None:
        return ventas_mp
    solo_tn = ventas_tn[~ventas_tn["id"].isin(ventas_mp["id"])]
    if solo_tn.empty:
        return ventas_mp
    return pd.concat([ventas_mp, solo_tn], ignore_index=True).sort_values("fecha_hora", ignore_index=True)


def unidades_por_conteo(conteos: pd.DataFrame, compras: pd.DataFrame | None) -> pd.DataFrame:
    """Unidades vendidas de cada producto entre dos conteos seguidos.

    vendido = stock del conteo anterior + compras recibidas en el medio − stock del conteo actual

    Una compra que llega el mismo día del conteo cuenta para ese conteo
    (la mercadería llega a la mañana y se cuenta al cierre).
    """
    c = conteos.sort_values(["sku", "fecha"]).copy()
    c["desde"] = c.groupby("sku")["fecha"].shift()
    c["stock_anterior"] = c.groupby("sku")["stock"].shift()
    c = c.dropna(subset=["desde"])  # el primer conteo de cada producto no tiene "anterior"

    if compras is not None and not compras.empty:
        # A cada compra le asignamos el primer conteo que ocurre en o después de su fecha.
        fechas_conteo = c[["sku", "fecha"]].rename(columns={"fecha": "fecha_conteo"})
        asignadas = pd.merge_asof(
            compras.sort_values("fecha"), fechas_conteo.sort_values("fecha_conteo"),
            left_on="fecha", right_on="fecha_conteo", by="sku", direction="forward",
        )
        recibido = asignadas.groupby(["sku", "fecha_conteo"])["cantidad"].sum()
        c = c.merge(recibido.rename("recibido"), left_on=["sku", "fecha"], right_index=True, how="left")
    else:
        c["recibido"] = 0
    c["recibido"] = c["recibido"].fillna(0)

    # Si da negativo, hubo un error de conteo o una compra sin anotar: lo dejamos visible.
    c["unidades"] = (c["stock_anterior"] + c["recibido"] - c["stock"]).astype(int)
    return (c.rename(columns={"fecha": "hasta"})[["desde", "hasta", "sku", "unidades"]]
            .reset_index(drop=True))


def _si_existe(carpeta: Path, nombre: str, lector):
    ruta = carpeta / nombre
    return lector(ruta) if ruta.exists() else None


def cargar_todo(carpeta: str | Path) -> dict:
    """Lee todas las fuentes que haya en `carpeta`. Las que faltan quedan en None."""
    carpeta = Path(carpeta)
    ventas_mp = _si_existe(carpeta, "mercadopago_liquidaciones.csv", leer_mercadopago)
    tn = _si_existe(carpeta, "tiendanube_ventas.csv", leer_tiendanube_ventas)
    ventas_tn, items = tn if tn is not None else (None, None)

    productos = _si_existe(carpeta, "planilla_productos.csv", leer_planilla_productos)
    catalogo_tn = _si_existe(carpeta, "tiendanube_productos.csv", leer_tiendanube_productos)
    if productos is not None and catalogo_tn is not None:
        # La planilla manda en costo y stock; de Tiendanube sacamos el precio de venta.
        productos = productos.merge(catalogo_tn[["sku", "precio"]], on="sku", how="left")
    elif productos is None:
        productos = catalogo_tn

    conteos = _si_existe(carpeta, "conteos_stock.csv", leer_conteos)
    compras = _si_existe(carpeta, "compras.csv", leer_compras)

    return {
        "ventas": unir_ventas(ventas_mp, ventas_tn),
        "venta_items": items,
        "productos": productos,
        "conteos_stock": conteos,
        "compras": compras,
        "unidades_semana": unidades_por_conteo(conteos, compras) if conteos is not None else None,
        "ipc": _si_existe(carpeta, "ipc.csv", leer_ipc),
    }
