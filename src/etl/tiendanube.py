"""Parsers de los exports de Tiendanube (ventas online y catálogo de productos)."""

import pandas as pd

from src.etl.comun import COLUMNAS_ITEMS, COLUMNAS_VENTAS, validar_columnas

COL_ORDEN = "Número de orden"
COL_TRANSACCION = "Identificador de la transacción en el medio de pago"
REQUERIDAS_VENTAS = [COL_ORDEN, "Fecha", "Estado del pago", "Total", "Medio de pago",
                     COL_TRANSACCION, "Precio del producto", "Cantidad del producto", "SKU"]
REQUERIDAS_PRODUCTOS = ["Nombre", "Categorías", "Precio", "Stock", "SKU"]


def _medio(texto: str) -> str:
    """'Mercado Pago - Crédito' → 'credito', etc."""
    texto = str(texto).lower()
    if "crédito" in texto or "credito" in texto:
        return "credito"
    if "débito" in texto or "debito" in texto:
        return "debito"
    if "dinero en cuenta" in texto:
        return "dinero_en_cuenta"
    return "otro"


def leer_tiendanube_ventas(archivo) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee el export de ventas (una fila por producto) y devuelve (ventas, venta_items).

    El id de cada venta es el de la transacción de MercadoPago, para poder
    cruzarla con el reporte de MP y no contarla dos veces.
    """
    df = pd.read_csv(archivo, sep=";", dtype={COL_ORDEN: str, COL_TRANSACCION: str})
    validar_columnas(df, REQUERIDAS_VENTAS, "Tiendanube (ventas)")
    df = df[df["Estado del pago"] == "Pagado"]

    # Si una orden no tiene transacción (pago por transferencia, por ejemplo), usamos el nro de orden.
    df["venta_id"] = df[COL_TRANSACCION].fillna("TN-" + df[COL_ORDEN])

    items = pd.DataFrame({
        "venta_id": df["venta_id"],
        "sku": df["SKU"],
        "cantidad": df["Cantidad del producto"],
        "precio_unitario": df["Precio del producto"],
    })[COLUMNAS_ITEMS]

    ordenes = df.drop_duplicates("venta_id")
    ventas = pd.DataFrame({
        "id": ordenes["venta_id"],
        "fecha_hora": pd.to_datetime(ordenes["Fecha"], format="%d/%m/%Y %H:%M:%S"),
        "canal": "online",
        "medio_pago": ordenes["Medio de pago"].map(_medio),
        "monto_bruto": ordenes["Total"],
        # Tiendanube no sabe cuánto cobró MP: eso lo completa el reporte de MP.
        "comision": float("nan"),
        "impuestos": float("nan"),
        "monto_neto": float("nan"),
        "fuente": "tiendanube",
    })[COLUMNAS_VENTAS]
    return ventas.reset_index(drop=True), items.reset_index(drop=True)


def leer_tiendanube_productos(archivo) -> pd.DataFrame:
    """Catálogo online: sku, nombre, categoría, precio de venta y stock."""
    df = pd.read_csv(archivo, sep=";")
    validar_columnas(df, REQUERIDAS_PRODUCTOS, "Tiendanube (productos)")
    return pd.DataFrame({
        "sku": df["SKU"],
        "nombre": df["Nombre"],
        "categoria": df["Categorías"],
        "precio": df["Precio"],
        "stock_actual": df["Stock"],
    })
