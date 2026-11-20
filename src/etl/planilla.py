"""Parsers de la planilla propia del comercio: productos, conteos de stock, compras e IPC."""

import pandas as pd

from src.etl.comun import validar_columnas


def leer_planilla_productos(archivo) -> pd.DataFrame:
    df = pd.read_csv(archivo)
    validar_columnas(df, ["sku", "nombre", "categoria", "costo", "stock_actual",
                          "stock_minimo", "dias_reposicion"], "la planilla de productos")
    return df


def leer_conteos(archivo) -> pd.DataFrame:
    """Conteos de stock: una fila por (fecha, sku) con las unidades contadas."""
    df = pd.read_csv(archivo, parse_dates=["fecha"])
    validar_columnas(df, ["fecha", "sku", "stock"], "los conteos de stock")
    return df


def leer_compras(archivo) -> pd.DataFrame:
    df = pd.read_csv(archivo, parse_dates=["fecha"])
    validar_columnas(df, ["fecha", "sku", "cantidad", "costo_unitario"], "las compras")
    return df


def leer_ipc(archivo) -> pd.DataFrame:
    df = pd.read_csv(archivo)
    validar_columnas(df, ["mes", "indice"], "el IPC")
    return df
