"""Parser de los cierres de caja por turno → efectivo vendido en cada turno."""

import pandas as pd

from src.etl.comun import ArchivoInvalido, validar_columnas

REQUERIDAS = ["fecha", "turno", "fondo_inicial", "efectivo_final", "salidas"]
TURNOS = {"mañana", "tarde"}


def efectivo_vendido(fondo_inicial: float, efectivo_final: float, salidas: float) -> float:
    """Lo que entró por ventas en efectivo durante el turno.

    En la caja quedó: fondo inicial + ventas − salidas. Despejando las ventas:
    ventas = efectivo final − fondo inicial + salidas
    """
    return efectivo_final - fondo_inicial + salidas


def leer_cierres_caja(archivo) -> pd.DataFrame:
    df = pd.read_csv(archivo, parse_dates=["fecha"])
    validar_columnas(df, REQUERIDAS, "los cierres de caja")

    turnos_raros = set(df["turno"]) - TURNOS
    if turnos_raros:
        raise ArchivoInvalido(f"Turnos desconocidos en los cierres de caja: {', '.join(map(str, turnos_raros))}")

    df["salidas"] = df["salidas"].fillna(0)
    # Si un turno se cargó dos veces, vale la última carga (la corrección).
    df = df.drop_duplicates(["fecha", "turno"], keep="last")
    df["efectivo_ventas"] = efectivo_vendido(df["fondo_inicial"], df["efectivo_final"], df["salidas"])

    columnas = ["fecha", "turno", "efectivo_ventas", "salidas", "fondo_inicial", "efectivo_final"]
    columnas += [c for c in ["responsable", "detalle_salidas", "cantidad_ventas"] if c in df.columns]
    return df[columnas].sort_values(["fecha", "turno"], ignore_index=True)
