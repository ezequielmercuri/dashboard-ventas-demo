"""Cosas que usan todos los parsers: el esquema único y la validación de columnas."""

import pandas as pd

# Columnas de la tabla `ventas` del esquema único (ver docs/PLAN.md, sección 5).
COLUMNAS_VENTAS = ["id", "fecha_hora", "canal", "medio_pago", "monto_bruto",
                   "comision", "impuestos", "monto_neto", "fuente"]
COLUMNAS_ITEMS = ["venta_id", "sku", "cantidad", "precio_unitario"]


class ArchivoInvalido(ValueError):
    """El archivo no tiene el formato esperado (por ejemplo, el dueño subió otro reporte)."""


def validar_columnas(df: pd.DataFrame, requeridas: list[str], fuente: str) -> None:
    """Corta con un mensaje claro si al archivo le faltan columnas."""
    faltan = [c for c in requeridas if c not in df.columns]
    if faltan:
        raise ArchivoInvalido(
            f"El archivo de {fuente} no tiene las columnas: {', '.join(faltan)}. "
            "¿Es el reporte correcto?"
        )
