"""Parser del reporte de liquidaciones de MercadoPago → tabla `ventas`."""

import pandas as pd

from src.etl.comun import COLUMNAS_VENTAS, validar_columnas

REQUERIDAS = ["SOURCE_ID", "TRANSACTION_TYPE", "TRANSACTION_DATE", "PAYMENT_METHOD_TYPE",
              "BUSINESS_UNIT", "TRANSACTION_AMOUNT", "FEE_AMOUNT", "TAXES_AMOUNT", "SETTLEMENT_NET_AMOUNT"]
MEDIOS = {"debit_card": "debito", "credit_card": "credito", "account_money": "dinero_en_cuenta"}


def leer_mercadopago(archivo) -> pd.DataFrame:
    """Lee el CSV de MP y devuelve una fila por venta con el esquema único.

    `archivo` puede ser una ruta o un archivo abierto (como el que sube Streamlit).
    """
    df = pd.read_csv(archivo, dtype={"SOURCE_ID": str})
    validar_columnas(df, REQUERIDAS, "MercadoPago")

    # El reporte también trae devoluciones, contracargos, etc. Por ahora solo cobros.
    df = df[df["TRANSACTION_TYPE"] == "SETTLEMENT"]

    # Las fechas vienen con zona horaria (-03:00): las pasamos a hora de Argentina sin zona.
    fecha = (pd.to_datetime(df["TRANSACTION_DATE"], utc=True)
             .dt.tz_convert("America/Argentina/Buenos_Aires")
             .dt.tz_localize(None))

    ventas = pd.DataFrame({
        "id": df["SOURCE_ID"],
        "fecha_hora": fecha,
        "canal": df["BUSINESS_UNIT"].map(lambda u: "online" if u == "ONLINE_PAYMENTS" else "local"),
        "medio_pago": df["PAYMENT_METHOD_TYPE"].map(MEDIOS).fillna("otro"),
        "monto_bruto": df["TRANSACTION_AMOUNT"],
        # MP informa los descuentos en negativo; en nuestro esquema van en positivo.
        "comision": -df["FEE_AMOUNT"],
        "impuestos": -df["TAXES_AMOUNT"],
        "monto_neto": df["SETTLEMENT_NET_AMOUNT"],
        "fuente": "mercadopago",
    })
    return ventas[COLUMNAS_VENTAS].reset_index(drop=True)
