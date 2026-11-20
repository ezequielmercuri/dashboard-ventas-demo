"""Tests de los cierres de caja: fórmula a mano, parser y comparación con la verdad."""

from datetime import date
from io import StringIO

import pandas as pd
import pytest

from scripts.generar_datos import generar, guardar, turno_de
from src.etl.caja import efectivo_vendido, leer_cierres_caja
from src.etl.comun import ArchivoInvalido
from src.etl.unificar import cargar_todo


def test_efectivo_vendido_caso_a_mano():
    # Arrancó con $20.000 de cambio, pagó $30.000 a un proveedor y al cerrar contó $135.000.
    # Ventas = 135.000 − 20.000 + 30.000 = 145.000
    assert efectivo_vendido(20_000, 135_000, 30_000) == 145_000


CAJA_CHICA = """fecha,turno,fondo_inicial,efectivo_final,salidas
2026-05-04,mañana,20000,80000,
2026-05-04,tarde,20000,50000,0
2026-05-04,tarde,20000,55000,0
"""


def test_parser_completa_salidas_y_usa_la_ultima_carga():
    c = leer_cierres_caja(StringIO(CAJA_CHICA))
    assert len(c) == 2
    assert c.loc[c["turno"] == "mañana", "efectivo_ventas"].item() == 60_000  # 80.000 − 20.000 + 0
    assert c.loc[c["turno"] == "tarde", "efectivo_ventas"].item() == 35_000  # vale la corrección


def test_parser_rechaza_turnos_desconocidos():
    with pytest.raises(ArchivoInvalido, match="noche"):
        leer_cierres_caja(StringIO("fecha,turno,fondo_inicial,efectivo_final,salidas\n2026-05-04,noche,0,0,0\n"))


# --- De punta a punta ------------------------------------------------------------

@pytest.fixture(scope="module")
def carpeta(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("data")
    guardar(generar(semilla=5, fecha_inicio=date(2026, 5, 1), meses=2), carpeta)
    return carpeta


@pytest.fixture(scope="module")
def comparacion(carpeta):
    """Une lo que dice la caja con el efectivo real de cada turno y los errores puestos a propósito."""
    caja = cargar_todo(carpeta)["caja"]
    ventas = pd.read_csv(carpeta / "verdad" / "ventas.csv", parse_dates=["fecha_hora"])
    efectivo = ventas[ventas["medio_pago"] == "efectivo"]
    real = (efectivo.assign(fecha=efectivo["fecha_hora"].dt.normalize(),
                            turno=efectivo["fecha_hora"].dt.hour.map(turno_de))
            .groupby(["fecha", "turno"])["monto_bruto"].sum().rename("real").reset_index())
    errores = pd.read_csv(carpeta / "verdad" / "errores_caja.csv", parse_dates=["fecha"])
    return (caja.merge(real, on=["fecha", "turno"], how="left").fillna({"real": 0})
            .merge(errores, on=["fecha", "turno"], how="left"))


def test_caja_coincide_con_la_verdad_salvo_faltantes(comparacion):
    sin_error = comparacion[comparacion["tipo"].isna()]
    assert (sin_error["efectivo_ventas"] == sin_error["real"]).all()


def test_faltantes_se_detectan(comparacion):
    faltantes = comparacion[comparacion["tipo"] == "faltante"]
    assert len(faltantes) >= 1
    assert ((faltantes["real"] - faltantes["efectivo_ventas"]) == faltantes["diferencia"]).all()


def test_turnos_sin_cargar_no_aparecen(carpeta, comparacion):
    errores = pd.read_csv(carpeta / "verdad" / "errores_caja.csv", parse_dates=["fecha"])
    sin_cargar = errores[errores["tipo"] == "sin_cargar"]
    assert len(sin_cargar) >= 1
    cargados = set(zip(comparacion["fecha"], comparacion["turno"]))
    assert not any((f, t) in cargados for f, t in zip(sin_cargar["fecha"], sin_cargar["turno"]))


def test_domingo_no_hay_caja_y_nunca_queda_negativa(comparacion):
    assert (comparacion["fecha"].dt.day_of_week != 6).all()
    assert (comparacion["efectivo_final"] >= 0).all()
