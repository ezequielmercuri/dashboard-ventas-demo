"""Tests del generador: chequean que los datos falsos cumplan las reglas del plan."""

from datetime import date

import pandas as pd
import pytest

from scripts.generar_datos import (
    calcular_cargos,
    exportar_mercadopago,
    exportar_tiendanube_ventas,
    generar,
    precio_del_mes,
)


@pytest.fixture(scope="module")
def datos():
    # 3 meses alcanzan para testear y es más rápido que 6.
    return generar(semilla=1, fecha_inicio=date(2026, 5, 1), meses=3)


# --- Funciones chicas, con casos calculados a mano --------------------------

def test_cargos_credito_en_local():
    # 10.000 × 4,5% = 450 de comisión; + 21% de IVA = 544,50. IIBB: 10.000 × 2% = 200.
    assert calcular_cargos(10_000, "credito", "local") == (544.50, 200.0)


def test_cargos_online_usan_tasa_online():
    # 10.000 × 5,99% = 599; × 1,21 = 724,79. IIBB = 200.
    assert calcular_cargos(10_000, "debito", "online") == (724.79, 200.0)


def test_efectivo_no_tiene_cargos():
    assert calcular_cargos(10_000, "efectivo", "local") == (0.0, 0.0)


def test_precio_del_mes():
    # 1.000 × 1,025² = 1.050,625 → redondeado a $10 = 1.050
    assert precio_del_mes(1_000, 2) == 1_050
    assert precio_del_mes(1_000, 0) == 1_000


# --- Reglas del negocio simulado --------------------------------------------

def test_es_reproducible():
    a = generar(semilla=7, meses=1)["ventas"]
    b = generar(semilla=7, meses=1)["ventas"]
    pd.testing.assert_frame_equal(a, b)


def test_neto_es_bruto_menos_cargos(datos):
    v = datos["ventas"]
    assert ((v["monto_bruto"] - v["comision"] - v["impuestos"]).round(2) == v["monto_neto"]).all()


def test_bruto_coincide_con_los_items(datos):
    it = datos["venta_items"]
    total_items = (it["cantidad"] * it["precio_unitario"]).groupby(it["venta_id"]).sum()
    v = datos["ventas"].set_index("id")
    assert (total_items.reindex(v.index).round(2) == v["monto_bruto"]).all()


def test_sabado_vende_mas_y_domingo_cierra_el_local(datos):
    local = datos["ventas"].query("canal == 'local'")
    por_dia = local.groupby(local["fecha_hora"].dt.date)["monto_bruto"].sum()
    dia_semana = pd.to_datetime(por_dia.index).day_of_week
    assert por_dia[dia_semana == 5].mean() > por_dia[dia_semana < 5].mean()
    assert (dia_semana == 6).sum() == 0  # no hay ventas en el local los domingos


def test_pico_del_1_al_5(datos):
    local = datos["ventas"].query("canal == 'local'")
    local = local[local["fecha_hora"].dt.day_of_week < 5]  # comparamos solo días hábiles
    tickets = local.groupby(local["fecha_hora"].dt.date).size()
    dia_mes = pd.to_datetime(tickets.index).day
    assert tickets[dia_mes <= 5].mean() > tickets[dia_mes > 5].mean()


def test_local_vende_solo_en_horario(datos):
    local = datos["ventas"].query("canal == 'local'")
    horas = set(local["fecha_hora"].dt.hour)
    assert horas <= {9, 10, 11, 12, 13, 16, 17, 18, 19}


def test_stock_nunca_negativo_y_hay_quiebres(datos):
    p = datos["productos"]
    assert (p["stock_actual"] >= 0).all()
    assert len(p) == 150
    # Algún producto quedó por debajo de su stock mínimo (para que haya alertas).
    assert (p["stock_actual"] < p["stock_minimo"]).any()


def test_hay_productos_estancados(datos):
    v, it = datos["ventas"], datos["venta_items"]
    ultimos_30 = v.loc[v["fecha_hora"] >= v["fecha_hora"].max() - pd.Timedelta(days=30), "id"]
    vendidos = set(it.loc[it["venta_id"].isin(ultimos_30), "sku"])
    assert len(set(datos["productos"]["sku"]) - vendidos) >= 10


# --- Exportación ------------------------------------------------------------

def test_mercadopago_no_incluye_efectivo(datos):
    v = datos["ventas"]
    mp = exportar_mercadopago(v)
    assert len(mp) == (v["medio_pago"] != "efectivo").sum()
    assert mp["TRANSACTION_AMOUNT"].sum() == pytest.approx(v.loc[v["medio_pago"] != "efectivo", "monto_bruto"].sum())


def test_ordenes_tiendanube_coinciden_con_mercadopago(datos):
    # Cada venta online aparece en las DOS fuentes: el ETL va a tener que deduplicarla.
    v = datos["ventas"]
    tn = exportar_tiendanube_ventas(v, datos["venta_items"], datos["productos"])
    mp = exportar_mercadopago(v).set_index("SOURCE_ID")
    ordenes = tn.drop_duplicates("Número de orden")
    montos_mp = mp.loc[ordenes["Identificador de la transacción en el medio de pago"], "TRANSACTION_AMOUNT"]
    assert (montos_mp.to_numpy() == ordenes["Total"].to_numpy()).all()
