"""Tests del ETL: casos chicos armados a mano + comparación contra data/verdad."""

from datetime import date
from io import StringIO

import pandas as pd
import pytest

from scripts.generar_datos import generar, guardar
from src.etl.comun import ArchivoInvalido
from src.etl.mercadopago import leer_mercadopago
from src.etl.tiendanube import leer_tiendanube_ventas
from src.etl.unificar import cargar_todo, unidades_por_conteo, unir_ventas

# --- MercadoPago --------------------------------------------------------------

MP_CHICO = """SOURCE_ID,TRANSACTION_TYPE,TRANSACTION_DATE,PAYMENT_METHOD_TYPE,BUSINESS_UNIT,TRANSACTION_AMOUNT,FEE_AMOUNT,TAXES_AMOUNT,SETTLEMENT_NET_AMOUNT
111,SETTLEMENT,2026-05-01T09:30:00.000-03:00,credit_card,POINT,10000,-544.5,-200,9255.5
222,SETTLEMENT,2026-05-01T22:10:00.000-03:00,account_money,ONLINE_PAYMENTS,5000,-362.4,-100,4537.6
333,REFUND,2026-05-02T10:00:00.000-03:00,credit_card,POINT,-10000,0,0,-10000
"""


def test_mercadopago_convierte_al_esquema_unico():
    v = leer_mercadopago(StringIO(MP_CHICO))
    assert list(v["id"]) == ["111", "222"]  # la devolución no es una venta
    assert v.loc[0, "fecha_hora"] == pd.Timestamp("2026-05-01 09:30:00")  # hora local, sin zona
    assert list(v["canal"]) == ["local", "online"]
    assert list(v["medio_pago"]) == ["credito", "dinero_en_cuenta"]
    assert v.loc[0, "comision"] == 544.5 and v.loc[0, "impuestos"] == 200  # en positivo
    # neto = 10.000 − 544,50 − 200 = 9.255,50
    assert v.loc[0, "monto_neto"] == 9255.5


def test_mercadopago_avisa_si_faltan_columnas():
    with pytest.raises(ArchivoInvalido, match="FEE_AMOUNT"):
        leer_mercadopago(StringIO("SOURCE_ID,TRANSACTION_TYPE\n1,SETTLEMENT\n"))


# --- Tiendanube ---------------------------------------------------------------

TN_CHICO = """Número de orden;Fecha;Estado del pago;Total;Medio de pago;Identificador de la transacción en el medio de pago;Precio del producto;Cantidad del producto;SKU
1001;01/05/2026 22:10:00;Pagado;5000;Mercado Pago - Dinero en cuenta;222;1500;2;DIE-0001
1001;01/05/2026 22:10:00;Pagado;5000;Mercado Pago - Dinero en cuenta;222;2000;1;DIE-0002
1002;02/05/2026 20:00:00;Pagado;3000;Transferencia;;3000;1;DIE-0003
1003;02/05/2026 21:00:00;Pendiente;9999;Mercado Pago - Crédito;444;9999;1;DIE-0004
"""


def test_tiendanube_agrupa_ordenes_e_items():
    ventas, items = leer_tiendanube_ventas(StringIO(TN_CHICO))
    assert list(ventas["id"]) == ["222", "TN-1002"]  # la 1003 no está pagada
    assert list(ventas["medio_pago"]) == ["dinero_en_cuenta", "otro"]
    assert len(items) == 3
    # La orden 1001: 2 × 1.500 + 1 × 2.000 = 5.000, igual al total
    o1 = items[items["venta_id"] == "222"]
    assert (o1["cantidad"] * o1["precio_unitario"]).sum() == ventas.loc[0, "monto_bruto"] == 5000


def test_unir_no_duplica_ventas_online():
    mp = leer_mercadopago(StringIO(MP_CHICO))
    tn, _ = leer_tiendanube_ventas(StringIO(TN_CHICO))
    v = unir_ventas(mp, tn)
    # 111 y 222 de MP + TN-1002 (solo está en Tiendanube). La 222 no se repite.
    assert sorted(v["id"]) == ["111", "222", "TN-1002"]
    assert v.loc[v["id"] == "222", "fuente"].item() == "mercadopago"


# --- Unidades por conteo de stock ----------------------------------------------

def test_unidades_por_conteo_caso_a_mano():
    f = pd.to_datetime
    conteos = pd.DataFrame({
        "fecha": f(["2026-05-01", "2026-05-08", "2026-05-15"] * 2),
        "sku": ["A"] * 3 + ["B"] * 3,
        "stock": [10, 8, 3, 4, 4, 0],
    })
    compras = pd.DataFrame({
        "fecha": f(["2026-05-08", "2026-05-10"]),  # la primera llega el mismo día del conteo
        "sku": ["A", "A"], "cantidad": [5, 2], "costo_unitario": [100, 100],
    })
    u = unidades_por_conteo(conteos, compras).set_index(["sku", "hasta"])["unidades"]
    assert u["A", f("2026-05-08")] == 7  # 10 + 5 − 8
    assert u["A", f("2026-05-15")] == 7  # 8 + 2 − 3
    assert u["B", f("2026-05-08")] == 0  # 4 + 0 − 4
    assert u["B", f("2026-05-15")] == 4  # 4 + 0 − 0


# --- De punta a punta: generador → CSV → ETL → comparar con la verdad -------------

@pytest.fixture(scope="module")
def carpeta(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("data")
    guardar(generar(semilla=3, fecha_inicio=date(2026, 5, 1), meses=2), carpeta)
    return carpeta


@pytest.fixture(scope="module")
def etl(carpeta):
    return cargar_todo(carpeta)


@pytest.fixture(scope="module")
def verdad(carpeta):
    ventas = pd.read_csv(carpeta / "verdad" / "ventas.csv", dtype={"id": str}, parse_dates=["fecha_hora"])
    items = pd.read_csv(carpeta / "verdad" / "venta_items.csv", dtype={"venta_id": str})
    return ventas, items


def test_ventas_coinciden_con_la_verdad_salvo_efectivo(etl, verdad):
    esperado = verdad[0].query("medio_pago != 'efectivo'").sort_values("id", ignore_index=True)
    obtenido = etl["ventas"].sort_values("id", ignore_index=True)
    pd.testing.assert_frame_equal(obtenido, esperado, check_dtype=False)


def test_items_coinciden_con_las_ventas_online(etl, verdad):
    ventas, items = verdad
    online = ventas.loc[ventas["canal"] == "online", "id"]
    esperado = items[items["venta_id"].isin(online)].sort_values(["venta_id", "sku"], ignore_index=True)
    obtenido = etl["venta_items"].sort_values(["venta_id", "sku"], ignore_index=True)
    pd.testing.assert_frame_equal(obtenido, esperado, check_dtype=False)


def test_conteos_recuperan_todo_lo_vendido(etl, verdad):
    # Los conteos ven local + online, aunque MP y Tiendanube no tengan el detalle del local.
    vendido_real = verdad[1].groupby("sku")["cantidad"].sum()
    por_conteo = etl["unidades_semana"].groupby("sku")["unidades"].sum()
    pd.testing.assert_series_equal(
        por_conteo.reindex(vendido_real.index, fill_value=0), vendido_real, check_names=False, check_dtype=False
    )


def test_productos_tienen_precio_de_tiendanube(etl):
    assert etl["productos"]["precio"].notna().all()


def test_faltan_archivos_y_no_explota(tmp_path):
    datos = cargar_todo(tmp_path)  # carpeta vacía
    assert all(v is None for v in datos.values())
