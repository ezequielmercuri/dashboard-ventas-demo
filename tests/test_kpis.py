"""Tests de src/kpis.py: cada fórmula con un caso chico calculado a mano."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import kpis

T = pd.Timestamp


def ventas_de(filas):
    """Arma una tabla `ventas` a partir de (fecha_hora, medio, bruto, comision, impuestos)."""
    df = pd.DataFrame(filas, columns=["fecha_hora", "medio_pago", "monto_bruto", "comision", "impuestos"])
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    df["monto_neto"] = df["monto_bruto"] - df["comision"] - df["impuestos"]
    return df


def caja_de(filas):
    """Arma una tabla `caja` a partir de (fecha, turno, efectivo_ventas, cantidad_ventas)."""
    df = pd.DataFrame(filas, columns=["fecha", "turno", "efectivo_ventas", "cantidad_ventas"])
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["cantidad_ventas"] = df["cantidad_ventas"].astype("Int64")
    return df


# =============================================================================
# Períodos
# =============================================================================

def test_filtrar_incluye_el_ultimo_dia_completo():
    df = pd.DataFrame({"f": pd.to_datetime(["2026-05-01 09:00", "2026-05-31 20:00", "2026-06-01 09:00"])})
    assert len(kpis.filtrar(df, "f", "2026-05-01", "2026-05-31")) == 2


def test_periodo_anterior():
    # Del 8 al 14 (7 días) → el anterior es del 1 al 7.
    assert kpis.periodo_anterior(date(2026, 5, 8), date(2026, 5, 14)) == (date(2026, 5, 1), date(2026, 5, 7))


def test_variacion_pct():
    assert kpis.variacion_pct(1_200, 1_000) == pytest.approx(20)  # (1200 − 1000) / 1000
    assert kpis.variacion_pct(800, 1_000) == pytest.approx(-20)
    assert kpis.variacion_pct(500, 0) is None


def test_variacion_real_descuenta_inflacion():
    # Vendí 1.100 contra 1.000 (+10%), pero el IPC pasó de 100 a 105.
    # En pesos de igual poder de compra: 1.100/105 = 10,476 contra 1.000/100 = 10 → +4,76%
    assert kpis.variacion_real_pct(1_100, 1_000, 105, 100) == pytest.approx(4.7619, abs=1e-3)


# =============================================================================
# Página 1: Resumen
# =============================================================================

def test_resumen_ventas_caso_a_mano():
    ventas = ventas_de([
        ("2026-05-04 10:00", "credito", 10_000, 544.5, 200),
        ("2026-05-04 11:00", "debito", 4_000, 72.6, 80),
    ])
    caja = caja_de([
        ("2026-05-04", "mañana", 6_000, 3),      # 3 ventas en efectivo
        ("2026-05-04", "tarde", 2_000, None),    # no anotó cuántas
    ])
    r = kpis.resumen_ventas(ventas, caja)
    assert r["bruto_total"] == 22_000                    # 10.000 + 4.000 + 6.000 + 2.000
    assert r["neto"] == pytest.approx(21_102.9)          # 22.000 − 544,5 − 200 − 72,6 − 80
    assert r["costo_mp_pct"] == pytest.approx(6.4079, abs=1e-3)  # 897,1 / 14.000 × 100
    assert r["operaciones"] == 5                         # 2 de MP + 3 en efectivo
    assert r["ticket_promedio"] == pytest.approx(4_000)  # (14.000 + 6.000) / 5; la tarde no cuenta
    assert r["turnos_sin_cantidad"] == 1


def test_resumen_sin_caja():
    r = kpis.resumen_ventas(ventas_de([("2026-05-04 10:00", "debito", 1_000, 10, 20)]), None)
    assert r["bruto_total"] == 1_000 and r["neto"] == 970 and r["ticket_promedio"] == 1_000


def test_ventas_diarias_completa_dias_y_promedio_movil():
    # Ventas de MP del día 1 al 8 de $100 por día, salvo el día 3 que no hubo nada.
    filas = [(f"2026-05-0{d} 10:00", "debito", 100, 0, 0) for d in range(1, 9) if d != 3]
    caja = caja_de([("2026-05-08", "mañana", 700, 7)])
    d = kpis.ventas_diarias(ventas_de(filas), caja)
    assert len(d) == 8                                   # aparece el día 3 en cero
    assert d.loc[d["fecha"] == T("2026-05-03"), "total"].item() == 0
    assert d["promedio_7d"].iloc[:6].isna().all()        # no hay 7 días todavía
    # Día 7: (100 + 100 + 0 + 100 + 100 + 100 + 100) / 7 = 600 / 7
    assert d["promedio_7d"].iloc[6] == pytest.approx(600 / 7)
    # Día 8: días 2 al 8 = (100 + 0 + 100 + 100 + 100 + 100 + 800) / 7 = 1.300 / 7
    assert d["promedio_7d"].iloc[7] == pytest.approx(1_300 / 7)


# =============================================================================
# Página 2: Cuándo vendo
# =============================================================================

def test_mapa_calor():
    ventas = ventas_de([
        ("2026-05-04 10:15", "debito", 100, 0, 0),   # lunes 10 h
        ("2026-05-04 10:45", "debito", 50, 0, 0),    # lunes 10 h
        ("2026-05-09 18:00", "debito", 300, 0, 0),   # sábado 18 h
    ])
    m = kpis.mapa_calor(ventas)
    assert m.loc[0, 10] == 150 and m.loc[5, 18] == 300 and m.loc[0, 18] == 0


def test_promedios_por_dia_semana_e_inicio_de_mes():
    # 14 días: el 4 y el 11 (lunes) venden 300; el resto 100. Además del 1 al 5 suman +50.
    dias = pd.date_range("2026-05-01", "2026-05-14")
    total = [(300 if d.day_of_week == 0 else 100) + (50 if d.day <= 5 else 0) for d in dias]
    diarias = pd.DataFrame({"fecha": dias, "total": total})
    por_dia = kpis.promedio_por_dia_semana(diarias)
    assert por_dia[0] == 325                     # lunes 4 (350) y lunes 11 (300) → 325
    assert por_dia[2] == 100                     # miércoles 6 y 13
    r = kpis.promedio_inicio_vs_resto(diarias)
    # Del 1 al 5: 150 + 150 + 150 + 350 + 150 = 950 / 5 = 190
    assert r["del_1_al_5"] == 190
    # Del 6 al 14: ocho días de 100 y el lunes 11 de 300 = 1.100 / 9
    assert r["resto"] == pytest.approx(1_100 / 9)


def test_mezcla_medios():
    ventas = ventas_de([
        ("2026-05-04 10:00", "credito", 6_000, 300, 120),
        ("2026-05-04 11:00", "debito", 2_000, 30, 40),
    ])
    m = kpis.mezcla_medios(ventas, caja_de([("2026-05-04", "mañana", 2_000, None)])).set_index("medio_pago")
    assert m.loc["credito", "pct"] == 60                    # 6.000 / 10.000
    assert m.loc["credito", "costo_pct"] == pytest.approx(7)  # 420 / 6.000
    assert m.loc["efectivo", "costo"] == 0 and m.loc["efectivo", "pct"] == 20


# =============================================================================
# Página 3: Productos
# =============================================================================

PRODUCTOS = pd.DataFrame({
    "sku": ["A", "B", "C"], "nombre": ["Almendras", "Avena", "Chía"], "categoria": ["Frutos", "Cereales", "Cereales"],
    "precio": [1_000, 400, 500], "costo": [600, 200, 300],
    "stock_actual": [10, 200, 30], "dias_reposicion": [15, 7, 7],
})


def unidades_de(filas):
    """(desde, hasta, sku, unidades)"""
    df = pd.DataFrame(filas, columns=["desde", "hasta", "sku", "unidades"])
    df[["desde", "hasta"]] = df[["desde", "hasta"]].apply(pd.to_datetime)
    return df


def test_ventas_por_producto_y_margen():
    u = unidades_de([("2026-05-01", "2026-05-08", "A", 5), ("2026-05-08", "2026-05-15", "A", 3),
                     ("2026-05-01", "2026-05-08", "B", 10)])
    p = kpis.ventas_por_producto(u, PRODUCTOS).set_index("sku")
    assert p.loc["A", "unidades"] == 8
    assert p.loc["A", "facturacion"] == 8_000            # 8 × 1.000
    assert p.loc["A", "margen"] == 3_200                 # 8 × (1.000 − 600)
    assert p.loc["A", "margen_pct"] == 40
    assert p.loc["C", "unidades"] == 0 and np.isnan(p.loc["C", "margen_pct"])
    cat = kpis.margen_por_categoria(p.reset_index()).set_index("categoria")
    assert cat.loc["Cereales", "margen"] == 2_000        # 10 × (400 − 200) + 0


def test_clasificar_abc():
    # Total 100. Acumulado ANTES de cada uno: 0, 50, 80, 90, 96.
    f = pd.Series([10, 50, 4, 30, 6, 0], index=["p3", "p1", "p5", "p2", "p4", "p6"])
    abc = kpis.clasificar_abc(f)
    assert abc.to_dict() == {"p3": "B", "p1": "A", "p5": "C", "p2": "A", "p4": "B", "p6": "C"}


def test_productos_sin_ventas():
    u = unidades_de([
        ("2026-04-01", "2026-04-08", "C", 4),   # C vendió, pero hace más de 30 días
        ("2026-04-25", "2026-05-02", "A", 2),
        ("2026-05-02", "2026-05-09", "B", 1),
        ("2026-05-02", "2026-05-09", "C", 0),
    ])
    sin = kpis.productos_sin_ventas(u, PRODUCTOS, dias=30)
    assert list(sin["sku"]) == ["C"]


# =============================================================================
# Página 4: Stock
# =============================================================================

def test_estado_stock_caso_a_mano():
    # 4 conteos semanales = 28 días. A vendió 28 (1 por día), B 56 (2 por día), C nada.
    semanas = [("2026-05-02", "2026-05-09"), ("2026-05-09", "2026-05-16"),
               ("2026-05-16", "2026-05-23"), ("2026-05-23", "2026-05-30")]
    u = unidades_de([(d, h, "A", 7) for d, h in semanas] + [(d, h, "B", 14) for d, h in semanas])
    e = kpis.estado_stock(PRODUCTOS, u).set_index("sku")
    assert e.loc["A", "venta_diaria"] == 1
    assert e.loc["A", "dias_cobertura"] == 10            # 10 / 1
    assert e.loc["A", "quiebre"]                         # 10 días < 15 de reposición
    assert e.loc["B", "dias_cobertura"] == 100           # 200 / 2
    assert e.loc["B", "sobrestock"] and not e.loc["B", "quiebre"]
    assert np.isinf(e.loc["C", "dias_cobertura"]) and e.loc["C", "sobrestock"]
    assert e.loc["A", "capital_inmovilizado"] == 6_000   # 10 × 600


def test_sin_stock_es_quiebre():
    p = PRODUCTOS.assign(stock_actual=[0, 200, 30])
    u = unidades_de([("2026-05-02", "2026-05-30", "A", 28)])
    assert kpis.estado_stock(p, u).set_index("sku").loc["A", "quiebre"]


def test_rotacion():
    # Producto A (costo 600): conteos de 10, 6 y 8 → promedio 8 → inventario de $4.800.
    # Vendió 12 → costo de lo vendido $7.200 → rotación 7.200 / 4.800 = 1,5
    conteos = pd.DataFrame({"fecha": pd.to_datetime(["2026-05-01", "2026-05-08", "2026-05-15"]),
                            "sku": "A", "stock": [10, 6, 8]})
    u = unidades_de([("2026-05-01", "2026-05-15", "A", 12)])
    assert kpis.rotacion(u, conteos, PRODUCTOS) == pytest.approx(1.5)


def test_sell_through():
    # A arrancó con 10, recibió 40 y vendió 30 → 30 / 50 = 60%
    conteos = pd.DataFrame({"fecha": pd.to_datetime(["2026-05-01", "2026-05-31"]), "sku": "A", "stock": [10, 20]})
    compras = pd.DataFrame({"fecha": pd.to_datetime(["2026-05-10"]), "sku": ["A"], "cantidad": [40]})
    u = unidades_de([("2026-05-01", "2026-05-31", "A", 30)])
    assert kpis.sell_through(u, conteos, compras)["A"] == pytest.approx(60)


# =============================================================================
# Control de caja
# =============================================================================

def test_turnos_sin_cargar():
    # Semana del lunes 4 al domingo 10 de mayo: 5 × 2 + 1 (sábado) = 11 turnos esperados.
    cargados = [(f"2026-05-0{d}", t) for d in range(4, 9) for t in ("mañana", "tarde")]
    cargados.remove(("2026-05-06", "tarde"))
    caja = caja_de([(f, t, 1_000, None) for f, t in cargados])
    faltan = kpis.turnos_sin_cargar(caja, "2026-05-04", "2026-05-10")
    assert list(zip(faltan["fecha"], faltan["turno"])) == [(T("2026-05-06"), "tarde"), (T("2026-05-09"), "mañana")]
