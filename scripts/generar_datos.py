"""Genera datos FALSOS de una dietética ficticia de La Plata.

Simula las ventas día por día durante N meses y exporta cada fuente con un
formato parecido al real (aproximado: hay que ajustarlo con exports reales):

  data/mercadopago_liquidaciones.csv  reporte de liquidaciones de MercadoPago
  data/tiendanube_ventas.csv          export de ventas online (una fila por producto)
  data/tiendanube_productos.csv       export del catálogo con stock
  data/planilla_productos.csv         planilla propia: costo, stock mínimo, reposición
  data/compras.csv                    reposiciones de mercadería
  data/ipc.csv                        índice de precios mensual (ficticio)
  data/verdad/ventas.csv              "la verdad" en el esquema único, para testear
  data/verdad/venta_items.csv         el ETL del Día 2 (incluye el efectivo, que no
                                      aparece en ninguna fuente)

Uso:  python -m scripts.generar_datos
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Configuración: todo lo "inventado" está acá arriba para poder tocarlo fácil.
# ---------------------------------------------------------------------------
SEMILLA = 42
FECHA_INICIO = date(2026, 5, 1)
MESES = 6
INFLACION_MENSUAL = 0.025  # suba de precios de lista por mes

TICKETS_LOCAL_POR_DIA = 55  # promedio de un día "normal" en el local
TICKETS_ONLINE_POR_DIA = 6
FACTOR_DIA_SEMANA = {0: 0.9, 1: 0.85, 2: 0.9, 3: 0.95, 4: 1.1, 5: 1.5, 6: 0.0}  # lun..dom (domingo cerrado)
FACTOR_INICIO_DE_MES = 1.3  # días 1 al 5: cobran sueldos y jubilaciones

# Horario del local (con siesta). Peso = qué tan probable es vender en esa hora.
HORAS_SEMANA = {9: 1, 10: 3, 11: 4, 12: 3, 16: 1, 17: 2, 18: 4, 19: 4}
HORAS_SABADO = {9: 2, 10: 4, 11: 5, 12: 4, 13: 2}
HORAS_ONLINE = {h: (3 if 20 <= h <= 23 else 1 if 8 <= h <= 19 else 0.3) for h in range(24)}

MEDIOS_LOCAL = {"efectivo": 0.25, "debito": 0.30, "credito": 0.20, "dinero_en_cuenta": 0.25}
MEDIOS_ONLINE = {"debito": 0.15, "credito": 0.60, "dinero_en_cuenta": 0.25}

# Costos de MercadoPago (valores APROXIMADOS, acreditación inmediata).
COMISION_LOCAL = {"debito": 0.015, "credito": 0.045, "dinero_en_cuenta": 0.008}
COMISION_ONLINE = 0.0599
IVA_SOBRE_COMISION = 0.21
RETENCION_IIBB = 0.02  # varía según el padrón de ARBA de cada comercio

CANTIDAD_A_CON_QUIEBRE = 4      # productos top con stock mínimo mal calculado
CANTIDAD_A_SIN_REPONER = 3      # productos top que no se reponen en las últimas semanas
DIAS_SIN_REPONER = 21
CANTIDAD_ESTANCADOS = 15        # productos que dejan de venderse al final
DIAS_ESTANCADOS = 45

CATALOGO = {
    "Frutos secos": ["Almendras", "Nueces mariposa", "Castañas de cajú", "Maní tostado sin sal",
                     "Mix de frutos secos", "Pasas de uva", "Avellanas"],
    "Cereales y semillas": ["Avena arrollada", "Granola", "Semillas de chía", "Semillas de lino",
                            "Semillas de girasol", "Quinoa", "Copos de maíz sin azúcar"],
    "Harinas": ["Harina de almendras", "Harina integral", "Harina de garbanzo", "Harina de arroz",
                "Premezcla sin TACC", "Harina de coco"],
    "Legumbres": ["Lentejas", "Garbanzos", "Porotos negros", "Arvejas partidas", "Porotos aduki"],
    "Infusiones": ["Yerba mate orgánica", "Yerba mate suave", "Té verde", "Té de hierbas", "Cacao amargo"],
    "Endulzantes": ["Miel pura", "Azúcar mascabo", "Stevia en polvo", "Azúcar de coco",
                    "Dulce de leche sin azúcar"],
    "Snacks": ["Barra de cereal", "Galletas de arroz", "Garrapiñada", "Chips de banana", "Mix tropical"],
    "Almacén natural": ["Aceite de coco", "Mantequilla de maní", "Tofu", "Fideos integrales",
                        "Arroz yamaní", "Polenta orgánica"],
    "Suplementos": ["Levadura nutricional", "Colágeno", "Proteína vegetal", "Spirulina"],
}
# Rango de precio por kg (en pesos, al inicio del período) para cada categoría.
PRECIO_KG = {
    "Frutos secos": (12000, 25000), "Cereales y semillas": (3000, 9000), "Harinas": (3000, 15000),
    "Legumbres": (2500, 5000), "Infusiones": (6000, 14000), "Endulzantes": (4000, 12000),
    "Snacks": (3000, 8000), "Almacén natural": (5000, 16000), "Suplementos": (15000, 40000),
}
PRESENTACIONES = [("250 g", 0.30), ("500 g", 0.55), ("1 kg", 1.0)]  # (nombre, factor de precio)


# ---------------------------------------------------------------------------
# Funciones chicas (testeables por separado)
# ---------------------------------------------------------------------------
def calcular_cargos(monto: float, medio_pago: str, canal: str) -> tuple[float, float]:
    """Devuelve (comision, impuestos) que MercadoPago descuenta de una venta.

    El efectivo no pasa por MercadoPago, así que no tiene cargos.
    La comisión lleva IVA encima; los impuestos son la retención de IIBB.
    """
    if medio_pago == "efectivo":
        return 0.0, 0.0
    tasa = COMISION_ONLINE if canal == "online" else COMISION_LOCAL[medio_pago]
    comision = round(monto * tasa * (1 + IVA_SOBRE_COMISION), 2)
    impuestos = round(monto * RETENCION_IIBB, 2)
    return comision, impuestos


def precio_del_mes(precio_base: float, meses_transcurridos: int) -> float:
    """Precio de lista después de aplicar la suba mensual, redondeado a $10."""
    return round(precio_base * (1 + INFLACION_MENSUAL) ** meses_transcurridos, -1)


def armar_catalogo(rng: np.random.Generator) -> pd.DataFrame:
    """Arma ~150 productos con precio, costo y popularidad."""
    filas = []
    for categoria, bases in CATALOGO.items():
        for base in bases:
            precio_kg = rng.uniform(*PRECIO_KG[categoria])
            for presentacion, factor in PRESENTACIONES:
                precio = round(precio_kg * factor, -1)
                filas.append({
                    "nombre": f"{base} {presentacion}",
                    "categoria": categoria,
                    "precio_base": precio,
                    "costo_base": round(precio * rng.uniform(0.55, 0.70), -1),
                    "dias_reposicion": int(rng.choice([3, 5, 7, 10, 15])),
                })
    catalogo = pd.DataFrame(filas)
    catalogo.insert(0, "sku", [f"DIE-{i:04d}" for i in range(1, len(catalogo) + 1)])

    # Popularidad tipo "ley de Zipf": pocos productos venden mucho, muchos venden poco.
    # Así aparece solo el patrón ABC (80/15/5).
    ranking = rng.permutation(len(catalogo)) + 1
    catalogo["ranking"] = ranking
    catalogo["peso"] = 1 / ranking ** 1.1
    return catalogo


def elegir_hora(rng: np.random.Generator, horas: dict) -> tuple[int, int]:
    """Elige hora y minuto según los pesos de cada hora."""
    opciones = list(horas.keys())
    pesos = np.array(list(horas.values()), dtype=float)
    hora = int(rng.choice(opciones, p=pesos / pesos.sum()))
    return hora, int(rng.integers(0, 60))


def elegir(rng: np.random.Generator, opciones: dict) -> str:
    return str(rng.choice(list(opciones.keys()), p=list(opciones.values())))


# ---------------------------------------------------------------------------
# Simulación principal
# ---------------------------------------------------------------------------
def generar(semilla: int = SEMILLA, fecha_inicio: date = FECHA_INICIO, meses: int = MESES) -> dict:
    """Simula el negocio y devuelve un diccionario de DataFrames (no escribe archivos)."""
    rng = np.random.default_rng(semilla)
    fake = Faker("es_AR")
    fake.seed_instance(semilla)

    catalogo = armar_catalogo(rng)
    n = len(catalogo)
    fecha_fin = (pd.Timestamp(fecha_inicio) + pd.DateOffset(months=meses)).date()
    dias = (fecha_fin - fecha_inicio).days

    # Venta diaria esperada de cada producto (unidades), para fijar stocks razonables.
    items_por_dia = (TICKETS_LOCAL_POR_DIA + TICKETS_ONLINE_POR_DIA) * 2.2 * 1.2
    venta_diaria = items_por_dia * catalogo["peso"] / catalogo["peso"].sum()

    # Productos especiales (elegidos entre los más vendidos / menos vendidos).
    top = catalogo.sort_values("ranking").index
    con_quiebre = list(top[:15][rng.permutation(15)[:CANTIDAD_A_CON_QUIEBRE]])
    sin_reponer = [i for i in top[:20] if i not in con_quiebre][:CANTIDAD_A_SIN_REPONER]
    estancados = list(top[-CANTIDAD_ESTANCADOS:])

    margen_seguridad = np.where(catalogo.index.isin(con_quiebre), 0.4, 1.3)
    stock_minimo = np.ceil(venta_diaria * catalogo["dias_reposicion"] * margen_seguridad).astype(int)
    stock_minimo = np.maximum(stock_minimo, 2)
    cantidad_pedido = np.maximum(np.ceil(venta_diaria * 30), 6).astype(int)
    stock = (stock_minimo + cantidad_pedido).to_numpy().copy()
    pedidos_pendientes: dict[int, tuple[date, int]] = {}  # índice -> (fecha de llegada, cantidad)

    ventas, items, compras = [], [], []
    id_mp = 98_000_000_000
    id_efectivo = 0
    nro_orden = 1000

    for d in range(dias):
        dia = fecha_inicio + timedelta(days=d)
        mes_idx = (dia.year - fecha_inicio.year) * 12 + dia.month - fecha_inicio.month

        # 1) Llegan los pedidos de reposición.
        for i, (llegada, cantidad) in list(pedidos_pendientes.items()):
            if llegada == dia:
                stock[i] += cantidad
                compras.append({
                    "fecha": dia, "sku": catalogo.at[i, "sku"], "cantidad": cantidad,
                    "costo_unitario": precio_del_mes(catalogo.at[i, "costo_base"], mes_idx),
                })
                del pedidos_pendientes[i]

        # 2) Cuántos tickets hay hoy.
        factor = FACTOR_DIA_SEMANA[dia.weekday()] * (FACTOR_INICIO_DE_MES if dia.day <= 5 else 1.0)
        n_local = rng.poisson(TICKETS_LOCAL_POR_DIA * factor)
        n_online = rng.poisson(TICKETS_ONLINE_POR_DIA * (FACTOR_INICIO_DE_MES if dia.day <= 5 else 1.0))
        canales = ["local"] * n_local + ["online"] * n_online

        pesos_hoy = catalogo["peso"].to_numpy().copy()
        if dias - d <= DIAS_ESTANCADOS:
            pesos_hoy[estancados] = 0.0

        # 3) Cada ticket: horario, productos, medio de pago y cargos.
        for canal in canales:
            if canal == "local":
                hora, minuto = elegir_hora(rng, HORAS_SABADO if dia.weekday() == 5 else HORAS_SEMANA)
                medio = elegir(rng, MEDIOS_LOCAL)
            else:
                hora, minuto = elegir_hora(rng, HORAS_ONLINE)
                medio = elegir(rng, MEDIOS_ONLINE)

            disponibles = pesos_hoy * (stock > 0)
            if disponibles.sum() == 0:
                continue
            n_items = min(1 + rng.poisson(1.2), int((disponibles > 0).sum()), 6)
            elegidos = rng.choice(n, size=n_items, replace=False, p=disponibles / disponibles.sum())

            if medio == "efectivo":
                id_efectivo += 1
                venta_id = f"EF-{id_efectivo:06d}"
            else:
                id_mp += int(rng.integers(1, 50))
                venta_id = str(id_mp)

            lineas = []
            for i in elegidos:
                cantidad = min(2 if rng.random() < 0.2 else 1, int(stock[i]))
                stock[i] -= cantidad
                lineas.append({
                    "venta_id": venta_id, "sku": catalogo.at[i, "sku"], "cantidad": cantidad,
                    "precio_unitario": precio_del_mes(catalogo.at[i, "precio_base"], mes_idx),
                })
            bruto = round(sum(l["cantidad"] * l["precio_unitario"] for l in lineas), 2)
            comision, impuestos = calcular_cargos(bruto, medio, canal)

            orden = None
            if canal == "online":
                nro_orden += 1
                orden = nro_orden
            ventas.append({
                "id": venta_id,
                "fecha_hora": datetime(dia.year, dia.month, dia.day, hora, minuto, int(rng.integers(0, 60))),
                "canal": canal, "medio_pago": medio, "monto_bruto": bruto,
                "comision": comision, "impuestos": impuestos,
                "monto_neto": round(bruto - comision - impuestos, 2),
                "fuente": "efectivo_sin_registro" if medio == "efectivo" else "mercadopago",
                "nro_orden_tiendanube": orden,
                "comprador": fake.name() if orden else None,
            })
            items.extend(lineas)

        # 4) A fin del día se piden reposiciones (salvo los que "se olvidaron" de reponer).
        for i in range(n):
            olvidado = i in sin_reponer and dias - d <= DIAS_SIN_REPONER
            if stock[i] <= stock_minimo.iloc[i] and i not in pedidos_pendientes and not olvidado:
                llegada = dia + timedelta(days=int(catalogo.at[i, "dias_reposicion"]))
                pedidos_pendientes[i] = (llegada, int(cantidad_pedido.iloc[i]))

    ventas = pd.DataFrame(ventas).sort_values("fecha_hora", ignore_index=True)
    items = pd.DataFrame(items)
    catalogo["stock_actual"] = stock
    catalogo["stock_minimo"] = stock_minimo
    catalogo["costo"] = [precio_del_mes(c, meses - 1) for c in catalogo["costo_base"]]
    catalogo["precio"] = [precio_del_mes(p, meses - 1) for p in catalogo["precio_base"]]

    return {
        "ventas": ventas,
        "venta_items": items,
        "productos": catalogo,
        "compras": pd.DataFrame(compras),
        "ipc": armar_ipc(rng, fecha_inicio, meses),
    }


def armar_ipc(rng: np.random.Generator, fecha_inicio: date, meses: int) -> pd.DataFrame:
    """IPC ficticio: arranca en 100 y sube cerca de la inflación configurada."""
    indice, filas = 100.0, []
    for m in range(meses):
        mes = (pd.Timestamp(fecha_inicio) + pd.DateOffset(months=m)).strftime("%Y-%m")
        filas.append({"mes": mes, "indice": round(indice, 2)})
        indice *= 1 + rng.normal(INFLACION_MENSUAL, 0.004)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Exportación con formato "de cada fuente"
# ---------------------------------------------------------------------------
TIPO_MP = {"debito": "debit_card", "credito": "credit_card", "dinero_en_cuenta": "account_money"}


def exportar_mercadopago(ventas: pd.DataFrame) -> pd.DataFrame:
    """Reporte de liquidaciones: solo lo que cobró MercadoPago (sin efectivo)."""
    mp = ventas[ventas["medio_pago"] != "efectivo"]
    fecha = mp["fecha_hora"].dt.strftime("%Y-%m-%dT%H:%M:%S.000-03:00")
    unidad = np.select(
        [mp["canal"] == "online", mp["medio_pago"] == "dinero_en_cuenta"], ["ONLINE_PAYMENTS", "QR"], "POINT"
    )
    return pd.DataFrame({
        "SOURCE_ID": mp["id"],
        "EXTERNAL_REFERENCE": mp["nro_orden_tiendanube"].astype("Int64"),
        "TRANSACTION_TYPE": "SETTLEMENT",
        "TRANSACTION_DATE": fecha,
        "SETTLEMENT_DATE": fecha,
        "PAYMENT_METHOD_TYPE": mp["medio_pago"].map(TIPO_MP),
        "BUSINESS_UNIT": unidad,
        "POS_ID": np.where(mp["canal"] == "local", "CAJA-1", ""),
        "TRANSACTION_AMOUNT": mp["monto_bruto"],
        "FEE_AMOUNT": -mp["comision"],
        "TAXES_AMOUNT": -mp["impuestos"],
        "SETTLEMENT_NET_AMOUNT": mp["monto_neto"],
        "INSTALLMENTS": 1,
    })


def exportar_tiendanube_ventas(ventas: pd.DataFrame, items: pd.DataFrame, productos: pd.DataFrame) -> pd.DataFrame:
    """Export de ventas online: una fila por producto, con los datos de la orden repetidos."""
    online = ventas[ventas["canal"] == "online"]
    detalle = items.merge(online, left_on="venta_id", right_on="id").merge(productos[["sku", "nombre"]], on="sku")
    medio = {"debito": "Mercado Pago - Débito", "credito": "Mercado Pago - Crédito",
             "dinero_en_cuenta": "Mercado Pago - Dinero en cuenta"}
    return pd.DataFrame({
        "Número de orden": detalle["nro_orden_tiendanube"].astype(int),
        "Fecha": detalle["fecha_hora"].dt.strftime("%d/%m/%Y %H:%M:%S"),
        "Estado de la orden": "Cerrada",
        "Estado del pago": "Pagado",
        "Nombre del comprador": detalle["comprador"],
        "Moneda": "ARS",
        "Subtotal de productos": detalle["monto_bruto"],
        "Descuento": 0,
        "Costo de envío": 0,
        "Total": detalle["monto_bruto"],
        "Medio de pago": detalle["medio_pago"].map(medio),
        "Identificador de la transacción en el medio de pago": detalle["venta_id"],
        "Nombre del producto": detalle["nombre"],
        "Precio del producto": detalle["precio_unitario"],
        "Cantidad del producto": detalle["cantidad"],
        "SKU": detalle["sku"],
    }).sort_values(["Número de orden", "SKU"])


def exportar_tiendanube_productos(productos: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Identificador de URL": productos["nombre"].str.lower().str.replace(" ", "-"),
        "Nombre": productos["nombre"],
        "Categorías": productos["categoria"],
        "Precio": productos["precio"],
        "Stock": productos["stock_actual"],
        "SKU": productos["sku"],
        "Mostrar en tienda": "SI",
    })


def guardar(datos: dict, carpeta: Path) -> None:
    (carpeta / "verdad").mkdir(parents=True, exist_ok=True)
    ventas, items, productos = datos["ventas"], datos["venta_items"], datos["productos"]

    exportar_mercadopago(ventas).to_csv(carpeta / "mercadopago_liquidaciones.csv", index=False)
    # Tiendanube exporta con punto y coma: el ETL tiene que contemplarlo.
    exportar_tiendanube_ventas(ventas, items, productos).to_csv(
        carpeta / "tiendanube_ventas.csv", index=False, sep=";")
    exportar_tiendanube_productos(productos).to_csv(carpeta / "tiendanube_productos.csv", index=False, sep=";")
    productos[["sku", "nombre", "categoria", "costo", "stock_actual", "stock_minimo", "dias_reposicion"]].to_csv(
        carpeta / "planilla_productos.csv", index=False)
    datos["compras"].to_csv(carpeta / "compras.csv", index=False)
    datos["ipc"].to_csv(carpeta / "ipc.csv", index=False)

    columnas_verdad = ["id", "fecha_hora", "canal", "medio_pago", "monto_bruto",
                       "comision", "impuestos", "monto_neto", "fuente"]
    ventas[columnas_verdad].to_csv(carpeta / "verdad" / "ventas.csv", index=False)
    items.to_csv(carpeta / "verdad" / "venta_items.csv", index=False)


def main() -> None:
    datos = generar()
    carpeta = Path(__file__).resolve().parents[1] / "data"
    guardar(datos, carpeta)
    v = datos["ventas"]
    print(f"Ventas: {len(v):,} | Ítems: {len(datos['venta_items']):,} | Productos: {len(datos['productos'])}")
    print(f"Facturación total: ${v['monto_bruto'].sum():,.0f} | Desde {v['fecha_hora'].min()} hasta {v['fecha_hora'].max()}")
    print(f"Archivos en {carpeta}")


if __name__ == "__main__":
    main()
