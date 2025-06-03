import gurobipy as gp
from gurobipy import GRB
import json
import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import optuna

# Importaciones adicionales para visualizar y guardar gráficos de Optuna
try:
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_contour
    )
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly o kaleido no instalado. Las gráficas de Optuna no se generarán.")


# ────────────────────────────────────────────────────────────────────────────────
# 1. LECTURA DE INSTANCIA
# ────────────────────────────────────────────────────────────────────────────────
def read_instance(filename):
    with open(filename, 'r') as f:
        return json.load(f)


# ────────────────────────────────────────────────────────────────────────────────
# 2. MODELO EXACTO (MILP)
# ────────────────────────────────────────────────────────────────────────────────
def build_model(instance, delta=10.0):
    eps = instance['energy_prices']
    T = [e['time'] for e in eps]
    num_periods = len(T)
    price = {t: eps[t]['price'] for t in range(num_periods)}
    Δt = instance['parking_config']['time_resolution']

    arrs = instance['arrivals']
    I = [v['id'] for v in arrs]
    α = {v['id']: v['arrival_time'] for v in arrs}
    β = {v['id']: v['departure_time'] for v in arrs}
    r = {v['id']: v['required_energy'] for v in arrs}
    w = {v['id']: v['willingness_to_pay'] for v in arrs}

    # Ventanas temporales por vehículo
    T_i = {i: [t for t in range(num_periods) if α[i] <= T[t] <= β[i]] for i in I}

    chs = instance['parking_config']['chargers']
    C = [c['charger_id'] for c in chs]
    P = {c['charger_id']: c['power'] for c in chs}
    η_c = {c['charger_id']: c['efficiency'] for c in chs}
    o = {c['charger_id']: c['operation_cost_per_hour'] for c in chs}
    N_s = instance['parking_config']['n_spots']
    L = instance['parking_config']['transformer_limit']

    r_min = {v['id']: v['min_charge_rate'] for v in arrs}
    r_max = {v['id']: v['max_charge_rate'] for v in arrs}
    C_i = {i: [c for c in C if r_min[i] <= P[c] <= r_max[i]] for i in I}

    M = max(T) + Δt

    m = gp.Model('EV_Scheduling')
    # Variables
    x = m.addVars(
        [(i, t, c) for i in I for t in T_i[i] for c in C_i[i]],
        vtype=GRB.BINARY, name='x'
    )
    z = m.addVars(
        [(i, t) for i in I for t in T_i[i]],
        vtype=GRB.BINARY, name='z'
    )
    E = m.addVars(I, lb=0.0, name='E')
    F = m.addVars(I, lb=0.0, name='F')
    f_var = m.addVar(lb=0.0, name='f')

    # Restricciones
    # R1: Un solo vehículo por cargador y período
    for t in range(num_periods):
        for c in C:
            m.addConstr(
                gp.quicksum(x[i, t, c] for i in I if (i, t, c) in x) <= 1,
                name=f'R1_c{c}_t{t}'
            )
    # R2: Un solo cargador por vehículo y período
    for i in I:
        for t in T_i[i]:
            m.addConstr(
                gp.quicksum(x[i, t, c] for c in C_i[i]) <= 1,
                name=f'R2_i{i}_t{t}'
            )
    # R3: Límite transformador en kW (sin Δt)
    for t in range(num_periods):
        m.addConstr(
            gp.quicksum(P[c] * x[i, t, c]
                        for i in I for c in C_i[i] if (i, t, c) in x)
            <= L,
            name=f'R3_t{t}'
        )
    # R4: Plazas de estacionamiento
    for t in range(num_periods):
        m.addConstr(
            gp.quicksum(z[i, t] for i in I if (i, t) in z) <= N_s,
            name=f'R4_t{t}'
        )
    # R5: Enlace x–z
    for i in I:
        for t in T_i[i]:
            m.addConstr(
                z[i, t] == gp.quicksum(x[i, t, c] for c in C_i[i]),
                name=f'R5_i{i}_t{t}'
            )
    # R6: Energía entregada (kWh), solo eficiencia del cargador
    for i in I:
        m.addConstr(
            E[i] == gp.quicksum(P[c] * Δt * η_c[c] * x[i, t, c]
                                for t in T_i[i] for c in C_i[i] if (i, t, c) in x),
            name=f'R6_i{i}'
        )
    # R7: No exceder demanda
    for i in I:
        m.addConstr(E[i] <= r[i], name=f'R7_req_i{i}')
    # R8: Tiempo de finalización con Big-M
    for i in I:
        for t in T_i[i]:
            m.addConstr(
                F[i] >= T[t] + Δt - M * (1 - z[i, t]),
                name=f'R8_i{i}_t{t}'
            )
    # R9: Tiempo de finalización ≤ departure
    for i in I:
        m.addConstr(F[i] <= β[i], name=f'R9_i{i}')
    # R11: Equidad
    m.addConstr(
        f_var == (1.0 / len(I)) * gp.quicksum(E[i] / r[i] for i in I),
        name='R11_fair'
    )

    # Función Objetivo
    revenue = gp.quicksum(
        w[i] * price[t] * P[c] * Δt * η_c[c] * x[i, t, c]
        for (i, t, c) in x
    )
    op_cost = gp.quicksum(o[c] * Δt * x[i, t, c] for (i, t, c) in x)
    fairness_pen = delta * gp.quicksum((E[i] / r[i] - f_var) ** 2 for i in I)

    m.setObjective(revenue - op_cost - fairness_pen, GRB.MAXIMIZE)
    m.update()

    data = (T, price, Δt, I, T_i, α, β, r, C, C_i, P, η_c, o, N_s, L, w)
    return m, data


def optimize_model(m):
    m.Params.TimeLimit = 7200
    m.Params.MIPGap = 0.02
    m.optimize()


def extract_metrics(model, data):
    T, price, Δt, I, T_i, α, β, r, C, C_i, P, η_c, o, N_s, L, w = data

    obj = model.ObjVal
    runtime = model.Runtime
    gap = 0.0
    if model.ObjBound > -GRB.INFINITY and abs(obj) > 1e-6:
        gap = abs(obj - model.ObjBound) / abs(obj) * 100

    # Energía entregada y nivel de servicio
    E_vals = np.array([model.getVarByName(f"E[{i}]").X for i in I])
    svc = E_vals / np.array([r[i] for i in I])
    avg_energy = float(E_vals.mean())
    avg_service = float(svc.mean())

    # Utilización de cargadores
    slots = len(T)
    charger_utils = []
    for c in C:
        active = 0
        for t in range(slots):
            for i in I:
                if t in T_i[i]:
                    v = model.getVarByName(f"x[{i},{t},{c}]")
                    if v and v.X > 0.5:
                        active += 1
                        break
        charger_utils.append(active / slots * 100)
    avg_charger_util = float(np.mean(charger_utils))

    # Utilización del transformador
    tx_usage = []
    for t in range(slots):
        u = 0
        for i in I:
            if t in T_i[i]:
                for c in C_i[i]:
                    v = model.getVarByName(f"x[{i},{t},{c}]")
                    if v and v.X > 0.5:
                        u += P[c]
        tx_usage.append(u)
    avg_transformer = float(np.mean(tx_usage))

    return {
        "objective_value": float(obj),
        "gurobi_runtime_sec": float(runtime),
        "mip_gap_percent": float(gap),
        "avg_energy_delivered_kWh": avg_energy,
        "avg_service_level": avg_service,
        "avg_charger_utilization_percent": avg_charger_util,
        "transformer_utilization_kW": avg_transformer
    }


# ────────────────────────────────────────────────────────────────────────────────
# 3. HEURÍSTICA CONSTRUCTIVA MEJORADA
# ────────────────────────────────────────────────────────────────────────────────
def heuristic_schedule(
    instance,
    delta=10.0,
    tournament_size=3,
    swap_threshold=0.10
):
    """
    Heurística constructiva mejorada:
     - Prioridad jerárquica: VAL[i] = w[i]; desempate con urgencia U[i] = rem_energy / remaining_time.
     - Cola de espera por periodo.
     - Torneos aleatorios de tamaño `tournament_size`.
     - Interrupciones parciales (swaps) con umbral `swap_threshold`.
     - Se asegura R2 y R4: un solo cargador/vehículo en t.
    """
    eps = instance['energy_prices']
    T = [e['time'] for e in eps]
    num_periods = len(T)
    price = {t: eps[t]['price'] for t in range(num_periods)}
    Δt = instance['parking_config']['time_resolution']

    arrs = instance['arrivals']
    I = [v['id'] for v in arrs]
    α = {v['id']: v['arrival_time'] for v in arrs}
    β = {v['id']: v['departure_time'] for v in arrs}
    r = {v['id']: v['required_energy'] for v in arrs}
    w = {v['id']: v['willingness_to_pay'] for v in arrs}

    # Ventanas temporales
    T_i = {i: [t for t in range(num_periods) if α[i] <= T[t] <= β[i]] for i in I}

    chs = instance['parking_config']['chargers']
    C = [c['charger_id'] for c in chs]
    P = {c['charger_id']: c['power'] for c in chs}
    η_c = {c['charger_id']: c['efficiency'] for c in chs}
    o = {c['charger_id']: c['operation_cost_per_hour'] for c in chs}
    N_s = instance['parking_config']['n_spots']
    L = instance['parking_config']['transformer_limit']

    r_min = {v['id']: v['min_charge_rate'] for v in arrs}
    r_max = {v['id']: v['max_charge_rate'] for v in arrs}
    C_i = {i: [c for c in C if r_min[i] <= P[c] <= r_max[i]] for i in I}

    # Estado inicial de plazas y transformador
    park = np.zeros(num_periods, dtype=int)
    tx = np.zeros(num_periods)
    # Disponibilidad de cada cargador en cada periodo
    avail = {c: np.ones(num_periods, dtype=bool) for c in C}

    assign = {}             # {(i,t,c): True}
    E_del = {i: 0.0 for i in I}
    landed = set()          # vehículos que completaron demanda

    # Cálculo inicial de urgencia y valor monetario
    U = {i: r[i] / max((β[i] - α[i]) * Δt, 1e-6) for i in I}
    VAL = {i: w[i] for i in I}

    for t in range(num_periods):
        # **Actualizar dinámicamente U[i] según energía restante y tiempo físico restante**
        for i in I:
            if i not in landed and t in T_i[i]:
                rem_energy = r[i] - E_del[i]
                remaining_slots = [tt for tt in T_i[i] if tt >= t]
                remaining_time = len(remaining_slots) * Δt
                U[i] = rem_energy / max(remaining_time, 1e-6)
            else:
                U[i] = 0.0

        # 1) Construir cola Q[t]: vehículos no satisfechos y activos en t
        Q = [i for i in I if (i not in landed and t in T_i[i])]

        # 2) Inicializar “ya asignado en este t” para garantizar R2 y R4
        assigned_this_period = {i: False for i in I}

        # 3) Mientras haya plazas libres, transformador disponible y Q no vacía:
        while park[t] < N_s and tx[t] < L and Q:
            # 3a) Selección por torneo aleatorio de tamaño `tournament_size`
            sample_size = min(tournament_size, len(Q))
            torneo = random.sample(Q, sample_size)
            # Ordenar torneo por (VAL[i], U[i])
            torneo.sort(key=lambda i: (VAL[i], U[i]), reverse=True)
            i_star = torneo[0]

            # 3b) Si i_star ya asignado en t, sacarlo de Q[t]
            if assigned_this_period[i_star]:
                Q.remove(i_star)
                continue

            # 3c) Buscar el primer cargador factible c_star
            c_star = None
            for c in C_i[i_star]:
                if avail[c][t] and (tx[t] + P[c] <= L):
                    c_star = c
                    break

            if c_star is None:
                # Si no hay cargador factible, sacarlo de Q[t]
                Q.remove(i_star)
                continue

            # 3d) Asignar (i_star, t, c_star)
            assign[(i_star, t, c_star)] = True
            park[t] += 1
            tx[t] += P[c_star]
            avail[c_star][t] = False
            assigned_this_period[i_star] = True

            # 3e) Actualizar energía entregada a i_star
            delivered = P[c_star] * Δt * η_c[c_star]
            E_del[i_star] += delivered
            if E_del[i_star] >= r[i_star] - 1e-6:
                landed.add(i_star)
                if i_star in Q:
                    Q.remove(i_star)

        # 4) Realizar a lo sumo un SWAP controlado en este periodo
        swapped = False
        for i_alt in Q:
            if swapped:
                break
            # Buscar un c_assigned donde hay (i_x, t, c_assigned) en assign
            for c_assigned in C:
                if swapped:
                    break
                i_x = None
                for (ii, tt, cc) in list(assign):
                    if tt == t and cc == c_assigned:
                        i_x = ii
                        break
                if i_x is None:
                    continue

                # Solo si VAL[i_alt] > VAL[i_x]
                if VAL[i_alt] <= VAL[i_x]:
                    continue

                # 4a) Calcular margen neto de i_x en (t, c_assigned)
                ingreso_x_t = w[i_x] * price[t] * P[c_assigned] * Δt * η_c[c_assigned]
                costo_x_t = o[c_assigned] * Δt
                margen_x_t = ingreso_x_t - costo_x_t

                # 4b) Encontrar mejor t2 ≥ t para i_x (Min pérdida)
                mejor_loss = float('inf')
                mejor_t2, mejor_c2 = None, None
                for t2 in T_i[i_x]:
                    if t2 <= t or park[t2] >= N_s or tx[t2] >= L:
                        continue
                    for c2 in C_i[i_x]:
                        if not avail[c2][t2] or tx[t2] + P[c2] > L:
                            continue
                        ingreso_x_t2 = w[i_x] * price[t2] * P[c2] * Δt * η_c[c2]
                        costo_x_t2 = o[c2] * Δt
                        margen_x_t2 = ingreso_x_t2 - costo_x_t2
                        loss_x = margen_x_t - margen_x_t2
                        if loss_x < mejor_loss:
                            mejor_loss = loss_x
                            mejor_t2 = t2
                            mejor_c2 = c2

                if mejor_t2 is None:
                    continue

                # 4c) Margen neto de i_alt en (t, c_assigned)
                ingreso_alt_t = w[i_alt] * price[t] * P[c_assigned] * Δt * η_c[c_assigned]
                costo_alt_t = o[c_assigned] * Δt
                margen_alt_t = ingreso_alt_t - costo_alt_t

                delta_margin = margen_alt_t - mejor_loss
                if delta_margin / max(margen_x_t, 1e-6) >= swap_threshold:
                    # 4d) Ejecutar swap
                    # 1) Remover a i_x
                    del assign[(i_x, t, c_assigned)]
                    park[t] -= 1
                    tx[t] -= P[c_assigned]
                    avail[c_assigned][t] = True
                    E_del[i_x] -= P[c_assigned] * Δt * η_c[c_assigned]
                    if E_del[i_x] < 0:
                        E_del[i_x] = 0.0
                    if i_x in landed and E_del[i_x] < r[i_x] - 1e-6:
                        landed.remove(i_x)
                    assigned_this_period[i_x] = False

                    # 2) Asignar i_x en (mejor_t2, mejor_c2)
                    assign[(i_x, mejor_t2, mejor_c2)] = True
                    park[mejor_t2] += 1
                    tx[mejor_t2] += P[mejor_c2]
                    avail[mejor_c2][mejor_t2] = False
                    E_del[i_x] += P[mejor_c2] * Δt * η_c[mejor_c2]
                    if E_del[i_x] >= r[i_x] - 1e-6:
                        landed.add(i_x)

                    # 3) Asignar i_alt en (t, c_assigned)
                    assign[(i_alt, t, c_assigned)] = True
                    park[t] += 1
                    tx[t] += P[c_assigned]
                    avail[c_assigned][t] = False
                    assigned_this_period[i_alt] = True
                    E_del[i_alt] += P[c_assigned] * Δt * η_c[c_assigned]
                    if E_del[i_alt] >= r[i_alt] - 1e-6:
                        landed.add(i_alt)

                    swapped = True
                    break

        # Fin de iteración periodo t

    # Calcular métricas finales de la heurística
    svc = {i: min(E_del[i] / r[i], 1.0) if r[i] > 0 else 0.0 for i in I}

    revenue = sum(w[i] * price[t] * P[c] * Δt * η_c[c]
                  for (i, t, c) in assign)
    op_cost = sum(o[c] * Δt for (i, t, c) in assign)
    f_bar = float(np.mean([svc[i] for i in I]))
    fairness_pen = delta * sum((svc[i] - f_bar) ** 2 for i in I)
    obj_h = revenue - op_cost - fairness_pen

    avg_energy = float(np.mean([E_del[i] for i in I]))
    avg_service = float(np.mean([svc[i] for i in I]))
    avg_charger_util = float(np.mean([
        sum(1 for (ii, tt, cc) in assign if cc == c) / num_periods * 100
        for c in C
    ]))
    avg_transformer = float(np.mean(tx))

    return {
        "assignment": assign,
        "E": E_del,
        "service_levels": svc,
        "heuristic_metrics": {
            "objective_value": obj_h,
            "avg_energy_delivered_kWh": avg_energy,
            "avg_service_level": avg_service,
            "avg_charger_utilization_percent": avg_charger_util,
            "transformer_utilization_kW": avg_transformer
        }
    }


# ────────────────────────────────────────────────────────────────────────────────
# 4. EXTRACCIÓN DE MÉTRICAS DESDE ASIGNACIÓN (PARA ALNS)
# ────────────────────────────────────────────────────────────────────────────────
def extract_metrics_from_assignment(assign, data, delta=10.0):
    """
    Dado un diccionario assign {(i,t,c): True}, calcula métricas idénticas a extract_metrics.
    """
    T, price, Δt, I, T_i, α, β, r, C, C_i, P, η_c, o, N_s, L, w = data

    # Energía entregada por vehículo y nivel de servicio
    E_vals = {i: 0.0 for i in I}
    for (i, t, c) in assign:
        E_vals[i] += P[c] * Δt * η_c[c]
    svc = {i: min(E_vals[i] / r[i], 1.0) if r[i] > 0 else 0.0 for i in I}
    avg_energy = float(np.mean([E_vals[i] for i in I]))
    avg_service = float(np.mean([svc[i] for i in I]))

    # Utilización de cargadores
    slots = len(T)
    charger_utils = []
    for c in C:
        active = 0
        for t in range(slots):
            for i in I:
                if t in T_i[i] and (i, t, c) in assign:
                    active += 1
                    break
        charger_utils.append(active / slots * 100)
    avg_charger_util = float(np.mean(charger_utils))

    # Utilización del transformador
    tx_usage = []
    for t in range(slots):
        u = 0
        for i in I:
            if t in T_i[i]:
                for c in C_i[i]:
                    if (i, t, c) in assign:
                        u += P[c]
        tx_usage.append(u)
    avg_transformer = float(np.mean(tx_usage))

    # Ingresos y costos operativos
    revenue = sum(w[i] * price[t] * P[c] * Δt * η_c[c]
                  for (i, t, c) in assign)
    op_cost = sum(o[c] * Δt for (i, t, c) in assign)

    f_bar = float(np.mean([svc[i] for i in I]))
    fairness_pen = delta * sum((svc[i] - f_bar) ** 2 for i in I)

    obj = float(revenue - op_cost - fairness_pen)

    return {
        "objective_value": obj,
        "avg_energy_delivered_kWh": avg_energy,
        "avg_service_level": avg_service,
        "avg_charger_utilization_percent": avg_charger_util,
        "transformer_utilization_kW": avg_transformer
    }


# ────────────────────────────────────────────────────────────────────────────────
# 5. OPERADORES DESTRUCCIÓN (Ω₋)
# ────────────────────────────────────────────────────────────────────────────────
def DRandom(x_current, data, k_min_frac=0.1, k_max_frac=0.3):
    """
    Destrucción aleatoria: elimina entre k_min_frac·|vehículos| y k_max_frac·|vehículos| vehículos completos.
    Retorna (x_part, veh_elim).
    """
    T, price, Δt, I, T_i, α, β, r, C, C_i, P, η_c, o, N_s, L, w = data
    veh_asg = sorted({i for (i, t, c) in x_current})
    if not veh_asg:
        return x_current.copy(), []

    k_min = max(1, int(math.floor(k_min_frac * len(veh_asg))))
    k_max = max(k_min, int(math.ceil(k_max_frac * len(veh_asg))))
    k = random.randint(k_min, k_max)
    k = min(k, len(veh_asg))
    elim = random.sample(veh_asg, k)

    x_part = x_current.copy()
    for (i, t, c) in list(x_part):
        if i in elim:
            del x_part[(i, t, c)]
    return x_part, elim


def DWorst(x_current, data, p=0.2):
    """
    Destrucción por peor contribución: elimina p·|veh_asg| vehículos de menor margen neto.
    Retorna (x_part, veh_elim).
    """
    T, price, Δt, I, T_i, α, β, r, C, C_i, P, η_c, o, N_s, L, w = data
    veh_asg = sorted({i for (i, t, c) in x_current})
    if not veh_asg:
        return x_current.copy(), []

    aportes = {}
    for i in veh_asg:
        ingreso_i = 0.0
        costo_i = 0.0
        for (ii, t, c) in x_current:
            if ii == i:
                ingreso_i += w[i] * price[t] * P[c] * Δt * η_c[c]
                costo_i += o[c] * Δt
        aportes[i] = ingreso_i - costo_i

    sorted_peores = sorted(aportes.items(), key=lambda item: item[1])
    k = int(math.ceil(p * len(veh_asg)))
    peores_k = [veh for (veh, _) in sorted_peores[:k]]

    x_part = x_current.copy()
    for (i, t, c) in list(x_part):
        if i in peores_k:
            del x_part[(i, t, c)]
    return x_part, peores_k


def DBand(x_current, data, b_frac=(0.05, 0.15)):
    """
    Destrucción por franja horaria: elimina todas las asignaciones en un bloque de b periodos consecutivos.
    Retorna (x_part, veh_elim).
    """
    T, price, Δt, I, T_i, α, β, r, C, C_i, P, η_c, o, N_s, L, w = data
    slots = len(T)
    if slots == 0:
        return x_current.copy(), []

    b_min = max(1, int(math.floor(b_frac[0] * slots)))
    b_max = max(b_min, int(math.ceil(b_frac[1] * slots)))
    b = random.randint(b_min, b_max)
    t0 = random.randint(0, slots - 1)
    franja = set(range(t0, min(slots, t0 + b)))

    x_part = x_current.copy()
    veh_elim = set()
    for (i, t, c) in list(x_part):
        if t in franja:
            veh_elim.add(i)
            del x_part[(i, t, c)]
    return x_part, list(veh_elim)


def DRelated(x_current, data, q_choices=(3, 5, 8)):
    """
    Destrucción de vehículos relacionados: elige un vehículo al azar y quita a sí y a sus q-1 vecinos más 'relacionados'.
    Retorna (x_part, veh_elim).
    """
    T, price, Δt, I, T_i, α, β, r, C, C_i, P, η_c, o, N_s, L, w = data
    veh_asg = sorted({i for (i, t, c) in x_current})
    if not veh_asg:
        return x_current.copy(), []

    i0 = random.choice(veh_asg)
    relaciones = []
    for j in veh_asg:
        if j == i0:
            continue
        val = abs(α[i0] - α[j]) + abs(β[i0] - β[j]) + abs(r[i0] - r[j])
        relaciones.append((j, val))
    relaciones.sort(key=lambda x: x[1])

    q = random.choice(q_choices)
    grupo = [i0] + [j for (j, _) in relaciones[: q - 1]]
    x_part = x_current.copy()
    for (i, t, c) in list(x_part):
        if i in grupo:
            del x_part[(i, t, c)]
    return x_part, grupo


# ────────────────────────────────────────────────────────────────────────────────
# 6. OPERADORES DE REPARACIÓN (Ω₊)
# ────────────────────────────────────────────────────────────────────────────────
def RGreedyRev(x_part, data):
    """
    Reparación golosa por ingreso máximo: reinsertar vehículos destruidos maximizando ganancia incremental.
    Retorna x_new (diccionario completo de asignaciones).
    """
    T, price, Δt, I, T_i, α, β, r, C, C_i, P, η_c, o, N_s, L, w = data

    assign = x_part.copy()
    # Reconstruir uso de plazas y transformador y disponibilidad de cargadores
    park = np.zeros(len(T), dtype=int)
    trans = np.zeros(len(T))
    avail = {c: np.ones(len(T), dtype=bool) for c in C}
    for (i, t, c) in assign:
        park[t] += 1
        trans[t] += P[c]
        avail[c][t] = False

    # Vehículos a reinsertar
    veh_asg = set(i for (i, t, c) in assign)
    all_veh = set(I)
    U = list(all_veh - veh_asg)

    while U:
        mejor_i, mejor_t, mejor_c = None, None, None
        mejor_incremento = -1e9
        for i in U:
            for t in T_i[i]:
                if park[t] >= N_s or trans[t] >= L:
                    continue
                for c in C_i[i]:
                    if (not avail[c][t]) or trans[t] + P[c] > L:
                        continue
                    ingreso_i = w[i] * price[t] * P[c] * Δt * η_c[c]
                    costo_i = o[c] * Δt
                    incremento = ingreso_i - costo_i
                    if incremento > mejor_incremento:
                        mejor_incremento = incremento
                        mejor_i, mejor_t, mejor_c = i, t, c
        if mejor_i is None:
            break
        assign[(mejor_i, mejor_t, mejor_c)] = True
        park[mejor_t] += 1
        trans[mejor_t] += P[mejor_c]
        avail[mejor_c][mejor_t] = False
        U.remove(mejor_i)

    return assign


def RGreedyServ(x_part, data):
    """
    Reparación golosa por nivel de servicio: prioriza ventanas ajustadas.
    Retorna x_new (diccionario completo de asignaciones).
    """
    T, price, Δt, I, T_i, α, β, r, C, C_i, P, η_c, o, N_s, L, w = data

    assign = x_part.copy()
    park = np.zeros(len(T), dtype=int)
    trans = np.zeros(len(T))
    avail = {c: np.ones(len(T), dtype=bool) for c in C}
    for (i, t, c) in assign:
        park[t] += 1
        trans[t] += P[c]
        avail[c][t] = False

    veh_asg = set(i for (i, t, c) in assign)
    all_veh = set(I)
    U = list(all_veh - veh_asg)
    # Ordenar U por urgencia inicial (se recalculará en ALNS si se requiere)
    urgencias = {i: r[i] / max((β[i] - α[i]) * Δt, 1e-6) for i in I}
    U.sort(key=lambda i: urgencias[i], reverse=True)

    while U:
        i = U.pop(0)
        mejor_t, mejor_c, max_nivel_serv = None, None, -1.0
        for t in T_i[i]:
            if park[t] >= N_s or trans[t] >= L:
                continue
            for c in C_i[i]:
                if (not avail[c][t]) or trans[t] + P[c] > L:
                    continue
                energia = P[c] * Δt * η_c[c]
                nivel_serv = min(energia / r[i], 1.0)
                if nivel_serv > max_nivel_serv:
                    max_nivel_serv = nivel_serv
                    mejor_t, mejor_c = t, c
                elif np.isclose(nivel_serv, max_nivel_serv):
                    ingreso = w[i] * price[t] * P[c] * Δt * η_c[c]
                    costo = o[c] * Δt
                    inc = ingreso - costo
                    ingreso_best = 0.0
                    if mejor_t is not None:
                        ingreso_best = (
                            w[i] * price[mejor_t] * P[mejor_c] * Δt * η_c[mejor_c]
                            - o[mejor_c] * Δt
                        )
                    if inc > ingreso_best:
                        mejor_t, mejor_c = t, c
        if mejor_t is not None:
            assign[(i, mejor_t, mejor_c)] = True
            park[mejor_t] += 1
            trans[mejor_t] += P[mejor_c]
            avail[mejor_c][mejor_t] = False

    return assign


def RMinCost(x_part, data):
    """
    Reparación por costo mínimo neto: inserta minimizando costo_ins - ingreso.
    Retorna x_new (diccionario completo de asignaciones).
    """
    T, price, Δt, I, T_i, α, β, r, C, C_i, P, η_c, o, N_s, L, w = data

    assign = x_part.copy()
    park = np.zeros(len(T), dtype=int)
    trans = np.zeros(len(T))
    avail = {c: np.ones(len(T), dtype=bool) for c in C}
    for (i, t, c) in assign:
        park[t] += 1
        trans[t] += P[c]
        avail[c][t] = False

    veh_asg = set(i for (i, t, c) in assign)
    all_veh = set(I)
    U = list(all_veh - veh_asg)

    while U:
        mejor_i, mejor_t, mejor_c = None, None, None
        min_costo_neto = float('inf')
        for i in U:
            for t in T_i[i]:
                if park[t] >= N_s or trans[t] >= L:
                    continue
                for c in C_i[i]:
                    if (not avail[c][t]) or trans[t] + P[c] > L:
                        continue
                    costo_ins = price[t] * P[c] * Δt + o[c] * Δt
                    ingreso = w[i] * price[t] * P[c] * Δt * η_c[c]
                    costo_neto = costo_ins - ingreso
                    if costo_neto < min_costo_neto:
                        min_costo_neto = costo_neto
                        mejor_i, mejor_t, mejor_c = i, t, c
        if mejor_i is None:
            break
        assign[(mejor_i, mejor_t, mejor_c)] = True
        park[mejor_t] += 1
        trans[mejor_t] += P[mejor_c]
        avail[mejor_c][mejor_t] = False
        U.remove(mejor_i)

    return assign


# ────────────────────────────────────────────────────────────────────────────────
# 7. ALNS (ADAPTIVE LARGE NEIGHBORHOOD SEARCH)
# ────────────────────────────────────────────────────────────────────────────────
def ALNS(
    instance,
    data,
    time_limit=60,
    delta=10.0,
    T0_factor=0.1,
    alpha_temp=0.995,
    lam=0.8,
    heur_tournament_size=None,
    heur_swap_threshold=None
):
    """
    Implementación de ALNS con tres hiperparámetros ajustables:
      - T0_factor: factor para inicializar la temperatura como T0 = max(1, T0_factor·Obj_init)
      - alpha_temp: factor de decaimiento de temperatura (entre 0 y 1)
      - lam: factor de decaimiento para actualizar pesos de operadores (entre 0 y 1)
    Además, recibe opcionalmente (heur_tournament_size, heur_swap_threshold) 
    para inicializar la heurística de forma óptima si se dispone de esos valores.
    Retorna métricas de la mejor solución.
    """
    # 1) Solución inicial = heurística constructiva mejorada 
    #    (usa parámetros optimizados si se le pasan, o los defaults sino)
    if heur_tournament_size is not None and heur_swap_threshold is not None:
        init_sol = heuristic_schedule(
            instance,
            delta,
            tournament_size=heur_tournament_size,
            swap_threshold=heur_swap_threshold
        )
    else:
        init_sol = heuristic_schedule(instance, delta)

    x_current = init_sol["assignment"].copy()
    metrics_init = init_sol["heuristic_metrics"].copy()
    Obj_current = metrics_init["objective_value"]

    # 2) Mejor global
    x_best = x_current.copy()
    Obj_best = Obj_current

    # 3) Operadores
    Ω_minus = [DRandom, DWorst, DBand, DRelated]
    Ω_plus = [RGreedyRev, RGreedyServ, RMinCost]

    ρ_minus = np.ones(len(Ω_minus))
    ρ_plus = np.ones(len(Ω_plus))

    # Temperatura inicial
    T0 = max(1.0, T0_factor * Obj_current)
    T = T0
    ω1, ω2, ω3, ω4 = 10, 5, 1, 0

    iter_total = 0
    iter_sin_mejora = 0
    t_start = time.time()

    # Bucle principal ALNS
    while True:
        iter_total += 1
        # 3a) Enfriar
        T *= alpha_temp

        # 3b) Selección adaptativa de operadores
        probs_minus = ρ_minus / ρ_minus.sum()
        probs_plus = ρ_plus / ρ_plus.sum()
        j = np.random.choice(len(Ω_minus), p=probs_minus)
        k = np.random.choice(len(Ω_plus), p=probs_plus)
        d = Ω_minus[j]
        r_op = Ω_plus[k]

        # 3c) Destrucción y reparación
        x_part, veh_elim = d(x_current, data)
        x_temp = r_op(x_part, data)

        # 3d) Evaluar x_temp
        met_temp = extract_metrics_from_assignment(x_temp, data, delta)
        Obj_temp = met_temp["objective_value"]

        # 3e) Criterio de aceptación (simulated annealing)
        Δ_obj = Obj_temp - Obj_current
        if Δ_obj >= 0:
            aceptado = True
        else:
            p_accept = math.exp(Δ_obj / T)
            aceptado = (random.random() < p_accept)

        Obj_current_old = Obj_current
        x_current_old = x_current.copy()

        if aceptado:
            x_current = x_temp.copy()
            Obj_current = Obj_temp

        # 3f) Actualizar mejor global
        if Obj_temp > Obj_best:
            Obj_best = Obj_temp
            x_best = x_temp.copy()
            iter_sin_mejora = 0
        else:
            iter_sin_mejora += 1

        # 3g) Calcular score ψ
        if Obj_temp > Obj_best:
            psi = ω1
        elif aceptado and (Obj_temp > Obj_current_old):
            psi = ω2
        elif aceptado:
            psi = ω3
        else:
            psi = ω4

        # 3h) Actualizar pesos
        ρ_minus[j] = lam * ρ_minus[j] + (1 - lam) * psi
        ρ_plus[k] = lam * ρ_plus[k] + (1 - lam) * psi

        # 3i) Criterios de parada
        if time.time() - t_start >= time_limit:
            break
        if iter_sin_mejora >= 5000:
            break
        if iter_total >= 50000:
            break

    # 4) Retornar métricas de la mejor global
    best_metrics = extract_metrics_from_assignment(x_best, data, delta)
    return best_metrics


# ────────────────────────────────────────────────────────────────────────────────
# 8. OPTIMIZACIÓN DE HIPERPARÁMETROS DIVIDIDA
# ────────────────────────────────────────────────────────────────────────────────
def objective_heuristic(trial, instance):
    """
    Función objetivo para tunear la heurística con Optuna.
    Permite buscar los valores de:
      - tournament_size: entero entre 2 y 10
      - swap_threshold: real entre 0.01 y 0.50
    Retorna el valor medio negativo de la función objetivo (para maximizar obj, Optuna minimiza).
    """
    tournament_size = trial.suggest_int("tournament_size", 2, 10)
    swap_threshold = trial.suggest_float("swap_threshold", 0.01, 0.50)

    results = []
    for _ in range(5):
        sol = heuristic_schedule(
            instance,
            tournament_size=tournament_size,
            swap_threshold=swap_threshold
        )
        results.append(sol["heuristic_metrics"]["objective_value"])

    return -np.mean(results)


def tune_heuristic(
    instances, heuristic_trials=50,
    out_heur="tuning_heuristic_results.xlsx"
):
    """
    Separa el tuning de la heurística de ALNS.
    Guarda los resultados en out_heur.
    Además, genera gráficas en directorios organizados:
      plots/heuristica/<nombre_instancia>/
    """
    # Carpeta raíz para gráficas de heurística
    base_dir = os.path.join("plots", "heuristica")
    os.makedirs(base_dir, exist_ok=True)

    heur_records = []
    for fname in instances:
        if not os.path.exists(fname):
            print(f"Alerta: {fname} no existe, saltando.")
            continue

        inst = read_instance(fname)
        study_h = optuna.create_study(direction="minimize")
        study_h.optimize(lambda trial: objective_heuristic(trial, inst),
                         n_trials=heuristic_trials)

        best_h = study_h.best_params
        heur_records.append({
            "instance": fname,
            "tournament_size": best_h["tournament_size"],
            "swap_threshold": best_h["swap_threshold"]
        })
        print(f"[Tuning Heurística] {fname}: {best_h}")

        if PLOTLY_AVAILABLE and study_h.trials:
            # Crear subcarpeta para esta instancia
            inst_name = os.path.splitext(os.path.basename(fname))[0]
            inst_dir = os.path.join(base_dir, inst_name)
            os.makedirs(inst_dir, exist_ok=True)
            try:
                fig_history = plot_optimization_history(study_h)
                fig_history.write_image(os.path.join(inst_dir, "history.png"))
                fig_importance = plot_param_importances(study_h)
                fig_importance.write_image(os.path.join(inst_dir, "importance.png"))
                fig_parallel = plot_parallel_coordinate(study_h)
                fig_parallel.write_image(os.path.join(inst_dir, "parallel.png"))
                fig_contour = plot_contour(
                    study_h,
                    params=["tournament_size", "swap_threshold"]
                )
                fig_contour.write_image(os.path.join(inst_dir, "contour.png"))
                print(f"Gráficas Optuna heurística para {fname} guardadas en '{inst_dir}'.")
            except Exception as e:
                print(f"Error generando gráficas de heurística en {fname}: {e}")

    df_h = pd.DataFrame(heur_records)
    df_h.to_excel(out_heur, index=False)
    print(f"Hiperparámetros Heurística guardados en '{out_heur}'.")


def objective_alns(trial, instance, data):
    """
    Función objetivo para tunear ALNS con Optuna.
    Hiperparámetros:
      - T0_factor: real entre 0.01 y 1.0
      - alpha_temp: real entre 0.90 y 0.999
      - lam: real entre 0.1 y 0.9
    Retorna el valor medio negativo de la función objetivo de ALNS.
    """
    T0_factor = trial.suggest_float("T0_factor", 0.01, 1.0)
    alpha_temp = trial.suggest_float("alpha_temp", 0.90, 0.999)
    lam = trial.suggest_float("lam", 0.1, 0.9)

    objs = []
    for _ in range(3):
        met = ALNS(
            instance, data,
            time_limit=30,       # 30s durante tuning
            T0_factor=T0_factor,
            alpha_temp=alpha_temp,
            lam=lam
        )
        objs.append(met["objective_value"])

    return -np.mean(objs)


def tune_alns(
    instances, alns_trials=50,
    out_alns="tuning_alns_results.xlsx"
):
    """
    Separa el tuning de ALNS de la heurística.
    Requiere que build_model esté disponible para cada instancia.
    Guarda los resultados en out_alns.
    Además, genera gráficas en directorios organizados:
      plots/metaheuristica/<nombre_instancia>/
    """
    # Carpeta raíz para gráficas de metaheurística
    base_dir = os.path.join("plots", "metaheuristica")
    os.makedirs(base_dir, exist_ok=True)

    alns_records = []
    for fname in instances:
        if not os.path.exists(fname):
            print(f"Alerta: {fname} no existe, saltando.")
            continue

        inst = read_instance(fname)
        _, data = build_model(inst)

        study_a = optuna.create_study(direction="minimize")
        study_a.optimize(lambda trial: objective_alns(trial, inst, data),
                         n_trials=alns_trials)

        best_a = study_a.best_params
        alns_records.append({
            "instance": fname,
            "T0_factor": best_a["T0_factor"],
            "alpha_temp": best_a["alpha_temp"],
            "lam": best_a["lam"]
        })
        print(f"[Tuning ALNS] {fname}: {best_a}")

        if PLOTLY_AVAILABLE and study_a.trials:
            # Crear subcarpeta para esta instancia
            inst_name = os.path.splitext(os.path.basename(fname))[0]
            inst_dir = os.path.join(base_dir, inst_name)
            os.makedirs(inst_dir, exist_ok=True)
            try:
                fig_history_a = plot_optimization_history(study_a)
                fig_history_a.write_image(os.path.join(inst_dir, "history.png"))
                fig_importance_a = plot_param_importances(study_a)
                fig_importance_a.write_image(os.path.join(inst_dir, "importance.png"))
                fig_parallel_a = plot_parallel_coordinate(study_a)
                fig_parallel_a.write_image(os.path.join(inst_dir, "parallel.png"))
                fig_contour_a = plot_contour(
                    study_a,
                    params=["T0_factor", "alpha_temp"]
                )
                fig_contour_a.write_image(os.path.join(inst_dir, "contour.png"))
                print(f"Gráficas Optuna ALNS para {fname} guardadas en '{inst_dir}'.")
            except Exception as e:
                print(f"Error generando gráficas de ALNS en {fname}: {e}")

    df_a = pd.DataFrame(alns_records)
    df_a.to_excel(out_alns, index=False)
    print(f"Hiperparámetros ALNS guardados en '{out_alns}'.")


# ────────────────────────────────────────────────────────────────────────────────
# 9. EJECUCIÓN CON O SIN HIPERPARÁMETROS (DRY & FLEXIBLE)
# ────────────────────────────────────────────────────────────────────────────────
def run_heuristic(
    instances,
    runs=30,
    output_path="heuristic_results.xlsx",
    hyperparams_file=None
):
    """
    Ejecuta la heurística (con o sin parámetros calibrados).
    Si hyperparams_file existe, lee (tournament_size, swap_threshold) de cada instancia;
    si no, usa valores por defecto (3, 0.10).
    Guarda todas las corridas en output_path.
    """
    # Leer hiperparámetros si se da el archivo
    heur_params = {}
    if hyperparams_file and os.path.exists(hyperparams_file):
        df_h = pd.read_excel(hyperparams_file)
        heur_params = {
            row["instance"]: (int(row["tournament_size"]), float(row["swap_threshold"]))
            for _, row in df_h.iterrows()
        }

    all_records = []
    for fname in instances:
        if not os.path.exists(fname):
            print(f"[Heurística] {fname} no existe, saltando.")
            continue

        # Determinar si hay parámetros óptimos para esta instancia
        if fname in heur_params:
            t_size, swap_thr = heur_params[fname]
        else:
            t_size, swap_thr = 3, 0.10  # valores por defecto

        inst = read_instance(fname)
        for run_id in range(1, runs + 1):
            t0 = time.time()
            sol = heuristic_schedule(
                inst,
                tournament_size=t_size,
                swap_threshold=swap_thr
            )
            m = sol["heuristic_metrics"].copy()
            m.update({
                "instance": fname,
                "run_id": run_id,
                "elapsed_time_sec": time.time() - t0
            })
            all_records.append(m)
            print(f"Heurística {fname} run {run_id}/{runs}: Obj={m['objective_value']:.2f}")

    df = pd.DataFrame(all_records)
    summary = df.groupby("instance").agg({
        "objective_value": ["mean", "std"],
        "avg_energy_delivered_kWh": ["mean", "std"],
        "avg_service_level": ["mean", "std"],
        "avg_charger_utilization_percent": ["mean", "std"],
        "transformer_utilization_kW": ["mean", "std"],
        "elapsed_time_sec": ["mean", "std"]
    })
    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, sheet_name="AllRuns", index=False)
        summary.to_excel(writer, sheet_name="SummaryStats")
    print(f"Resultados Heurística guardados en '{output_path}'.")


def run_alns(
    instances,
    runs=30,
    output_path="alns_results.xlsx",
    hyperparams_file=None,
    heur_hyperparams_file=None
):
    """
    Ejecuta ALNS (con o sin parámetros calibrados).
    Si hyperparams_file existe, lee (T0_factor, alpha_temp, lam) de cada instancia;
    si no, usa valores por defecto (0.1, 0.995, 0.8).
    Además, si heur_hyperparams_file existe, lee los (tournament_size, swap_threshold)
    óptimos de la heurística para pasarlos a la inicialización de ALNS.
    Guarda todas las corridas en output_path.
    """
    # Leer hiperparámetros de ALNS si se da el archivo
    alns_params = {}
    if hyperparams_file and os.path.exists(hyperparams_file):
        df_a = pd.read_excel(hyperparams_file)
        alns_params = {
            row["instance"]: (float(row["T0_factor"]),
                              float(row["alpha_temp"]),
                              float(row["lam"]))
            for _, row in df_a.iterrows()
        }

    # Leer hiperparámetros de la heurística si se da el archivo
    heur_params = {}
    if heur_hyperparams_file and os.path.exists(heur_hyperparams_file):
        df_h = pd.read_excel(heur_hyperparams_file)
        heur_params = {
            row["instance"]: (int(row["tournament_size"]),
                              float(row["swap_threshold"]))
            for _, row in df_h.iterrows()
        }

    all_records = []
    for fname in instances:
        if not os.path.exists(fname):
            print(f"[ALNS] {fname} no existe, saltando.")
            continue

        # Determinar si hay parámetros ALNS óptimos para esta instancia
        if fname in alns_params:
            T0f, a_temp, lam = alns_params[fname]
        else:
            T0f, a_temp, lam = 0.1, 0.995, 0.8  # valores por defecto

        # Determinar si hay parámetros heurísticos óptimos para esta instancia
        if fname in heur_params:
            ht_size, hs_thr = heur_params[fname]
        else:
            ht_size, hs_thr = None, None

        inst = read_instance(fname)
        _, data = build_model(inst)
        for run_id in range(1, runs + 1):
            t0 = time.time()
            met = ALNS(
                inst, data,
                time_limit=60,
                T0_factor=T0f,
                alpha_temp=a_temp,
                lam=lam,
                heur_tournament_size=ht_size,
                heur_swap_threshold=hs_thr
            )
            met.update({
                "instance": fname,
                "run_id": run_id,
                "elapsed_time_sec": time.time() - t0
            })
            all_records.append(met)
            print(f"ALNS {fname} run {run_id}/{runs}: Obj={met['objective_value']:.2f}")

    df = pd.DataFrame(all_records)
    summary = df.groupby("instance").agg({
        "objective_value": ["mean", "std"],
        "avg_energy_delivered_kWh": ["mean", "std"],
        "avg_service_level": ["mean", "std"],
        "avg_charger_utilization_percent": ["mean", "std"],
        "transformer_utilization_kW": ["mean", "std"],
        "elapsed_time_sec": ["mean", "std"]
    })
    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, sheet_name="AllRuns", index=False)
        summary.to_excel(writer, sheet_name="SummaryStats")
    print(f"Resultados ALNS guardados en '{output_path}'.")


# ────────────────────────────────────────────────────────────────────────────────
# 10. EJECUCIÓN INDEPENDIENTE DE CADA MÉTODO
# ────────────────────────────────────────────────────────────────────────────────
def run_milp(instances, output_path="milp_results.xlsx"):
    """
    Ejecuta el modelo MILP para cada instancia y almacena métricas en un archivo Excel.
    """
    all_results = []
    for fname in instances:
        if not os.path.exists(fname):
            print(f"Alerta: {fname} no existe, saltando.")
            continue
        inst = read_instance(fname)
        model, data = build_model(inst)
        t0 = time.time()
        optimize_model(model)
        m_met = extract_metrics(model, data)
        m_met["instance"] = fname
        m_met["elapsed_time_sec"] = time.time() - t0
        all_results.append(m_met)

    df = pd.DataFrame(all_results)
    df.to_excel(output_path, index=False)
    print(f"Resultados MILP guardados en '{output_path}'.")


# ────────────────────────────────────────────────────────────────────────────────
# 11. PUNTO DE ENTRADA PRINCIPAL
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Definir lista de archivos de instancia
    instance_files = [f"test_system_{i}.json" for i in range(1, 8)]

    # Ejemplos de ejecución:

    # 1) Ejecutar tuning de heurística por separado:
    tune_heuristic(instance_files, heuristic_trials=50, out_heur="tuning_heuristic_results.xlsx")

    # 2) Ejecutar tuning de ALNS por separado:
    tune_alns(instance_files, alns_trials=50, out_alns="tuning_alns_results.xlsx")

    # 3) Ejecutar 30 corridas de heurística (usar parámetros calibrados si existen):
    run_heuristic(instance_files, runs=30,
                output_path="heuristic_results.xlsx",
                hyperparams_file="tuning_heuristic_results.xlsx")

    # 4) Ejecutar 30 corridas de ALNS (usar parámetros calibrados si existen para ALNS,
    #    y además pasar parámetros óptimos de heurística si se dispone):
    run_alns(instance_files, runs=30,
             output_path="alns_results.xlsx",
             hyperparams_file="tuning_alns_results.xlsx",
             heur_hyperparams_file="tuning_heuristic_results.xlsx")

    # 5) Ejecutar solo MILP:
    run_milp(instance_files, output_path="milp_results.xlsx")

    pass
