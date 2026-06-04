import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import erf
from scipy.optimize import brentq

st.set_page_config(page_title="Гидравлические эпюры", layout="wide")
st.title("Расчёт полей скоростей при расширении потока с отрывом")
st.markdown("""
Моделирование по методу турбулентных струй.  
Реализован итерационный подбор члена давления `ξ` для сохранения расхода и метод наложения для ограниченного русла.
""")

# ВВОД ИСХОДНЫХ ДАННЫХ
st.sidebar.header("Исходные параметры")
b = st.sidebar.number_input("Полуширина начальной струи b, м", value=2.0, min_value=0.1)
B = st.sidebar.number_input("Полная ширина русла после расширения B, м", value= b * 5.0, min_value=b * 2.0)
V0 = st.sidebar.number_input("Начальная скорость V₀, м/с", value=1.5, min_value=0.1)
C = st.sidebar.number_input("Опытный коэффициент C", value=0.082, min_value=0.01, max_value=0.2, step=0.001)
n_streams = st.sidebar.number_input("Число расчётных струй", value=10, min_value=5, max_value=50)
tol = st.sidebar.number_input("Точность подбора расхода |ΣQ - Q₀|", value=0.001, min_value=0.0001, max_value=0.01, format="%.4f")
x_input = st.sidebar.text_input("Координаты x створов (через запятую), м", value="2.0, 5.0, 10.0")
x_sections = []
for x_str in x_input.split(","):
    try:
        x_val = float(x_str.strip())
        if x_val > 0:
            x_sections.append(x_val)
    except ValueError:
        pass

Q0 = 2 * b * V0  # Начальный расход на единицу глубины

# РАСЧЁТНОЕ ЯДРО
def kramp(z):
    """Функция Крампа: Φ(z) = 0.5 * (1 + erf(z/√2))"""
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

def V2_unbounded(y, x, b, C, V0):
    """Базовое уравнение для безграничного расширения"""
    Z1 = (y - b) / (2 * C * x)
    Z2 = (y + b) / (2 * C * x)
    return V0**2 * (kramp(Z2) - kramp(Z1))

def solve_xi_for_x(x, y_mid, b, B, C, V0, Q0, tol):
    """Итерационный подбор ξ до совпадения расхода"""
    def residual(xi):
        V2 = np.zeros_like(y_mid)
        for i, y in enumerate(y_mid):
            # Метод наложения для ограниченного пространства
            V2[i] = V2_unbounded(y, x, b, C, V0) + V2_unbounded(B - y, x, b, C, V0) - xi
        # Расчёт с учётом знака (обратное течение даёт отрицательный расход)
        V = np.sign(V2) * np.sqrt(np.abs(V2))
        dy = B / len(y_mid)
        return np.sum(V) * dy - Q0

    # Поиск корня методом Брента
    try:
        xi_opt = brentq(residual, 0.0, 2.0, xtol=tol, maxiter=100)
        return xi_opt
    except ValueError:
        return 0.0

def compute_full_profile(x, y_grid, b, B, C, V0, xi):
    V2 = np.zeros_like(y_grid)
    for i, y in enumerate(y_grid):
        V2[i] = V2_unbounded(y, x, b, C, V0) + V2_unbounded(B - y, x, b, C, V0) - xi
    V = np.sign(V2) * np.sqrt(np.abs(V2))
    return V

# ЗАПУСК РАСЧЁТА
if st.button("Рассчитать эпюры", type="primary"):
    if not x_sections:
        st.error("Введите хотя бы одну координату x.")
    else:
        y_mid = np.linspace(B/(2*n_streams), B - B/(2*n_streams), n_streams)  # середины струй
        y_plot = np.linspace(0, B, 300)
        
        tab1, tab2 = st.tabs(["Эпюры скоростей", "Таблица расчёта (стр. 8-11)"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(9, 5))
            for x in x_sections:
                xi = solve_xi_for_x(x, y_mid, b, B, C, V0, Q0, tol)
                V = compute_full_profile(x, y_plot, b, B, C, V0, xi)
                ax.plot(y_plot, V, linewidth=2.5, label=f"x = {x} м  (ξ = {xi:.4f})")
            
            ax.axhline(0, color="k", linewidth=0.5)
            ax.set_xlim(0, B)
            ax.set_ylim(bottom=-V0*0.2)  # небольшой запас для визуализации обратных токов
            ax.set_xlabel("Поперечная координата y, м")
            ax.set_ylabel("Осреднённая скорость V, м/с")
            ax.set_title("Эпюры распределения скоростей в заданных створах")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.success(f"Расчёт завершён. Q₀ = {Q0:.3f} м²/с. Точность подбора: ±{tol}")

        with tab2:
            st.subheader("Промежуточные данные (пример для первого створа)")
            x_ref = x_sections[0]
            xi_ref = solve_xi_for_x(x_ref, y_mid, b, B, C, V0, Q0, tol)
            
            Z1 = (y_mid - b) / (2 * C * x_ref)
            Z2 = (y_mid + b) / (2 * C * x_ref)
            Phi1 = kramp(Z1)
            Phi2 = kramp(Z2)
            V2_base = V0**2 * (Phi2 - Phi1)
            V2_sym = V2_base + np.array([V2_unbounded(B-y, x_ref, b, C, V0) for y in y_mid])
            V2_final = V2_sym - xi_ref
            V_final = np.sign(V2_final) * np.sqrt(np.abs(V2_final))
            Q_str = V_final * (B / n_streams)
            
            df = pd.DataFrame({
                "№ струи": range(1, n_streams+1),
                "y (середина), м": np.round(y_mid, 3),
                "Z₁": np.round(Z1, 3),
                "Z₂": np.round(Z2, 3),
                "Φ(Z₁)": np.round(Phi1, 4),
                "Φ(Z₂)": np.round(Phi2, 4),
                "V² баз., м²/с²": np.round(V2_base, 4),
                "V² налож., м²/с²": np.round(V2_sym, 4),
                "V² - ξ, м²/с²": np.round(V2_final, 4),
                "V, м/с": np.round(V_final, 3),
                "Q струи, м²/с": np.round(Q_str, 4)
            })
            st.dataframe(df, use_container_width=True)
            st.caption(f"Суммарный расход: ΣQ = {np.sum(Q_str):.4f} м²/с | ξ = {xi_ref:.4f} | Δ = {abs(np.sum(Q_str)-Q0):.5f}")