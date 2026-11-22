from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

from config import MONTH_OPTIONS


def show_taylor_chart(
    mean_values,
    variance_values,
    selected_years,
    selected_month,
    selected_day_type,
    selected_hour_range,
    selected_rain_option,
    mean_log = None,
    variance_log = None,
    excluded_high_mean_points=1,
    excluded_low_mean_points=0,
    binned=True
):
    """Wyświetl wykres prawa Taylora z chmurą stacji i dopasowaniem."""

    years_label = ", ".join(str(year) for year in selected_years)
    period_label = f"{selected_month} {years_label}"

    if selected_day_type != "Wszystkie dni":
        period_label += f", {selected_day_type}"

    if selected_hour_range is not None:
        period_label += f", {selected_hour_range[0]} - {selected_hour_range[1]} h"
    
    if selected_rain_option != "Wszystkie dni":
        period_label += f", {selected_rain_option.lower()}"

    st.subheader(f"Prawo Taylora - {period_label}")

    if binned:
        mean_log = np.asarray(mean_log)
        variance_log = np.asarray(variance_log)
    else:
        mean_log = np.asarray(mean_values)
        variance_log = np.asarray(variance_values)

    order = np.argsort(mean_log)
    mean_log = mean_log[order]
    variance_log = variance_log[order]

    # Punkty uwzględnione w dopasowaniu
    n = len(mean_log)
    fit_mask = np.zeros(n, dtype=bool)
    fit_mask[excluded_low_mean_points:max(0, n - excluded_high_mean_points)] = True

    if fit_mask.sum() < 2:
        st.warning("Za mało punktów do dopasowania. Obniż progi lub zmniejsz liczbę pomijanych punktów.")
        return

    excluded_mask = ~fit_mask

    # Dopasowanie prawa Taylora
    x = np.log(mean_log[fit_mask])
    y = np.log(variance_log[fit_mask])

    alpha, log_A = np.polyfit(x, y, 1)
    A = np.exp(log_A)

    # Ocena jakości dopasowania
    y_pred = alpha * x + log_A
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    alpha_str = f"{alpha:.3f}".replace(".", ",")
    r_squared_str = f"{r_squared:.3f}".replace(".", ",")

    # Generowanie dopasowanej funkcji
    x_fit = np.geomspace(
        mean_log[fit_mask].min(),
        mean_log[fit_mask].max(),
        100
    )
    y_fit = A * x_fit**alpha

    # Wykres
    fig, ax = plt.subplots(figsize=(10, 7))

    # Chmura punktów – poszczególne stacje
    ax.scatter(
        mean_values,
        variance_values,
        color="#80C4E9",
        alpha=0.5,
        s=36,
        zorder=1,
        label="Poszczególne stacje"
    )

    if binned:

        # Punkty po binowaniu uwzględnione w dopasowaniu
        ax.scatter(
            mean_log[fit_mask],
            variance_log[fit_mask],
            color="#102A43",
            alpha=0.7,
            marker="D",
            s=70,
            zorder=3,
            label="Punkty po binowaniu logarytmicznym"
        )

        # Punkty pominięte w dopasowaniu
        if excluded_mask.any():
            ax.scatter(
                mean_log[excluded_mask],
                variance_log[excluded_mask],
                marker="D",
                facecolors="white",
                edgecolors="#102A43",
                linewidths=1.8,
                s=85,
                zorder=3,
                label="Punkty pominięte w dopasowaniu"
            )

        # Dopasowana funkcja potęgowa
        ax.plot(
            x_fit,
            y_fit,
            color="#102A43",
            linewidth=2.5,
            zorder=2,
            label=fr"Dopasowanie: $\alpha = {alpha_str}$; $R^2 = {r_squared_str}$"
        )

    else:
        # Dopasowana funkcja potęgowa
        ax.plot(
            x_fit,
            y_fit,
            color="#102A43",
            linewidth=2.5,
            zorder=2,
            label=fr"Dopasowanie: $\alpha = {alpha_str}$; $R^2 = {r_squared_str}$"
        )

    # Skale i opisy
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Średnia dzienna liczba wypożyczeń")
    ax.set_ylabel("Wariancja")

    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10))
        axis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:g}")
        )
        axis.set_minor_locator(NullLocator())

    ax.grid(True, which="major", axis="both", alpha=0.3)
    ax.legend(frameon=False)

    fig.suptitle(
        f"Prawo Taylora - {period_label}",
        fontsize=16
    )

    fig.tight_layout()

    try:
        st.pyplot(fig, width=1000)

        # Eksport PNG
        with BytesIO() as image_buffer:
            fig.savefig(
                image_buffer,
                format="png",
                dpi=300,
                bbox_inches="tight"
            )

            month_number = MONTH_OPTIONS.index(selected_month)
            file_years = "_".join(str(year) for year in selected_years)
            file_period = (
                f"{file_years}_{month_number:02d}"
                if month_number else file_years
            )
            weather_suffix = {
                "Wszystkie dni": "",
                "Dni deszczowe": "_deszczowe",
                "Dni bezdeszczowe": "_bezdeszczowe",
            }[selected_rain_option]
            st.download_button(
                "Zapisz wykres jako PNG",
                data=image_buffer.getvalue(),
                file_name=f"prawo_taylora_{file_period}{weather_suffix}.png",
                mime="image/png",
                on_click="ignore",
            )
    finally:
        plt.close(fig)
