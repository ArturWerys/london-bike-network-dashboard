import numpy as np
from scipy.stats import gmean


def log_binning_aw(mean, variance, a):
    # Funkcję opracował Artur Werys

    # Konwersja danych na tablice NumPy
    mean = np.asarray(mean)
    variance = np.asarray(variance)

    mean_min = min(mean)
    mean_max = max(mean)

    # Generowanie granic przedziałów logarytmicznych
    granice = [mean_min]

    while granice[-1] <= mean_max:
        granice.append(granice[-1] * a)

    # Listy punktów reprezentacyjnych
    srednie_reprezentacyjne = []
    wariancje_reprezentacyjne = []

    # Obliczanie średnich geometrycznych dla kolejnych przedziałów
    for i in range(len(granice) - 1):

        lewy_kraniec = granice[i]
        prawy_kraniec = granice[i + 1]

        # Wybór stacji należących do danego przedziału
        maska = (mean >= lewy_kraniec) & (mean < prawy_kraniec)

        srednie_w_przedziale = mean[maska]
        wariancje_w_przedziale = variance[maska]

        # Pominięcie pustych przedziałów
        if len(srednie_w_przedziale) == 0:
            continue

        # Średnie geometryczne dla stacji w danym przedziale
        srednia_geo = gmean(srednie_w_przedziale)
        wariancja_geo = gmean(wariancje_w_przedziale)

        srednie_reprezentacyjne.append(srednia_geo)
        wariancje_reprezentacyjne.append(wariancja_geo)

    return srednie_reprezentacyjne, wariancje_reprezentacyjne
