import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# Chemin du dossier contenant les fichiers TXT
dossier = "C:\\projet 1 tpop\\Projet2"

def traiter_dossier(dossier):
    if not os.path.exists(dossier):
        print(f"Erreur : Le dossier '{dossier}' n'existe pas.")
        return None

    fichiers_txt = [f for f in os.listdir(dossier) if f.endswith(".txt")]
    if not fichiers_txt:
        print(f"Aucun fichier .txt trouvé dans '{dossier}'.")
        return None

    data_dict = {}
    for fichier in fichiers_txt:
        chemin_fichier = os.path.join(dossier, fichier)
        try:
            df = pd.read_csv(chemin_fichier, sep="\t", decimal=",", header=None, names=["LongueurOnde", "Transmittance"])
            df = df.dropna()
            df["LongueurOnde"] = pd.to_numeric(df["LongueurOnde"], errors='coerce')
            df["Transmittance"] = pd.to_numeric(df["Transmittance"], errors='coerce')
            df = df.dropna()
            
            for _, row in df.iterrows():
                longueur_onde = row["LongueurOnde"]
                transmittance = abs(row["Transmittance"])
                if 350 <= longueur_onde <= 1000:
                    if longueur_onde not in data_dict:
                        data_dict[longueur_onde] = []
                    data_dict[longueur_onde].append(transmittance)
        except Exception as e:
            print(f"Erreur lors de la lecture de {fichier}: {e}")

    return data_dict

# Chargement des données
data_dict = traiter_dossier(dossier)
if not data_dict:
    print("Erreur : Le dossier contient peu ou pas de données exploitables.")
    exit()

# Moyenne des valeurs
longueurs_ondes = np.array(sorted(data_dict.keys()))
transmittances_moyennes = np.array([np.mean(data_dict[lo]) for lo in longueurs_ondes])

# Normalisation
tmax = np.max(transmittances_moyennes)
transmittances_normalisees = transmittances_moyennes / tmax

# Distribution de la puissance totale
puissance_totale = 90.9  # mW
somme_transmittances = np.sum(transmittances_normalisees)
puissances_par_longueur_onde_W = (transmittances_normalisees / somme_transmittances) * (puissance_totale / 1000)

# Fonction gaussienne pour le fit
def gaussienne(x, A, x0, sigma):
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

# Estimation des paramètres initiaux
A_init = np.max(puissances_par_longueur_onde_W)
x0_init = longueurs_ondes[np.argmax(puissances_par_longueur_onde_W)]
sigma_init = 50  # Largeur approximative

# Ajustement gaussien avec bornes
params, covariance = curve_fit(gaussienne, longueurs_ondes, puissances_par_longueur_onde_W, 
                               p0=[A_init, x0_init, sigma_init],
                               bounds=([0, 500, 10], [np.inf, 650, 200]))
A_fit, x0_fit, sigma_fit = params

# Calcul de R^2
residus = puissances_par_longueur_onde_W - gaussienne(longueurs_ondes, *params)
ss_res = np.sum(residus**2)
ss_tot = np.sum((puissances_par_longueur_onde_W - np.mean(puissances_par_longueur_onde_W))**2)
r_squared = 1 - (ss_res / ss_tot)

# Tracé du fit
x_fit = np.linspace(min(longueurs_ondes), max(longueurs_ondes), 500)
y_fit = gaussienne(x_fit, *params)

# Courbe photopique V(λ)
V_lambda_data = {
    380: 0.0004, 400: 0.0100, 420: 0.0480, 440: 0.1400, 460: 0.3362, 480: 0.6150,
    500: 0.8540, 520: 0.9540, 540: 0.8700, 560: 0.7500, 580: 0.6100, 600: 0.4700,
    620: 0.3490, 640: 0.2220, 660: 0.1390, 680: 0.0750, 700: 0.0390, 720: 0.0190,
    740: 0.0093, 760: 0.0046, 780: 0.0022
}

lambda_values = np.array(list(V_lambda_data.keys()))
V_lambda_values = np.array(list(V_lambda_data.values()))
V_lambda_interp = interp1d(lambda_values, V_lambda_values, kind='linear', fill_value=0, bounds_error=False)

# Calcul des lumens
lumen = 683 * np.trapz(puissances_par_longueur_onde_W * V_lambda_interp(longueurs_ondes), longueurs_ondes)

# Tracé
fig, axs = plt.subplots(2, 1, figsize=(8, 10))
axs[0].plot(longueurs_ondes, transmittances_normalisees, label="Transmittance Normalisée", color='blue')
axs[0].set_xlabel("Longueur d'onde (nm)")
axs[0].set_ylabel("Transmittance Normalisée")
axs[0].legend()

axs[1].plot(longueurs_ondes, puissances_par_longueur_onde_W, label="Puissance (W)", color='red')
axs[1].plot(x_fit, y_fit, label="Fit Gaussien", linestyle='dashed', color='green')
axs[1].set_xlabel("Longueur d'onde (nm)")
axs[1].set_ylabel("Puissance (W)")
axs[1].legend()

plt.tight_layout()
plt.show()

# Résultats
print(f"Équation du fit gaussien : P(lambda) = {A_fit:.6e} * exp(-((lambda - {x0_fit:.2f})²) / (2 * {sigma_fit:.2f}²))")
print(f"Flux lumineux total : {lumen:.2f} lumens")
print(f"Coefficient de détermination (R²) : {r_squared:.4f}")