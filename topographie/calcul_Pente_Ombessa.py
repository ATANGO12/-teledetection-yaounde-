#MODULE SUR LE CALCUL DE LA PENTE
import rasterio
import numpy as np
import matplotlib.pyplot as plt

with rasterio.open("raster_ombessa.tif") as src:
    data = src.read(1)
    res_x, res_y = src.res
    meta = src.meta.copy
#Récuperer la résolution du raster 
print("Résolution x:", res_x)
print("Résolution y:", res_y)

#calculs des gradients
dz_dy, dz_dx = np.gradient(data, res_y, res_x) # Ceci calcul les variations d'altitudes Nord_sud et est_ouest

#Calcul de pente
slope_rad = np.arctan(np.sqrt(dz_dy**2 + dz_dx**2))
slope_deg = np.degrees(slope_rad)

#Afficher la pente
plt.imshow(slope_deg, cmap="terrain")
plt.colorbar(label = "pentes(degrés)")
plt.title("Carte de pente_ombessa")
plt.show()

#vérification
print("pente min:", np.nanmin(slope_deg))
print("pente max:", np.nanmax(slope_deg))

# reclassifier la pente
zones = np.zeros_like(slope_deg, dtype=np.uint8)

zones[(slope_deg >= 0) & (slope_deg < 5)] = 1 #très favorable por la construction
zones[(slope_deg >= 5 ) & (slope_deg < 15)] = 2 # Zone favorable avec terrassement
zones[(slope_deg >= 15)] = 3 # Non constructible 

#vérifier le résultat visuel 
plt.imshow(zones, cmap='gist_earth')
plt.colorbar(label = "classes de pente")
plt.title("zones constructibles selon la pente(pentes allant de de la classe 1 à 2)")
plt.show()

# Ouvrir le MNT original pour copier les métadonnées géographiques 
with rasterio.open("raster_Ombessa.tif") as src:
    meta = src.meta.copy()

# Mise à jour pour le nouveau raster (1 bande, float32)
meta.update({
    "count": 1,
    "dtype": "float32"
})
# Export du raster pente
with rasterio.open("pentes_ombessa", "w", **meta) as dst:
    dst.write(slope_deg, 1)