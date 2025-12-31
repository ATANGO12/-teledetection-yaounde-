import rasterio
import numpy as np
import geopandas as gpd 
from sklearn.ensemble import RandomForestClassifier
from rasterio.mask import mask
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt 

# chemin de l'export 
output_path = r"C:\Users\VALENTIN\Desktop\sig\Resultat\classification_yaounde.tif"
raster_path = "image.tif"
# ouvrir l'image
with rasterio.open(raster_path) as src:
    image = src.read()
    profile = src.profile.copy()
    transform = src.transform
    crs = src.crs
    height, width = src.height, src.width

#Préparer l'image pour la prédiction
image_reshape = image.reshape(image.shape[0], -1).T

# Charger les echantillons
training = gpd.read_file('training.shp')
print("crs de l'image:", crs)
print("echantillons:", training.crs)

# Extraire les pixels
x = []
y = []
for _, row in training.iterrows():
    geom = row.geometry
    label = row['class'] # Récupère la class 1, 3, 4, 5

    if geom.is_empty:
        continue

# Crée un masque pour les pixels dans le polygone/point

    mask = rasterio.features.geometry_mask([geom.__geo_interface__],
                                           out_shape = (height, width),
                                           transform=transform,
                                           invert=True)
    values = image[:, mask] # valeurs  des bandes sous le masque
    if values.shape[1] > 0: # si au moins un pixel
        x.append(values.T) # Ajoute les pixels (transposé)
        y.extend([label] * values.shape[1]) # Ajoute la classe pour chaque pixel
if len(x) == 0:
    print("Erreur: Aucun pixel extrait. Vérifier")
    exit()

x = np.vstack(x) # assemble tout en un grand tableau 
y = np.array(y)
print(f"pixels extraits: len{(y)}")
print(f"classes: {np.unique(y, return_counts=True)}")

#Train/test préparer l'entrainement (70%) et test (30%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

#Entrainner le modèle Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(x_train, y_train)

# Tester le modèle sur les données de test 
y_pred = rf.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

#Classification sur toute l'image
classification = rf.predict(image_reshape)
classification_map = classification.reshape(height, width)

#indice de kappa 
kappa = cohen_kappa_score(y_test, y_pred)
print(f"cohen's kappa: {kappa:.3f}")

#Exporter la carte classifiée en GeoTIFF
profile.update(count=1, dtype='uint8', nodata=0)
with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(classification_map.astype('uint8'), 1)

#AFFICHAGE
plt.figure(figsize=(12, 12))
plt.imshow(classification_map, cmap='tab10')
plt.title("classification Yaounde")
plt.show()