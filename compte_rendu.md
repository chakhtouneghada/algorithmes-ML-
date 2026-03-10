<img src="https://github.com/user-attachments/assets/aab39e9b-9f9d-45b2-a521-df8d07a4a965" width="200" height="auto" /> DGHOUGHI CHOROUK 
<!-- version réduite via HTML dans Markdown -->
<img src="https://github.com/user-attachments/assets/9606d462-735a-401a-8d8f-a15fd7cab70f" width="200" /> CHAKHTOUNE GHADA 
![unnamed](https://github.com/user-attachments/assets/ca583b38-7484-4a0f-b5c7-4d3a4f0f7edd) EL ALAOUI EL AMRANI ZAHIA 



# Développement de Modèles Prédictifs
Ce document décrit les procédures de préparation des données, l'implémentation de modèles de classification et les méthodes d'optimisation utilisées dans le projet.

1. Importation et Préparation des Données
Le flux de travail commence par le chargement de jeux de données standards de Scikit-Learn pour la validation des algorithmes.

Python

from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris() # 150 x 4, 3 classes
wine = datasets.load_wine() # 178 x 13, 3 classes
X, y = iris.data, iris.target

# Division stratifiée 80/20
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
Utilité technique :

datasets : Fournit des données prêtes à l'emploi pour tester la classification (Iris, Wine) ou la régression (California Housing) [cite : 1].

train_test_split : Sépare les données pour évaluer le modèle sur des données inconnues. L'option stratify=y est cruciale ici : elle garantit que les proportions des classes (par exemple, 33% pour chaque type d'Iris) sont maintenues dans les deux sous-ensembles [cite : 1].

2. Modèle de Régression Logistique avec Pipeline
Le notebook utilise un Pipeline pour automatiser le prétraitement et éviter le "data leakage" (fuite de données).

Python

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

lr = Pipeline([
    ("sc", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=42))
])
lr.fit(X_tr, y_tr)
proba = lr.predict_proba(X_te)
Analyse du code :

StandardScaler : Normalise les caractéristiques (moyenne 0, variance 1). L'utilisation au sein d'un Pipeline assure que le changement d'échelle appliqué au test se base uniquement sur les paramètres du groupe d'entraînement [cite : 1].

max_iter=1000 : Augmenté pour éviter les avertissements de non-convergence de l'algorithme [cite : 1].

predict_proba : Calcule les probabilités par classe, permettant une analyse plus fine que la simple prédiction binaire [cite : 1].

3. Arbre de Décision et Visualisation
Un Arbre de Décision est appliqué au dataset wine. Ce modèle est apprécié pour sa transparence et sa capacité d'interprétation.

Python

from sklearn.tree import DecisionTreeClassifier, plot_tree

dt = DecisionTreeClassifier(max_depth=4, min_samples_split=5, random_state=42)
dt.fit(X_wine_tr, y_wine_tr)

# Visualisation
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(dt, feature_names=wine.feature_names, class_names=wine.target_names, filled=True, ax=ax)
Points clés :

max_depth=4 : Contraint la croissance de l'arbre pour éviter le surapprentissage (overfitting) [cite : 1].

feature_importances_ : Le code permet d'extraire l'importance de chaque variable basée sur l'indice de Gini [cite : 1].

Résultats obtenus :
L'arbre de décision généré [cite : 1] montre que :

Le premier critère de séparation est l'intensité de la couleur (color_intensity <= 3.82).

Si l'intensité est faible, le modèle regarde la teneur en cendres (ash) pour identifier la classe 1.

Si l'intensité est élevée, il utilise les flavonoïdes (flavanoids) pour distinguer les classes 0 et 2.

[Image de l'arbre de décision généré pour le dataset Wine montrant les critères de Gini et les échantillons par nœud]

4. Optimisation et Validation Croisée
Le notebook introduit des méthodes avancées pour trouver les meilleurs réglages du modèle.

Recherche Aléatoire (RandomizedSearchCV)
Python

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    "n_estimators"  : randint(50, 500),
    "learning_rate" : uniform(0.01, 0.3),
    "max_depth"     : randint(2, 8),
}
rs = RandomizedSearchCV(gbr, param_distributions=param_dist, n_iter=50, cv=5, scoring="r2")
Cette méthode explore aléatoirement l'espace des paramètres, ce qui est souvent plus efficace qu'une recherche exhaustive (GridSearch) pour les modèles complexes [cite : 1].

Validation Croisée Répétée
L'utilisation de RepeatedStratifiedKFold et cross_validate permet de s'assurer que les performances obtenues ne sont pas dues au hasard du découpage des données [cite : 1].

Conclusion
L'approche démontrée dans ce notebook est rigoureuse. Elle combine la normalisation, l'automatisation via pipelines, et l'optimisation d'hyperparamètres. Ces étapes garantissent un modèle robuste et prêt pour un déploiement en production.
