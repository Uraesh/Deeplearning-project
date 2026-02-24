# Resume EDA - Wisconsin

Notebook source: `notebooks/01_eda_wisconsin.ipynb`

## Constats principaux

- Le dataset est globalement propre apres suppression de `Unnamed:_32`.
- La cible `diagnosis` est binaire (`M`/`B`) et directement exploitable.
- Les variables numeriques montrent des correlations fortes avec la cible.
- Le risque de fuite directe est limite en excluant `id` des features.

## Actions retenues pour le pipeline

- Encodage de la cible en `M -> 1` et `B -> 0`.
- Exclusion de `id` et des colonnes non informatives.
- Imputation mediane et standardisation.
- Validation croisee + selection de seuil par sensibilite cible.
