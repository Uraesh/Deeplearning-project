# Systeme IA de Detection du Cancer du Sein (V1)

Projet de fin de module apprentissage profond, construit comme un mini-projet hospitalier:
pipeline d'entrainement CPU-first en PyTorch, API FastAPI, interface web de demonstration,
gouvernance des donnees et logique de mise en production.

## 1. Objectif du projet

- Construire un classifieur `MALIGNANT` vs `BENIGN` fiable sur le jeu de donnees Wisconsin.
- Travailler selon une logique reelle: reproductibilite, auditabilite, metriques cliniques.
- Limiter la consommation CPU/memoire et les couts cloud avec une strategie de cache.
- Preparer une extension future vers MIAS (imagerie).

## 2. Ce qui a ete fait (historique conception -> implementation)

1. Cadrage produit et medical
- Cible clinique: outil de triage (pas un diagnostic autonome).
- Priorisation de la sensibilite (detection des cas malins).
- Definition d'une gouvernance de donnees (`docs/data_governance.md`).

2. Pipeline data tabulaire
- Chargement robuste CSV.
- Normalisation des noms de colonnes (`spaces -> underscores`).
- Exclusion des colonnes non exploitables (`id`, `Unnamed*`, colonnes 100% NaN).
- Encodage du target (`M -> 1`, `B -> 0`).
- Split stratifie train/test.

3. Modelisation PyTorch CPU-first
- Modele `TabularMLP` (`src/breast_cancer_ai/model.py`).
- Regularisation: dropout, batch norm, weight decay.
- Entrainement avec:
  - `BCEWithLogitsLoss` ponderee (desequilibre des classes),
  - gradient clipping,
  - scheduler `ReduceLROnPlateau`,
  - early stopping.

4. Controle de l'overfitting
- Holdout test externe (evaluation finale).
- Cross-validation stratifiee K-fold sur train+val.
- Probabilites OOF (out-of-fold) pour choisir le seuil.
- Seuil choisi pour atteindre d'abord une sensibilite cible.
- Rapport des ecarts trainval vs test (`overfit_gap`).

5. Artefacts, cache et tracabilite
- Versionning par run dans `models/runs/...`.
- Alias stable dans `models/latest/`.
- `training_signature` = hash (config + signature dataset).
- Si signature identique, reutilisation des artefacts sans reentrainement (`cache_hit`).
- Ecriture automatique de:
  - `model.pt`,
  - `metrics.json`,
  - `report.md`,
  - `cache_manifest.json`.

6. API de prod (FastAPI)
- Endpoints:
  - `GET /health`
  - `GET /model_info`
  - `POST /predict`
  - `POST /predict_batch`
- Validation stricte des payloads (Pydantic).
- Chargement du modele au demarrage (lifespan).
- `request_id` par appel pour audit.

7. Interface web de demo
- `GET /` sert une interface HTML/CSS/JS branchee a l'API.
- Design moderne, couleurs fortes, responsive mobile/desktop.
- Generation dynamique du formulaire depuis `feature_names`.
- Affichage resultat: probabilite, label, seuil, request id.
- Telechargement d'un compte-rendu medical patient (`.txt`) apres inference.
- Dashboard Plotly separe pour medecins (`/performance-dashboard`): AUC, sensibilite/specificite, matrice de confusion.

8. Scripts d'exploitation et maintenance
- Initialisation d'environnement virtuel avec cache de dependances.
- Lancement train/API.
- Monitoring memoire/stockage/API.
- Nettoyage des caches/fichiers intermediaires.

9. Containerisation Docker
- API en conteneur CPU.
- Volume des modeles monte en local.
- Auto-train au demarrage seulement si `model.pt` absent.

10. Qualite de code
- Corrections Pylance/Pylint.
- Typage renforce (TypedDict, cast controles, signatures explicites).
- Tests unitaires sur data, metriques et modele.

## 3. Architecture technique

- `src/breast_cancer_ai/data.py`: ingestion, nettoyage, splits.
- `src/breast_cancer_ai/model.py`: MLP tabulaire.
- `src/breast_cancer_ai/train.py`: CV OOF, entrainement final, artefacts, cache.
- `src/breast_cancer_ai/inference.py`: chargement artefact + prediction.
- `src/breast_cancer_ai/api.py`: service FastAPI + interface.
- `src/breast_cancer_ai/web/*`: frontend.
- `configs/train_config.yaml`: hyperparametres.
- `scripts/*.ps1`: operations.
- `docs/*`: gouvernance, fiche modele, checklist prod, plan MIAS.
- `notebooks/01_eda_wisconsin.ipynb`: exploration de donnees reproductible.
- `docs/eda_summary.md`: synthese EDA pour la soutenance.

## 4. Commandes utilisees

### 4.1 Prerequis PowerShell (si scripts bloques)

Option session courante uniquement (recommandee):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Ou execution one-shot:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

### 4.2 Installation / initialisation

```powershell
.\scripts\setup.ps1
```

Forcer une reinstallation des dependances:

```powershell
.\scripts\setup.ps1 -ForceInstall
```

### 4.3 Entrainement

```powershell
.\scripts\run_train.ps1
```

Forcer un nouvel entrainement (ignorer cache):

```powershell
.\scripts\run_train.ps1 -ForceRetrain
```

Avec config personnalisee:

```powershell
.\scripts\run_train.ps1 -ConfigPath configs/train_config.yaml
```

### 4.4 Lancer l'API + interface web

```powershell
.\scripts\run_api.ps1
Start-Process http://127.0.0.1:8000/
```

Routes utiles:
- Interface web: `http://127.0.0.1:8000/`
- Dashboard performance medecins: `http://127.0.0.1:8000/performance-dashboard`
- Documentation API (Swagger): `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`
- Endpoint metriques dashboard: `http://127.0.0.1:8000/performance`

Resolution automatique des metriques:
- Priorite 1: variable d'environnement `METRICS_PATH`.
- Priorite 2: fichier `metrics.json` voisin du `MODEL_PATH` charge.
- Priorite 3: `models/latest/metrics.json`.

### 4.5 Tester l'API rapidement

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json
```

Test prediction sur la premiere ligne de `data.csv`:

```powershell
$row = Import-Csv data.csv | Select-Object -First 1
$features = @{}
foreach ($p in $row.PSObject.Properties) {
  if ([string]::IsNullOrWhiteSpace($p.Name)) { continue }
  if ($p.Name -in @('id','diagnosis')) { continue }
  if ($p.Name -like 'Unnamed*') { continue }
  if ([string]::IsNullOrWhiteSpace([string]$p.Value)) { continue }
  $features[$p.Name.Trim().Replace(' ','_')] = [double]$p.Value
}
$body = @{features=$features} | ConvertTo-Json -Depth 6
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 6
```

### 4.6 Notebook EDA (presentation soutenance)

```powershell
$env:PYTHONPATH='src'
python -m notebook notebooks/01_eda_wisconsin.ipynb
```

Si `notebook` n'est pas installe:

```powershell
pip install notebook
python -m notebook notebooks/01_eda_wisconsin.ipynb
```

### 4.7 Monitoring memoire, stockage, sante API

```powershell
.\scripts\monitor.ps1
```

Ce script affiche:
- sante API,
- metriques dashboard via `/performance`,
- test rapide `/predict`,
- memoire process API (si local Python),
- espace disque (drives),
- taille totale projet,
- taille detaillee des artefacts `models/`.

### 4.8 Nettoyage

Nettoyage standard:

```powershell
.\scripts\clean.ps1
```

Nettoyage profond (inclut `models/latest`):

```powershell
.\scripts\clean.ps1 -Deep
```

Purge cache pip:

```powershell
.\scripts\clean.ps1 -PipCache
```

### 4.9 Tests unitaires

```powershell
$env:PYTHONPATH='src'
pytest -q tests
```

### 4.10 Docker

Demarrage standard:

```powershell
docker compose up
```

Rebuild image (si dependances/Dockerfile changent):

```powershell
docker compose up --build
```

Arret:

```powershell
docker compose down
```

### 4.11 Deploiement en ligne (Render)

1. Pousser le projet sur GitHub.
2. Sur Render: `New +` -> `Web Service` -> connecter le repo.
3. Parametres service:
- Runtime: `Docker`
- Branch: `main` (ou ta branche de release)
- Region: la plus proche de vos utilisateurs
- Health Check Path: `/health`
4. Variables d'environnement (optionnelles):
- `MODEL_PATH=/app/models/latest/model.pt`
- `METRICS_PATH=/app/models/latest/metrics.json`
5. Lancer le deploy.

Notes importantes:
- Le conteneur demarre l'API sur le port fourni par l'environnement cloud (`PORT`).
- Si `model.pt` est absent dans l'image, le conteneur entraine automatiquement un modele avant de lancer l'API.
- L'interface est servie sur la racine `/`, donc une seule URL suffit pour API + dashboard.

## 5. Strategies de reduction couts CPU/memoire

- Initialisation avec cache des dependances (`requirements` hash).
- Cache d'entrainement par signature (evite les re-train inutiles).
- `num_workers=0` et `pin_memory=False` en CPU.
- Nettoyage memoire periodique (`cleanup_memory` + `gc`).
- Nettoyage artefacts/caches via `scripts/clean.ps1`.
- Docker relance API sans rebuild systematique.

## 6. Gouvernance et conformite interne

Voir:
- `docs/data_governance.md`
- `docs/model_card.md`
- `docs/production_checklist.md`
- `docs/eda_summary.md`

Points cles:
- pas d'identifiants patients dans les entrees modele,
- tracabilite des versions et seuils,
- controle de la sensibilite cible,
- revue interne avant release.

## 7. Structure du depot

```text
configs/
docs/
models/
scripts/
src/breast_cancer_ai/
tests/
data.csv
Dockerfile
docker-compose.yml
requirements.txt
```

## 8. Feuille de route (prochaine etape)

- Ajouter pipeline imagerie MIAS separe.
- Ajouter la route API `/predict_image`.
- Ajouter calibration avancee (courbe de calibration + decision curve).
- Ajouter CI (tests + lint) pour collaboration a 5.
