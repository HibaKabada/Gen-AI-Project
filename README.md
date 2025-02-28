# Projet : Gestion de Documents et Chatbot avec Streamlit

Ce projet est une application web qui permet de gérer des documents, de les stocker dans une base de données Cloud SQL, et de mettre en œuvre une logique de recherche (retrieval) pour interagir avec ces documents via un chatbot. L'interface utilisateur est développée avec Streamlit, et l'application expose des endpoints API pour gérer les interactions avec la base de données et le chatbot. Enfin, une partie de l'interface permet de gérer la description d'images et de collecter des feedbacks utilisateurs, qui sont stockés dans un fichier Excel pour générer un tableau de bord interactif.

## Table des matières

1. [Prérequis](#prérequis)
2. [Installation](#installation)
3. [Structure du Projet](#structure-du-projet)
4. [Utilisation](#utilisation)
5. [Métriques et Évaluation](#métriques-et-évaluation)
6. [Bonnes Pratiques](#bonnes-pratiques)
7. [Contribuer](#contribuer)
8. [Licence](#licence)

## Prérequis

- requirements.txt
- virtualenv 
 
## Installation

1. **Création d'un environnement virtuel**

   Pour isoler les dépendances du projet, commencez par créer un environnement virtuel :

   ```bash
   python -m venv venv

2. **Installation des dépendance**
   ```bash
   pip install -r requirements.txt

3. **Configuration de la base de données Cloud SQL**

   Créez une instance Cloud SQL et configurez-la pour accéder à votre base de données.
   Utilisez le notebook notebook_cloud_sql.ipynb pour créer la table et alimenter la base de données avec les documents nécessaires.

5. **Exécution du projet** 

   Exécutez le fichier vector.py pour gérer la connexion à la base de données et mettre en œuvre la logique de recherche.

    Exécutez api.py pour démarrer les endpoints API:
   
    ```bash
    uvicorn api:app --host 0.0.0.0 --port 8181
    ```
    Enfin, exécutez home.py pour lancer l'interface Streamlit.

## Structure du Projet
   - __notebook_cloud_sql.ipynb__ : Notebook utilisé pour créer la table dans Cloud SQL et alimenter la base de données avec des documents.
   - __vector.py__ : Gère la connexion à la base de données et implémente la logique de recherche (retrieval).
   - __api.py__ : Expose les endpoints API pour interagir avec la base de données et le chatbot.
   - __home.py__ : Interface Streamlit pour interagir avec l'application.
   - __eval.py__ : Script pour tester le chatbot et générer des métriques d'évaluation (BLEU score, ROUGE score, etc.).
   - feedback.xlsx : Fichier Excel pour stocker les feedbacks utilisateurs.
   - dashboard/ : Dossier contenant les éléments du tableau de bord.
