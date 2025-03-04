# Projet : ESG RAG

Ce projet est une application web qui met en œuvre un système RAG (Retrieval-Augmented Generation) en utilisant l'API de Gemini pour générer des réponses contextuelles à partir de documents stockés dans une table Cloud SQL. L'interface utilisateur est développée avec Streamlit, et l'application expose des endpoints API pour gérer les interactions avec la base de données, le modèle Gemini, et le chatbot. 
Une partie de l'interface permet également de gérer la description d'images et de collecter des feedbacks utilisateurs, qui sont stockés dans un fichier Excel pour générer un tableau de bord interactif.

## Lien vers l'application déployée: 
https://ac-hk-projet-streamlit-1021317796643.europe-west1.run.app
## Table des matières

1. [Prérequis](#prérequis)
2. [Installation](#installation)
3. [Structure du Projet](#structure-du-projet)
4. [Utilisation](#utilisation)
5. [Métriques et Évaluation](#métriques-et-évaluation)
6. [Bonnes Pratiques](#bonnes-pratiques)
7. [Captures de notre application](#captures-de-notre-application)
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
   - __feedback.xlsx__ : Fichier Excel pour stocker les feedbacks utilisateurs.
   - __dashboard/__ : Dossier contenant les éléments du tableau de bord.

## Utilisation

1. **Création d'un environnement virtuel**

   Pour lancer l'interface utilisateur, exécutez :
   
   ```bash
   streamlit run home.py
   ```
   L'interface permet de :
   - __Rechercher des documents.__ 
   - __Interagir avec le chatbot.__ 
   - __Décrire des images.__ 
   - __Soumettre des feedbacks.__


2. **API Endpoints**

   Les endpoints API exposés dans api.py permettent de :
   - __Rechercher des documents.__ 
   - __Interagir avec le chatbot.__ 
   - __Stocker des feedbacks.__ 
4. **Feedback et Tableau de Bord**

   Les feedbacks utilisateurs sont stockés dans feedback.xlsx.
   Un tableau de bord est généré pour visualiser les feedbacks et interagir avec le chatbot.

## Métriques et Évaluation

  Le script eval.py permet de tester le chatbot en utilisant des métriques comme le BLEU score et le ROUGE score. 
  Les résultats sont sauvegardés dans un tableau de bord, et les questions/réponses utilisées pour les tests sont enregistrées pour une analyse ultérieure.
  Pour exécuter l'évaluation :
  ```bash
   python eval.py
   ```
## Bonnes Pratiques

Nous avons utilisé Flake8 pour nous assurer que le code respecte les bonnes pratiques de développement. 
Pour lancer Flake8 :
```bash
flake8 .
```
## Captures de notre application
![ESG1](https://github.com/user-attachments/assets/5244df60-cf9c-43da-8f8e-3f12ec726ea6)

![ESG2](https://github.com/user-attachments/assets/db6d217e-1e0a-4e95-8dc4-b7d636a0e962)

![ESG3](https://github.com/user-attachments/assets/f5ca17f8-d92e-46f4-b676-2375ec3c03ae)

![feedback_cap](https://github.com/user-attachments/assets/f501311f-087e-4344-8425-185695e9ded3)

![feedback3](https://github.com/user-attachments/assets/eca345ed-977a-45b6-bf97-f8a5142a39d3)

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

N'hésitez pas à explorer le code et à nous faire part de vos retours !
