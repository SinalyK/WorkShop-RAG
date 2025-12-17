# WORSHOP RAG
Cet atelier construit un système de Retrieval-Augmented Generation (RAG), un agent intelligent capable de rechercher dans des documents externes, de récupérer des informations pertinentes et de générer des réponses fondées sur ces données du livre 
## "Building Agentic Al Systems by Matthew R. Scott and Dr. Alex Acero 288 pages"


# RAG Agent avec Bi-encoder, Cross-encoder, Self-Retrieval et Tools

## Description

Ce projet met en place un système de **Retrieval-Augmented Generation (RAG)** piloté par un **agent ReAct**.  
L’agent est capable de :
- interroger une base de connaissances locale indexée dans **ChromaDB** (vector store) ;
- utiliser un **bi-encoder (SentenceTransformer)** pour le retrieval ;
#### Bi-encoder
| Modèle                                 | max tokens par défaut | Remarques                       |
| -------------------------------------- | --------------------- | ------------------------------- |
| `all-MiniLM-L6-v2`                     | **256**               | le plus rapide et léger         |
| `all-MiniLM-L12-v2`                    | **256**               | version plus grande             |
| `paraphrase-MiniLM-L6-v2`              | **256**               | très populaire                  |
| `distiluse-base-multilingual-cased-v2` | **512**               | multilingue                     |
| `all-mpnet-base-v2`                    | **384**               | très performant mais plus lent  |
| `bert-base-nli-mean-tokens`            | **512**               | modèle BERT classique           |
| `multi-qa-MiniLM-L6-cos-v1`            | **512**               | pour la recherche sémantique    |
| `LaBSE`                                | **512**               | multilingue puissant mais lourd |
- utiliser un **cross-encoder** pour le *reranking* des documents ;
#### Cross Encoder
| Modèle                                   | Description                                                           | Max tokens |
| ---------------------------------------- | --------------------------------------------------------------------- | ---------- |
| `cross-encoder/ms-marco-MiniLM-L-6-v2`   | Très rapide, entraîné sur MS MARCO pour le re-ranking de recherche    | 512        |
| `cross-encoder/ms-marco-electra-base`    | Plus lourd, mais très performant pour recherche sémantique            | 512        |
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` | Ultra léger et rapide (bon pour API)                                  | 256        |
| `cross-encoder/nli-deberta-base`         | Entraîné pour *entailment/contradiction*                              | 512        |
| `cross-encoder/nli-roberta-base`         | Version RoBERTa pour la NLI                                           | 512        |
| `cross-encoder/stsb-roberta-base`        | Entraîné sur STS-B (similarité de phrases)                            | 512        |
| `cross-encoder/stsb-TinyBERT-L-4`        | Version TinyBERT pour STS-B                                           | 256        |
| `cross-encoder/quora-roberta-base`       | Entraîné pour détection de doublons Quora                             | 512        |
| `cross-encoder/qnli-electra-base`        | Modèle de QNLI (question entailment)                                  | 512        |
| `cross-encoder/ce-roberta-large-stsb`    | Grand modèle RoBERTa fine-tuné sur STS-B                              | 512        |
- déclencher du **self-retrieval** (le modèle génère lui-même les requêtes de recherche internes) ;
- appeler différents **tools** (retrieval RAG, recherche web, API métier comme la météo, etc.).

Ce README accompagne le notebook de cours pour expliquer les principaux concepts et la structure du code.

---


## Architecture générale
```
WORKSHOP-RAG/
├── .ipynb_checkpoints/
├── app/
│ ├── init.py
│ ├── agent.py # Agent ReAct, définition des tools, self-retrieval, etc.
│ ├── app.py # Interface Streamlit pour interagir avec le système
│ ├── prompts.py # Prompts (RAG, ReAct, CoT/ToT, etc.)
│ ├── trash.py # Code de tests / brouillon (non essentiel)
│ ├── utils.py # Fonctions utilitaires (chargement docs, split, config…)
│ └── workshop.py # Logique principale du workshop (pipelines, helpers)
├── chroma_store/ # Stockage local de la base vectorielle Chroma
├── docs/ # Documents sources pour le RAG
├── .env
├── README.md
├── requirements.txt # Dépendances Python
└── Workshop.ipynb # Notebook du cours (explications théoriques + démos)
```
Pipeline logique :


Vue d’ensemble de l’architecture logique :

1. **Workshop.ipynb**  
   - Introduit RAG, chunking, embeddings, bi-encoder / cross-encoder, Chroma, self-retrieval et agents ReAct.  
   - Contient des démonstrations interactives.

2. **app/utils.py**  
   - Chargement des documents depuis `docs/`.  
   - Split / chunking des textes.  
   - Création / connexion au `chroma_store/`.  

3. **app/agent.py**  
   - Définition de l’agent ReAct.  
   - Déclaration des tools (retriever RAG, éventuellement web, météo, etc.).  
   - Gestion du self-retrieval et du reranker.

4. **app/app.py**  
   - Application **Streamlit** pour interagir avec l’agent via une interface web simple.  

5. **chroma_store/**  
   - Stockage persistant des embeddings (VectorStore Chroma).

---

## Concepts principaux

### Retrieval-Augmented Generation (RAG)

- Combine **retrieval** (recherche de documents pertinents) et **generation** (réponse produite par un LLM).  
- Le LLM s’appuie sur les documents du corpus (dans `docs/`) plutôt que sur sa seule mémoire interne.

### Document Load and Split

- Chargement des documents (PDF, TXT, Markdown, etc.) → objets `Document` avec texte + métadonnées.  
- Splitting / chunking :
  - *chunk size* typique : 200–1000 tokens ;
  - *overlap* : 10–20 % ;
  - stratégies : fixe, structurelle, hybride.

### SentenceTransformer, Bi-encoder et Cross-encoder

- **SentenceTransformer** en bi-encoder :
  - encode les chunks et la requête de l’utilisateur ;
  - sert à la recherche sémantique rapide via Chroma.
- **Cross-encoder** :
  - lit `(requête, chunk)` ensemble ;
  - rerank le top‑k initial pour ne garder que les documents les plus pertinents.

### VectorStore : ChromaDB

- Stocke `embedding + texte + metadata` pour chaque chunk.  
- Fournit les opérations de similarity search et de filtrage par métadonnées.

### Self-Retrieval et Reranker Retrieving

- **Self-retrieval** : le modèle génère lui-même les requêtes de recherche internes et décide quand appeler le retriever.  
- **Reranker retrieving** :
  - bi-encoder + Chroma : top‑k candidats ;
  - cross-encoder : reranking de haute précision sur ce petit ensemble.

### Agents et ReAct

- **ReAct** (Reason + Act) :
  - Thought → Action (appel de tool) → Observation → Thought → … → Final Answer.  
- **Tools** :
  - retrieval RAG (Chroma) ;
  - outils web / API ;
  - outils utilitaires (calculs, parsing, etc.).

---

## Installation

### 1. Créer et activer un environnement virtuel

Depuis la racine du projet (`WORKSHOP-RAG/`) :

#### Sous Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```
#### Sous Windows (PowerShell)
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```
#### Variable d'enviroennement .env

```bash
GOOGLE_API_KEY=""
GROQ_API_KEY=
TAVILY_API_KEY=""
```

### 2. Installer les dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```


## Utilisation

### 1. Exécuter le notebook de workshop

Pour suivre la partie théorique et les démos pas à pas :

```bash
jupyter notebook
```
puis ouvrir Workshop.ipynb dans l’interface Jupyter

Le notebook explique :
- le pipeline Document Load & Split ;
- la création du vector store Chroma ;
- le rôle du bi-encoder, cross-encoder et du reranker retrieving ;
- le self-retrieval et l’agent ReAct.

---

### 2. Lancer l’interface Streamlit (`app/app.py`)

Pour interagir avec l’agent via une interface web :

```bash
streamlit run app/app.py
```

Puis ouvrir le lien fourni (généralement `http://localhost:8501`) dans votre navigateur.

Dans l’application Streamlit, vous pouvez :
- poser des questions basées sur les documents du dossier `docs/` ;
- observer les réponses générées par le LLM ;
- (éventuellement) afficher des informations de debug sur les documents récupérés et les tools appelés.

---

## Remarques

- Le dossier `chroma_store/` est créé et/ou mis à jour lors de la phase d’indexation (embeddings + stockage dans Chroma).  
- Si vous modifiez les documents dans `docs/`, pensez à relancer la phase d’indexation (via le notebook ou un script dans `app/`) pour mettre à jour le vector store.

---

## Licence

Ce projet est fourni à des fins pédagogiques dans le cadre d’un workshop sur le RAG et les agents.  
Vous êtes libre de le cloner, de l’adapter et de l’étendre pour vos propres expérimentations ou projets d’apprentissage.  

Si vous réutilisez une partie significative du code ou des notebooks dans un autre projet public, merci de :
- mentionner la source originale du workshop ;
- ajouter un lien vers ce dépôt dans votre propre README.

---

## Auteurs et développeurs

Ce workshop a été développé par :
- **LAMKADEM Ayoud** – Élève  ingénieur en IA à l'Ecole Nationale de l'Intelligence Artificielle et du Digital 
- **COULIBALY Adama** – Élève  ingénieur en IA à l'Ecole Nationale de l'Intelligence Artificielle et du Digital 
- **KANADJIGUI Sinaly** – Élève  ingénieur en IA à l'Ecole Nationale de l'Intelligence Artificielle et du Digital 