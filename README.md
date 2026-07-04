# Agent RAG d'Analyse de Données Financières

Interrogation en langage naturel de rapports financiers (PDF, CSV) via une
architecture RAG (Retrieval-Augmented Generation) et une base vectorielle
Pinecone. Le raisonnement et la synthèse finale sont assurés par un modèle
Claude (Anthropic).

## Comment ça marche (résumé)

```
data/*.pdf, *.csv
      │
      ▼
1. Chargement (PyPDFLoader / pandas)
      │
      ▼
2. Découpage en chunks (~1000 caractères, overlap 150)
      │
      ▼
3. Vectorisation (sentence-transformers, local, gratuit)
      │
      ▼
4. Indexation dans Pinecone (base vectorielle managée)

À la question de l'utilisateur :
5. La question est vectorisée avec le même modèle
6. Pinecone renvoie les 5 chunks les plus proches (recherche sémantique)
7. Les chunks + la question sont envoyés à Claude, qui rédige la réponse
   en citant ses sources
```

## Pourquoi ces choix

- **Embeddings locaux (sentence-transformers `all-MiniLM-L6-v2`)** : évite de
  dépendre d'une API d'embedding payante en plus de Claude et Pinecone.
  Suffisant pour une preuve de concept ; remplaçable plus tard par Voyage AI
  ou OpenAI si besoin de meilleure précision.
- **Pinecone (serverless)** : stockage vectoriel managé, pas d'infra à gérer,
  index créé automatiquement au premier lancement si absent.
- **Claude comme générateur** : reçoit uniquement les passages pertinents
  (pas tout le document) donc coût et latence maîtrisés ; le prompt système
  l'oblige à citer ses sources et à admettre l'absence d'information plutôt
  que d'halluciner.
- **PDF vs CSV traités différemment** : le PDF est découpé en texte libre
  (chunks de longueur fixe) ; le CSV est transformé ligne par ligne en phrase
  ("colonne: valeur, ...") car chaque ligne d'un tableau financier est déjà
  une unité de sens autonome (ex: une transaction, un poste de bilan).

## Installation

```bash
cd finance-rag-agent
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # puis renseigner vos clés API
```

Clés nécessaires dans `.env` :
- `ANTHROPIC_API_KEY` : console.anthropic.com
- `PINECONE_API_KEY` : app.pinecone.io

## Utilisation

1. Déposer vos fichiers dans `data/` (rapports PDF, exports CSV).
2. Indexer :
   ```bash
   python src/ingest.py
   ```
3. Interroger :
   ```bash
   python src/query.py "Quel est le chiffre d'affaires du T3 2025 ?"
   ```
   ou en mode interactif :
   ```bash
   python src/query.py
   ```

## Limites connues / pistes d'amélioration

- Pas de gestion de la mise à jour incrémentale : relancer `ingest.py`
  ré-upsert avec les mêmes IDs (`chunk-0`, `chunk-1`, ...), donc un nouveau
  jeu de documents plus petit laissera des chunks obsolètes dans l'index.
  Pour un usage réel, prévoir un `index.delete(delete_all=True)` avant
  ré-ingestion, ou des IDs stables par document.
- Recherche purement sémantique (pas de filtre par date/entreprise/source) :
  pour de gros volumes, ajouter des métadonnées filtrables dans Pinecone.
- Le découpage CSV ligne-par-ligne perd le contexte global du tableau
  (totaux, en-têtes multi-niveaux) : à affiner si les fichiers sont complexes.
