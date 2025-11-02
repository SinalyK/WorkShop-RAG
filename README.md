# WORSHOP RAG
This workshop build a Retrieval-Augmented Generation (RAG) system,an intelligent agent that can search external documents,retrieve relevant information and generate answers grounded in that data.

## Document to retrieve "Building Agentic Al Systems by Matthew R. Scott and Dr. Alex Acero 288 pages"

### Chunking:Lanchain-text-splitters(500 tokens)
### Encoders: 
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
### VectorStore (Chormadb-clientpersistent)
### Groq(Modèle à definir )
#### Retrieving (topk et self-retrieving)
#### LLM judge
#### Agent(LangChain-Agent)
#### Tools(Tavily,RAG)


