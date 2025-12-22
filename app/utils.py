from termcolor import colored  # pip install termcolor
import textwrap


def verbose_context(documents):
    """
    Affiche les documents récupérés avec un style similaire au verbose de LangChain.
    """
    if not documents:
        print(colored("> No documents retrieved.", "red"))
        return

    print(colored(f"\n> Retrouvé {len(documents)} documents:", "yellow", attrs=["bold"]))

    for i, doc in enumerate(documents):
        # En-tête du document
        header = f"--- [Document {i+1}] ------------------------------------------------"
        print(colored(header, "blue"))

        # Affichage des métadonnées (Source, page, etc.)
        if hasattr(doc, "metadata") or (isinstance(doc, tuple)):
            meta_str = f"Metadata: {doc[0]['metadata'] if isinstance(doc, tuple) else doc.metadata}"
            print(colored(meta_str, "cyan"))

        # Affichage du contenu
        if hasattr(doc, "page_content") or (isinstance(doc, tuple)):
            print(colored("\nContent:", "white", attrs=["bold"]))
            # Wrap du texte pour qu'il ne dépasse pas 100 caractères de large (lisibilité)
            wrapped_content = textwrap.fill(
                doc[0]["page_content"] if isinstance(doc, tuple) else doc.page_content, width=100
            )
            print(colored(wrapped_content, "green"))

        # Séparateur de fin
        print(colored("=" * 100, "blue"))
        print("\n")


def fillful_data(retriever, reranker_rag, query, template, reranker=True):

    if not query:
        return "query must be a string"

    if not reranker:
        context = retriever.invoke(query)
        verbose_context(context)
        return template.format(user_query=query, rag_context=context if context else "None")

    context = reranker_rag(query)
    verbose_context(context)
    return template.format(user_query=query, rag_context=context if context else "None")


def fillful_data_agent(query):

    if not query:
        return "query must be a string"

    context = reranker_rag(query)
    # verbose_context(context)
    # return template.format(user_query=query, rag_context=context if context else "None")
    return context
