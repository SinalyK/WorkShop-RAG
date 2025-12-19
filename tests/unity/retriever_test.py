import pytest
from app.workshop import reranker_rag, fillful_data


def test_reranker_rag():
    query = "What is Agentic AI?"
    results = reranker_rag(query)
    
    assert results is not None, "Reranker returned None"
    assert isinstance(results, list), "Reranker output is not a list"
    assert len(results) > 0, "Reranker returned an empty list"
    
    # Check structure of first result
    #print("Reranker Results:", results)

def test_fillful_data():
    query = "Explain the concept of Agentic AI."
    context = fillful_data(query)
    
    assert context is not None, "fillful_data returned None"
    assert isinstance(context, list), "fillful_data output is not a list"
    assert len(context) > 0, "fillful_data returned an empty list"
    
    # Check structure of first context item
    print("Fillful Data Context:", context)




if __name__ == "__main__":
    #print(test_reranker_rag())
    print(test_fillful_data())