from app.workshop import agent, retriever_tool, search_tool, weather_tool
import asyncio
import pytest 



def test_retriever_tool():
    query = "Agentic AI systems"
    response = retriever_tool.func(query)
    print("Retriever Tool Response:", response)
    assert response is not None
    assert isinstance(response, str) or isinstance(response, list)

def test_search_tool():
    query = "Latest advancements in AI in 2024"
    response = search_tool.func({"query": query})
    print("Search Tool Response:", response)
    assert response is not None
    assert isinstance(response, dict)

def test_weather_tool():
    city = "Berkane"
    response = weather_tool.func({"city": city})
    print("Weather Tool Response:", response)
    assert response is not None
    assert "temperature" in str(response).lower() or "weather" in str(response).lower()

def test_agent_invoke():
    query = "what's agentic system  according the retriever?"
    query = "What's the weather in Paris today?"
    response = asyncio.run(agent.arun(query))

    print("Agent Response:", response)
    assert "final_answer" in response.keys()
    assert response.get("final_answer","")


if __name__ == "__main__":
    print(test_agent_invoke())
    #print(test_retriever_tool())
    #print(test_search_tool())
    #print(test_weather_tool())