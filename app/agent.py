import json
import re
import traceback
from logging import Logger
from typing import Any, Dict, List, Optional, TypedDict
from google.api_core.exceptions import ResourceExhausted
import backoff
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from app.prompt import system_prompt_fast, final_formatted_prompt, final_formatted_prompt_

logger = Logger(__name__)


# State definition
class AgentState(TypedDict):
    messages: List[BaseMessage]
    user_question: str
    current_step: str
    thought: str
    action: str
    action_input: str
    observation: str
    final_answer: str
    iteration_count: int
    max_iterations: int


class MarkdownAnswer(BaseModel):
    markdown_answer: str


class ReActAgent:
    def __init__(
        self,
        llm: BaseChatModel = None,
        llm_nd: BaseChatModel = None,
        tools: List[BaseTool] = [],
        max_iterations: int = 20,
    ):
        self.llm = llm
        self.max_iterations = max_iterations
        self.steps: BaseMessage = []
        self.documents = []
        self.state: AgentState = None
        self.llm_with_structured_output = llm.with_structured_output(MarkdownAnswer)

        self.tool_map = {tool.name: tool for tool in tools}
        # tool_names = [tool.name for tool in tools]

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph"""
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("start", self._start_node)
        graph.add_node("plan", self._plan_node)
        graph.add_node("execute_tool", self._execute_tool_node)
        graph.add_node("reflect", self._reflect_node)
        graph.add_node("finish", self._finish_node)

        # Add edges
        graph.add_edge("start", "plan")
        graph.add_conditional_edges(
            "plan",
            self._should_continue,
            {"continue": "execute_tool", "finish": "finish"},
        )
        graph.add_edge("execute_tool", "reflect")
        graph.add_conditional_edges(
            "reflect",
            self._should_continue_after_reflection,
            {"continue": "plan", "finish": "finish", "max_iterations": "finish"},
        )
        graph.add_edge("finish", END)

        # Set entry point
        graph.set_entry_point("start")

        return graph.compile()

    async def _start_node(self, state: AgentState) -> AgentState:
        """Initialize the agent state"""

        self.state = {
            **state,
            "current_step": "start",
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
        }
        return {
            "current_step": "start",
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
        }

    async def _plan_node(self, state: AgentState) -> AgentState:
        """Plan the next action using the LLM"""
        # Build the conversation context

        messages = await self._build_conversation_history(state)

        # Add system prompt to guide next step
        final_prompt = system_prompt_fast.format(
            tools_description="\n".join(
                [
                    f"{i+1}. {tool.name}: {tool.description}"
                    for i, tool in enumerate(self.tool_map.values())
                ]
            ),
            tools_name=", ".join([tool.name for tool in self.tool_map.values()]),
        )

        full_messages = [HumanMessage(content=final_prompt)] + messages

        # Get LLM response
        response = None

        try:
            logger.info("Executing LLM plan...")

            response = await self._safe_invoke(full_messages)
        except Exception as e:
            print(f"Exception: {e}")
            response = await self.llm.ainvoke(full_messages)

        # Parse the response
        print("LLM Response:", response.content)
        parsed_response = await self._parse_llm_response(response.content)
        print("Parsed Response:", parsed_response)

        # get response final in final_answer when the response has any reasoning fields
        if not any(parsed_response.values()) and (response and response.content):
            parsed_response["final_answer"] = str(response.content)

        print("node plan executed with success")

        self.state = {
            **self.state,
            "current_step": "plan",
            "thought": parsed_response.get("thought", ""),
            "action": parsed_response.get("action", ""),
            "action_input": parsed_response.get("action_input", ""),
            "final_answer": parsed_response.get("final_answer", ""),
            "messages": state["messages"] + [response],
        }
        return {
            "current_step": "plan",
            "max_iterations": self.state["max_iterations"],
            "iteration_count": state["iteration_count"],
        }

    async def _execute_tool_node(self, state: AgentState) -> AgentState:
        """Execute the selected tool"""
        logger.info("Executing tool...")

        action = await self._clean_tool_text(self.state["action"])
        action_input = self.state["action_input"]
        observation = ""

        try:
            # Parse action input if it's a string
            if isinstance(action_input, str):
                try:
                    action_input = json.loads(action_input)
                except json.JSONDecodeError:
                    # If it's not valid JSON, use it as is for single parameter tools
                    pass

            # Execute the tool with specific handling for each tool type
            if action in self.tool_map:
                tool = self.tool_map[action]

                # Handle ChromaRetriever (expects query parameter)
                if action == "Retriever":

                    if isinstance(action_input, dict):
                        query = action_input.get("query", str(action_input))
                    else:
                        query = str(action_input)
                    observation = tool.func(query)

                elif action == "TavilySearch":

                    if isinstance(action_input, dict):
                        query = action_input
                    else:
                        query = str(action_input)
                    try:
                        observation = tool.func(query)
                    except Exception as e:
                        logger.info(f"Erreur lors de l'exécution de TavilySearch")
                        self.tool_map[action] = tool
                        observation = tool.func(query)

                # Handle WeatherAction (expects city name)
                elif action == "WeatherAction":

                    if isinstance(action_input, dict):
                        observation = tool.func(action_input.get("city", ""))
                    else:
                        # Try to parse as JSON for WeatherAction
                        try:
                            parsed_input = json.loads(str(action_input))
                            observation = tool.func(parsed_input.get("city", ""))
                        except Exception as e:
                            observation = f"Error: WeatherAction requires JSON input with 'url', 'method', and optional 'payload', {e}"

                # Generic tool execution
                else:
                    if isinstance(action_input, dict):
                        observation = tool.func(action_input)
                    else:
                        observation = tool.func(action_input)
            else:
                if action != "":
                    observation = f"Error: Unknown action '{action}'. Available actions: {list(self.tool_map.keys())}"

        except Exception as e:
            observation = f"Error executing {action}: {str(e)}"

        self.state = {
            **self.state,
            "current_step": "execute_tool",
            "observation": str(observation),
            "iteration_count": state["iteration_count"] + 1,
        }

        return {
            "max_iterations": self.state["max_iterations"],
            "current_step": "execute_tool",
            "iteration_count": state["iteration_count"] + 1,
        }

    async def _reflect_node(self, state: AgentState) -> AgentState:
        """Reflect on the observation and update state"""
        logger.info("Reflecting on observation...")

        # Check for authentication errors
        observation = self.state["observation"]
        if "401" in observation or "error" in observation.lower():
            self.state = {
                **self.state,
                "current_step": "reflect",
                "final_answer": f"error: {observation}",
            }
            return {
                "max_iterations": self.state["max_iterations"],
                "iteration_count": state["iteration_count"],
                "current_step": "reflect",
            }

        self.state = {
            **self.state,
            "current_step": "reflect",
        }
        return {
            "iteration_count": self.state["iteration_count"],
            "current_step": "reflect",
            "max_iterations": self.state["max_iterations"],
        }

    async def _finish_node(self, state: AgentState) -> AgentState:
        """Finish the agent execution"""
        logger.info("Finishing agent execution...")

        # memory management
        self.steps = []
        groq = True

        try:
            final_prompt = (
                final_formatted_prompt.format(
                    question=self.state.get("user_question", ""),
                    answer=self.state.get("final_answer", ""),
                )
                + final_formatted_prompt_
            )
            final_answer = await self._safe_final_invoke(
                [HumanMessage(content=final_prompt)], structured_output=False
            )
            if isinstance(final_answer, MarkdownAnswer) or 1:
                print("bien de utilisé, final answer")

                if not groq:
                    content = final_answer.markdown_answer
                else:
                    try:
                        data = json.loads(final_answer.content)
                        content = [d for d in data.values()][-1]
                    except Exception as e:
                        content = final_answer.content

                self.state["final_answer"] = content
        except Exception as e:
            traceback.print_exc()
            print(f"l'erreur de appel final: {e}")

        return {
            **self.state,
            "iteration_count": self.state["iteration_count"],
            "current_step": "finish",
            "max_iterations": self.state["max_iterations"],
        }

    async def _should_continue(self, state: AgentState) -> str:
        """Decide whether to continue or finish based on current state"""
        # Check if we have a final answer
        if self.state.get("final_answer") and self.state.get("action") == "":
            return "finish"

        # Check if we have an action to execute
        if self.state.get("action") and self.state.get("action") != "":
            return "continue"

        # Check max iterations
        if state["iteration_count"] >= state["max_iterations"]:
            return "finish"

        return "continue"

    async def _should_continue_after_reflection(self, state: AgentState) -> str:
        """Decide whether to continue after reflection"""
        # Check if we have a final answer
        if self.state.get("final_answer") and self.state.get("action") == "":
            return "finish"

        # Check max iterations
        if state["iteration_count"] >= state["max_iterations"]:
            return "max_iterations"

        # Check for authentication errors
        observation = self.state.get("observation", "")
        if "401" in observation or "authentication" in observation.lower():
            return "finish"

        return "continue"

    async def _build_conversation_history(self, state: AgentState) -> List[BaseMessage]:
        """Build the conversation history for the LLM"""

        # human - agent memory
        history = []

        if state.get("iteration_count") == 0:

            if len(history) > 2:
                history = [
                    f"summary of previous conversations: {await self._summarize_history(history)}"
                ]

        # ReAct steps memory
        # messages = history + [
        #     HumanMessage(content=f"Question: {state['user_question']}")
        # ]

        # Add previous steps
        messages = self.steps

        if self.state.get("thought"):
            messages.append(AIMessage(content=f"Thought: {self.state['thought']}"))
            self.steps.append(AIMessage(content=f"Thought: {self.state['thought']}"))

        if self.state.get("action"):
            messages.append(AIMessage(content=f"Action: {self.state['action']}"))
            self.steps.append(AIMessage(content=f"Action: {self.state['action']}"))

        if self.state.get("action_input"):
            messages.append(AIMessage(content=f"Action Input: {str(self.state['action_input'])}"))
            self.steps.append(AIMessage(content=f"Action Input: {str(self.state['action_input'])}"))

        if self.state.get("observation"):
            messages.append(HumanMessage(content=f"Observation: {str(self.state['observation'])}"))
            self.steps.append(
                HumanMessage(content=f"Observation: {str(self.state['observation'])}")
            )

        print(f"Messages reconstruits : {messages}")
        return messages

    async def _summarize_history(self, history: List[BaseMessage]) -> str:
        """Summarize the conversation history."""
        if not history:
            return "No previous conversations found."

        return await self._safe_invoke(f"Summarize the following conversation history:\n{history}")

    async def _parse_llm_response(self, response: str) -> Dict[str, str]:
        """Parse the LLM response to extract thought, action, etc."""
        result = {"thought": "", "action": "", "action_input": "", "final_answer": ""}

        # Parse using regex patterns
        patterns = {
            "thought": r"Thought:\s*((?:.|\n)*?)(?=\n*(Action:|Final Answer:|$))",
            "action": r"Action:\s*(.*?)\s*(?=\n*(Action Input:|Thought:|Final Answer:|Observation:|$))",
            "action_input": r"Action Input:\s*((?:.|\n)*?)(?=\n*(Observation:|Thought:|Final Answer:|Action:|$))",
            "final_answer": r"(?:Final Answer:|Agent Response:)\s*((?:.|\n)*?)(?=\n*(Thought:|Action:|Action Input:|Retrieve:|Observation:|$))",
        }

        async def normalize_response(text: str) -> str:
            text = re.sub(r"(Action:\s*\w+)(Action Input:)", r"\1\n\2", str(text))
            text = re.sub(r"(Action Input:\s*\{[^}]+\})(Thought:)", r"\1\n\2", str(text))
            text = re.sub(r"(Action Input:\s*\{[^}]+\})(Observation:)", r"\1\n\2", str(text))
            text = re.sub(r"(Action Input:\s*\{[^}]+\})(Action:)", r"\1\n\2", str(text))

            text = text.replace("```json", "").replace("```", "")
            return text

        response = await normalize_response(response)

        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                result[key] = match.group(1).strip()

        return result

    def run(self, question: str) -> Dict[str, Any]:
        """Run the agent with a question"""
        initial_state = {
            "messages": [],
            "user_question": question,
            "current_step": "",
            "thought": "",
            "action": "",
            "action_input": "",
            "observation": "",
            "final_answer": "",
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
        }

        # Execute the graph
        final_state = self.graph.invoke(initial_state, {"recursion_limit": 100})

        return {
            "question": question,
            "final_answer": final_state.get("final_answer", "No final answer generated"),
            "iterations": final_state.get("iteration_count", 0),
            "execution_path": self._extract_execution_path(final_state),
        }

    async def arun(self, question: str) -> Dict[str, Any]:
        """Run the agent with a question"""
        initial_state = {
            "messages": [],
            "user_question": question,
            "current_step": "",
            "thought": "",
            "action": "",
            "action_input": "",
            "observation": "",
            "final_answer": "",
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
        }

        # Execute the graph
        final_state = await self.graph.ainvoke(initial_state, {"recursion_limit": 100})

        return {
            "question": question,
            "final_answer": final_state.get("final_answer", "No final answer generated"),
            "iterations": final_state.get("iteration_count", 0),
            "execution_path": self._extract_execution_path(final_state),
        }

    def _extract_execution_path(self, state: AgentState) -> List[Dict[str, str]]:
        """Extract the execution path from the final state"""
        path = []
        if state.get("thought"):
            path.append({"step": "thought", "content": state["thought"]})
        if state.get("action"):
            path.append({"step": "action", "content": state["action"]})
        if state.get("action_input"):
            path.append({"step": "action_input", "content": state["action_input"]})
        if state.get("observation"):
            path.append({"step": "observation", "content": state["observation"]})
        return path

    def _rotate_llm(self):
        if not isinstance(self.llm, ChatGoogleGenerativeAI):
            self.llm = ChatGroq(model="openai/gpt-oss-20b")
        else:
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    @backoff.on_exception(backoff.expo, (ResourceExhausted, Exception), max_tries=7, jitter=None)
    async def _safe_invoke(self, full_messages):
        try:
            chunks = []
            async for chunk in self.llm.astream(full_messages):
                if chunk.content:
                    chunks.append(chunk.content)

            return AIMessage(content="".join(chunks))
            # return await self.llm.ainvoke(full_messages)
        except ResourceExhausted as e:
            print("[Quota] Clé API dépassée, on change...")
            self._rotate_llm()
            raise e
        except Exception as e:
            print(f"[Erreur] {e}, on essaye un autre LLM...")
            self._rotate_llm()
            raise e

    @backoff.on_exception(backoff.expo, (ResourceExhausted, Exception), max_tries=7, jitter=None)
    async def _safe_final_invoke(self, full_messages, structured_output=True):
        try:
            if structured_output:
                return await self.llm_with_structured_output.ainvoke(full_messages)
            else:
                return await self.llm.ainvoke(full_messages)
        except ResourceExhausted as e:
            print("[Quota] Clé API dépassée, on change...")
            self._rotate_llm()
            raise e
        except Exception as e:
            print(f"[Erreur] {e}, on essaye un autre LLM...")
            self._rotate_llm()
            raise e

    async def _clean_tool_text(self, text: str) -> str:
        try:
            text = text.strip()
            text = text.encode("utf-8").decode("unicode_escape")
            text = text.replace("\\", "")
            text = re.sub(r"[^\x20-\x7E]+", "", text)
            text = re.sub(r"[^a-zA-Z]", "", text)

            if "weatheraction" in text.lower():
                text = "WeatherAction"
            if "retriever" in text.lower():
                text = "Retriever"
            if "tavilysearch" in text.lower():
                text = "TavilySearch"

            print(f"Action nettoyée : {text}")
        except:
            pass
        return text
