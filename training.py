from typing import TypedDict, List

import json

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from weathertools import WeatherTools

from mermaid import MermaidGraphGenerator

class AgentState(TypedDict):
    """ 
    I give that to the Agent to know data types
    """
    user_input: str
    response: str
    messages: List[BaseMessage]

class LLMAgent:
    """
    Agent that is going to use the LLM and use tools
    """

    def __init__(self, model_name="llama3.1", base_url="http://localhost:11434"):
        """ 
        Init agent
        """
        self.llm = ChatOllama(model=model_name, base_url=base_url)
        self.tool_node = WeatherTools.get_tool_node()
        self.graph = self.build_graph()

        # Generate graph (I use Mermaid)
        self.graph_visualizer = MermaidGraphGenerator(self.graph)
        self.graph_visualizer.generate_html()

        

    def build_graph(self):
        """
        Constructs the LangGraph state graph that orchestrates the agent's behavior.
        """
        graph = StateGraph(AgentState)  

        graph.add_node("process_request", self.process_request)
        graph.add_node("call_tool", self.tool_node)
        graph.add_node("generate_response", self.get_response)


        graph.add_edge(START, "process_request")  
        graph.add_edge("process_request", "call_tool") 
        graph.add_edge("call_tool", "generate_response")
        graph.add_edge("generate_response", END)  

        return graph.compile() 

    def decide_tool(self, user_input):
        """
        Analyzes the user's request and decides which tool to use.
        """
        prompt = f"""
        You are an intelligent weather assistant.
        - If the question is about "weather", use 'get_weather'.
        - If it asks for "temperature", use 'get_temperature'.
        - Otherwise, ask for more details.

        Example:
        - "What is the weather like in Paris?" → get_weather
        - "What is the temperature in New York?" → get_temperature
        - "Is it raining in San Francisco?" → get_weather

        User question: "{user_input}"
        Respond only with: get_weather or get_temperature.
        """

        tool_choice = self.llm.invoke(prompt).content.strip() # I get the tool that the user want.
        return tool_choice if tool_choice in ["get_weather", "get_temperature"] else None

    def process_request(self, state: AgentState) -> AgentState:
        user_input = state["user_input"]

        extracted_location = WeatherTools.get_location.invoke(user_input)

        print(f"Extracted location: {extracted_location}")
        tool_choice = self.decide_tool(user_input)

        state["messages"].append(HumanMessage(content=user_input))

        print(f"The tool choice is {tool_choice}")
        print(f"User input: {user_input}")

        if tool_choice:
            tool_call_message = AIMessage(
                content="",
                tool_calls=[{
                    "name": tool_choice,
                    "args": {"location": extracted_location.strip().lower()},
                    "id": "tool_call_1",
                    "type": "tool_call",
                }]
            )

            state["messages"].append(tool_call_message)
            state["response"] = tool_call_message
        else:
            clarification_message = AIMessage(content="I don't understand your request. Could you clarify?")
            state["messages"].append(clarification_message)
            state["response"] = clarification_message

        return state


    def get_response(self, state: AgentState) -> AgentState:
        """
        Get the response from the LLM based on the state.
        """
        messages = state["messages"]

        response = self.llm.invoke(messages)
        ai_response = AIMessage(content=response.content)

        state["messages"].append(ai_response)
        state["response"] = ai_response

        return state

    def ask(self, user_input):
        print(f">> - User: {user_input}")

        initial_state = {
            "user_input": user_input,
            "response": "",
            "messages": []
        }

        final_state = self.graph.invoke(initial_state)
        response = final_state["response"]

        print(f">> - Agent: {response.content}")

        return response.content



agent = LLMAgent()

agent.ask("What is the weather at San Francisco ?")
