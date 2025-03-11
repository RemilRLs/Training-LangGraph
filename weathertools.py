from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

class WeatherTools:
    """
    Class that contain all tools to get the weather and that can be used by the agent
    """

    llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434")

    @tool
    def get_weather(location: str):
        """ Sends the weather for a given location 

        :param location: Location name
        """

        print(f"[+] Executing get_weather with location: {location}")
        weather_data = {
            "paris": "It is 15°C and sunny in Paris.",
            "san francisco": "It is 12°C and foggy in San Francisco.",
            "new york": "It is 25°C and clear in New York."
        }

        return weather_data.get(location.lower(), f"Je ne connais pas la météo pour {location}.")

    @tool
    def get_temperature(location: str):
        """
        Get the temperature for a given location

        :param location: Location name
        """
        temperature_data = {
            "paris": "15°C",
            "san francisco": "12°C",
            "new york": "25°C"
        }
        return temperature_data.get(location.lower(), f"Température inconnue pour {location}.")

    @tool 
    def get_location(user_input: str):
        """ 
        Return the location of the user where he wants to know the weather

        :param user_input: User input
        """
        prompt = f"""
        You are an advanced natural language processing assistant.
        Your task is to extract the **city or location name** from the user's input.

        - The user may ask about the weather, temperature, or conditions in a specific place.
        - If a **city, country, or location** is mentioned, extract it **as is**.
        - If multiple locations are mentioned, extract the **most relevant one** based on the context.
        - If no location is found, return **"Unknown"**.

        ### **User Input:**
        "{user_input}"

        ### **Your Output (Location Only, Nothing Else):**
        """
        location = WeatherTools.llm.invoke(prompt).content.strip()

        return location if location else "Unknown"


    @classmethod
    def get_tool_node(cls):
        """
        Return list of tools
        """
        tools = [cls.get_weather, cls.get_temperature]
        return ToolNode(tools)