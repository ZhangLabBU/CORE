from src.agents.tools import Tool
from src.docstore.Meteorology import MeteorologyAgent


import time
from typing import Union
import asyncio


tools = [
    Tool(
        name="AnalyzeWeather",
        func=MeteorologyAgent().analyze_weather_async,
        description=(
            "AnalyzeWeather(longitude: float, latitude: float) -> str:\n"
            " - Analyzes the weather information for a given location.\n"
            " - `longitude`: The longitude of the location.\n"
            " - `latitude`: The latitude of the location.\n"
            " - Returns a string with weather details including conditions, temperature, humidity, and wind speed."
        ),
        stringify_rule=lambda args: f"AnalyzeWeather({args[0]}, {args[1]})",
    ),
    Tool(
        name="EvaluateImpact",
        func=MeteorologyAgent().evaluate_impact_async,
        description=(
            "EvaluateImpact(weather_info: str, event_type: str) -> str:\n"
            " - Evaluates the impact of weather conditions on specific events such as 'fire' or 'explosion'.\n"
            " - `weather_info`: A string containing the weather information, including conditions and wind speed.\n"
            " - `event_type`: The  only two types of events to choose, 'fire' or 'explosion'.\n"
            " - Returns an analysis of the weather's potential influence on the event and suggested actions."
        ),
        stringify_rule=lambda args: f"EvaluateImpact({args[0]}, {args[1]})",
    ),
    Tool(
        name="AnalyzeRescueImpact",
        func=MeteorologyAgent().analyze_rescue_impact_async,
        description=(
            "AnalyzeRescueImpact(weather_info: str) -> str:\n"
            " - Analyzes the impact of current weather conditions on rescue operations.\n"
            " - `weather_info`: A string containing weather details including conditions and wind speed.\n"
            " - Returns an analysis of how weather conditions may affect rescue activities and provides operational advice."
        ),
        stringify_rule=lambda args: f"AnalyzeRescueImpact({args[0]})",
    ),
]
