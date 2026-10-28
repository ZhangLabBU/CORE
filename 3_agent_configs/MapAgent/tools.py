from src.agents.tools import Tool
from src.docstore.Map import MapAgent


import time
from typing import Union
import asyncio



tools = [
    Tool(
        name="Map",
        func=MapAgent().map_async,
        description=(
            "Map(search_target: str, longitude: float, latitude: float) -> str:\n"
            " - Returns information about nearby services such as fire stations, police stations, and hospitals based on the provided search target.\n"
            " - Supports synchronous calls to get details about fire stations, police stations, and hospitals.\n"
            " - `search_target`: A string specifying the service type, e.g., 'fire station', 'police station', 'hospital'.\n"
            " - `longitude`: The longitude of the current location.\n"
            " - `latitude`: The latitude of the current location.\n"
            " - Returns a string containing formatted information about the relevant service."
        ),
        stringify_rule=lambda args: f"Map({args[0]}, {args[1]}, {args[2]})",
    ),
    Tool(
        name="FindNearestContact",
        func=MapAgent().find_fastest_contact_async,
        description=(
            "FindNearestContact(map_result: str) -> str:\n"
            " - Extracts the phone number of the nearest target (e.g., hospital, fire station, police station) from the results provided by the Map tool.\n"
            " - Supports parsing and identifying the closest service based on the returned details from the Map function.\n"
            " - `map_result`: The result string returned by the Map tool, which contains information about various nearby services.\n"
            " - Returns a string containing the phone number of the nearest service, or an error message if the phone number cannot be found."
        ),
        stringify_rule=lambda args: f"FindNearestContact({args[0]})",
    ),
    Tool(
        name="Dial",
        func=MapAgent().dial_async,
        description=(
            "Dial(phone_number: str) -> str:\n"
            " - Call the nearest hospital or police or fire station for help."
            " - Supports synchronous calling of emergency services based on the provided phone number.\n"
            " - `phone_number`: The phone number of the nearest hospital or police or fire station, e.g.,'010-68790499'.\n"
            " - Returns a string indicating whether the call was successful and which service was contacted."
        ),
        stringify_rule=lambda args: f"Dial({args[0]})",
    ),
]
