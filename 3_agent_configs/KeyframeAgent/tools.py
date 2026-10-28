# /home/yzt/AgentDNS/configs/KeyframeAgent/tools.py
from src.agents.tools import Tool
from src.docstore.KeyframeRecognition import KeyframeRecognition

recognition = KeyframeRecognition()

def _sr_keyframe_recognition(args):
    a0 = args[0] if len(args) > 0 else '""'
    a1 = args[1] if len(args) > 1 else '""'
    return f"KeyframeRecognition({a0}, {a1})"

tools = [
    Tool(
        name="KeyframeRecognition",
        func=recognition.keyframe_recognition_async,
        description=(
            "KeyframeRecognition(image_paths_str: str, instruction: str) -> list:\n"
            " - Analyze image(s) with instruction and return results."
        ),
        stringify_rule=_sr_keyframe_recognition,
    ),
]

print("[LOADED TOOLS] configs/KeyframeAgent/tools.py loaded")
print("[LOADED TOOLS] names =", [t.name for t in tools])
