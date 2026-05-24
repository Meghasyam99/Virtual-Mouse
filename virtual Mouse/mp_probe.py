import importlib

mods = [
    "mediapipe",
    "mediapipe.solutions",
    "mediapipe.python",
    "mediapipe.python.solutions",
    "mediapipe.python.solutions.hands",
]

mp = importlib.import_module("mediapipe")
print("mediapipe.__version__ =", getattr(mp, "__version__", None))
print("hasattr(mediapipe, 'solutions') =", hasattr(mp, "solutions"))

for m in mods[1:]:
    try:
        importlib.import_module(m)
        print("OK:", m)
    except Exception as e:
        print("FAIL:", m, repr(e))
