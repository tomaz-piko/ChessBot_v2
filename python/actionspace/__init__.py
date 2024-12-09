import json
import os

current_dir = os.path.dirname(__file__)

map_w, map_b = None, None
with open(f"{current_dir}/map_w.json") as f:
    map_w = json.load(f)

with open(f"{current_dir}/map_b.json") as f:
    map_b = json.load(f)