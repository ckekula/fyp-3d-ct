import json
data = json.loads(open("patches/_patch_checkpoint_abnormal.json").read())  # adjust path/tag as needed
coords = data["coords"]
print(len(coords))
bad = [c for c in coords if len(c) != 3 or any(not isinstance(x, int) for x in c)]
print(len(bad), bad[:5])
maxval = max(x for c in coords for x in c)
minval = min(x for c in coords for x in c)
print(maxval, minval)