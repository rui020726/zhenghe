import random, os
random.seed(20241113)
maps = [f"img_{i}.png" for i in range(6000, 10000)]
subset = random.sample(maps, 250)
with open("test_manifest.txt", "w") as f:
    for m in subset:
        f.write(m + "\n")


