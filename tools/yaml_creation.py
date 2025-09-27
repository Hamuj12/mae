import os
import yaml

# --- Config ---
HDR_DIR = "/home/hm25936/hdris"   # change this to your HDRI folder
OUTPUT_FILE = "background_config.yml"
DEFAULT_STRENGTH = 100.0

# --- Collect HDRIs ---
backgrounds = []
for fname in sorted(os.listdir(HDR_DIR)):
    if fname.lower().endswith((".hdr", ".exr")):
        full_path = os.path.abspath(os.path.join(HDR_DIR, fname))
        backgrounds.append({
            "path": full_path,
            "strength": DEFAULT_STRENGTH
        })

# --- YAML Data ---
config = {
    "selection_mode": "random",
    "backgrounds": backgrounds
}

# --- Write YAML ---
with open(OUTPUT_FILE, "w") as f:
    yaml.dump(config, f, sort_keys=False)

print(f"✅ Wrote {len(backgrounds)} backgrounds to {OUTPUT_FILE}")