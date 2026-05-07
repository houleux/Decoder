import json
import base64
import os

notebook_path = "RELDEC/notebook_runs/continuous_reldec/active_run/wran/plot_eval_results.ipynb"

with open(notebook_path, "r") as f:
    nb = json.load(f)

plot_num = 1
for cell in nb["cells"]:
    if "subset" in str(cell.get("id", "")) or "# Subset" in "".join(cell.get("source", [])):
        for output in cell.get("outputs", []):
            if "data" in output and "image/png" in output["data"]:
                image_data = output["data"]["image/png"]
                if isinstance(image_data, list):
                    image_data = "".join(image_data)
                
                filename = f"subset_plot_{plot_num}.png"
                with open(filename, "wb") as img_f:
                    img_f.write(base64.b64decode(image_data))
                print(f"Saved {filename}")
                plot_num += 1

