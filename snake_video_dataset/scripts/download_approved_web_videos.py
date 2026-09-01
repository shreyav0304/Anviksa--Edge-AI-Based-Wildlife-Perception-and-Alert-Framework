"""Download only pre-approved, license-verified web videos from the manifest."""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "metadata/source_manifest.csv"
BASE = ROOT / "source_videos/web_collected"
URLS = {
    "WEB_V001": "https://upload.wikimedia.org/wikipedia/commons/4/46/Colubrine_sea_krait_%28Laticauda_colubrina%29.webm",
    "WEB_V002": "https://upload.wikimedia.org/wikipedia/commons/c/c7/Sea_Snake_eating_Moray_Eel%2C_Fiji_%28Laticauda_colubrina_vs._Gymnothorax_sp.%29.webm",
    "WEB_V003": "https://upload.wikimedia.org/wikipedia/commons/b/bb/Iran_-_Pseudocerastes_urarachnoides_2015.webm",
    "WEB_V004": "https://upload.wikimedia.org/wikipedia/commons/9/9c/Timber_rattlesnake_%28Crotalus_horridus%29_in_Pennsylvania.webm",
    "WEB_NV001": "https://upload.wikimedia.org/wikipedia/commons/2/21/Blackneck_garter_snake_%28Thamnophis_cyrtopsis%29.webm",
    "WEB_NV002": "https://upload.wikimedia.org/wikipedia/commons/6/6d/Dice_snake_%28Natrix_tessellata%29_hiding_beneath_a_rock_underwater.webm",
    "WEB_NV003": "https://upload.wikimedia.org/wikipedia/commons/1/17/Elaphe_climacophora_-_Japanese_rat_snake_-_2015_10_4.webm",
    "WEB_NV004": "https://upload.wikimedia.org/wikipedia/commons/b/b4/Ball_python_%28Python_regius%29_in_a_zoo.webm",
}


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
        fields = list(rows[0])
    for row in rows:
        if row["download_allowed"] != "YES":
            continue
        if row["license_verification_status"] != "VERIFIED" or row["label_verification_status"] != "VERIFIED":
            raise RuntimeError(f"Approval contract failed for {row['source_id']}")
        destination = BASE / row["assigned_class"] / row["local_filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size > 0:
            row["download_status"] = "DOWNLOADED"
            continue
        request = urllib.request.Request(URLS[row["source_id"]], headers={"User-Agent": "AnviksaResearchDataset/1.0"})
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if temporary.stat().st_size <= 0:
                raise RuntimeError("empty download")
            temporary.replace(destination)
            row["download_status"] = "DOWNLOADED"
        except Exception:
            row["download_status"] = "DOWNLOAD_FAILED"
            if temporary.exists():
                temporary.unlink()
    with MANIFEST.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
