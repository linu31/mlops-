"""Generate reproducible synthetic pond-night datasets."""
from pathlib import Path
import csv, random

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["pond_id","pond_area_acres","stocking_density_per_m2","do_8pm_mg_l","water_temp_c","daily_feed_kg","planned_aerator_hours","expected_rainfall_mm","oxygen_risk"]

def make_row(i, rainy=False, rng=None):
    rng = rng or random
    area = round(rng.uniform(0.5, 3.0), 2)
    density = round(rng.uniform(60, 150), 1)
    temp = round(rng.uniform(26, 33), 1)
    feed = round(area*density*rng.uniform(0.0007, 0.00125), 2)
    aer = round(rng.uniform(2, 8), 1)
    rain = round(rng.uniform(8, 55) if rainy else rng.uniform(0, 12), 1)
    do = round(rng.uniform(5.0, 8.2) - (0.15 if rainy else 0), 2)
    biomass = area*4046.86*density/1000
    feed_ratio = feed/max(biomass, 0.01)
    score = (density-100)/22 + (temp-29)/2.8 + (feed_ratio-0.0009)*1800 + (5-aer)/2.2 + (rain-8)/18 - (do-6.3)*1.5
    if rainy: score += 0.85
    score += rng.gauss(0, 0.75)
    return [f"POND-{i%18+1:02d}", area, density, do, temp, feed, aer, rain, "High" if score > 0.45 else "Normal"]

def generate():
    random.seed(231)
    for filename, rainy, offset in [("pond_nights_v1.csv", False, 0), ("pond_nights_v2_rainy.csv", True, 300)]:
        rows = [make_row(i+offset, rainy) for i in range(300)]
        path = ROOT/"data"/filename
        with path.open("w", newline="", encoding="utf-8") as f:
            writer=csv.writer(f); writer.writerow(FIELDS); writer.writerows(rows)
    print("Generated V1 and rainy-period V2 (300 rows each).")

if __name__ == "__main__":
    generate()
