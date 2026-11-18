"""Project-wide constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "Customer_Data.csv"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

SEED = 42
ID_COL = "CUST_ID"
K_RANGE = range(2, 11)
SKEW_THRESHOLD = 1.0

# Palette used across figures and the app.
PALETTE = ["#B71C1B", "#5B6B8C", "#A6AD3C", "#D5A021", "#8C564B", "#17BECF", "#798517", "#D72A2B"]
