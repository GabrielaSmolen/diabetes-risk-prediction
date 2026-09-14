from pathlib import Path
import pandas as pd

COLUMNS = [
    'pregnancies',
    'glucose',
    'blood_pressure',
    'skin_thickness',
    'insulin',
    'bmi',
    'diabetes_pedigree',
    'age',
    'outcome'
]

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "diabetes.csv"


def load_data() -> pd.DataFrame:
    """Loads dataset from local cache or downloads it if not present."""
    DATA_DIR.mkdir(exist_ok=True)

    if not DATA_FILE.exists():
        print("Downloading raw dataset...")
        df = pd.read_csv(URL, names=COLUMNS)
        df.to_csv(DATA_FILE, index=False)
    else:
        df = pd.read_csv(DATA_FILE)

    return df
