import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

DATA_SET_URL = "./data/original_set.csv"

EXPORT=True
EXPORT_FILE="./data/cleaned_set.csv"

#* Load the dataset from CSV file
df = pd.read_csv(DATA_SET_URL)

#* Drop unnecessary columns that won't be used in analysis
df.drop(labels=["Titre","Localisation"], axis=1, inplace=True)


#* Rename Columns
df = df.rename(columns={
    'Prix': 'price',
    'Année-Modèle': 'model_year',
    'Boite de vitesses': 'transmission',
    'Type de carburant': 'fuel_type',
    'Kilométrage': 'mileage',
    'Marque': 'brand',
    'Modèle': 'model',
    'Nombre de portes': 'number_of_doors',
    'Origine': 'origin',
    'Première main': 'first_owner',
    'Puissance fiscale': 'tax_horsepower',
    'État': 'condition',
    'ABS': 'abs',
    'Airbags': 'airbags',
    'CD/MP3/Bluetooth': 'multimedia',
    'Caméra de recul': 'backup_camera',
    'Climatisation': 'air_conditioning',
    'ESP': 'esp',
    'Jantes aluminium': 'aluminum_wheels',
    'Limiteur de vitesse': 'speed_limiter',
    'Ordinateur de bord': 'onboard_computer',
    'Radar de recul': 'parking_sensors',
    'Régulateur de vitesse': 'cruise_control',
    'Sièges cuir': 'leather_seats',
    'Système de navigation/GPS': 'navigation_gps',
    'Toit ouvrant': 'sunroof',
    'Verrouillage centralisé à distance': 'remote_central_locking',
    'Vitres électriques': 'power_windows'
})

#* Remove rows where 'brand' is missing, as it's essential
df.dropna(subset=["brand"], inplace=True)

#* Replace zero values in 'number_of_doors' with NaN, to impute them later
df["number_of_doors"].replace(0, np.nan)

#* Impute missing 'number_of_doors' using the mean strategy
impute_stratigy = SimpleImputer(strategy="mean")
df["number_of_doors"] = impute_stratigy.fit_transform(df["number_of_doors"].values.reshape(-1, 1))

#* Convert the imputed values from float to integer
df["number_of_doors"] = df["number_of_doors"].astype(int)

#* handle origin
df = df[df['origin'].isin(['WW au Maroc', 'Dédouanée', 'Importée neuve','Pas encore dédouanée'])]


#* Normalize 'first_owner' column: keep only "Oui" and "Non", replace others with "Non"
df["first_owner"] = df["first_owner"].where(df["first_owner"].isin(["Oui","Non"]), "Non")
df["first_owner"] = df["first_owner"].replace("Oui",True).replace("Non",False)

#* Drop rows with missing 'tax_horsepower' values
df = df.dropna(subset=["tax_horsepower"])

#* Drop rows with missing 'condition' values, as condition is important
df = df.dropna(subset=["condition"])

#* Clean 'price' column:
#  1. Remove 'DH' and special characters, strip whitespace
df["price"] = df["price"].astype(str).str.replace("DH", "").str.replace("\u202f", "").str.strip()

#  2. Convert cleaned values to numeric, invalid strings become NaN
df["price"] = pd.to_numeric(df["price"], errors='coerce')

#  4. Convert price values to integers
df["price"] = df["price"].astype(int)

#* Handle outliers in 'price' using IQR method
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]

#* Clip 'price' to within IQR bounds and store in a new column
df['price'] = df['price'].clip(lower_bound, upper_bound)


#* Convert mileage ranges like "120 000 - 129 999" to the mean value
def mileage_to_mean(x):
    try:
        parts = x.split('-')
        return (int(parts[0].replace(' ', '')) + int(parts[1].replace(' ', ''))) / 2
    except:
        return np.nan

df['mileage'] = df['mileage'].apply(mileage_to_mean)

#* Fill missing mileage values with the column mean
df['mileage'] = df['mileage'].fillna(df['mileage'].mean())

#* Convert 'model_year' to numeric and fill missing values with median
df['model_year'] = pd.to_numeric(df['model_year'], errors='coerce')
df['model_year'] = df['model_year'].fillna(df['model_year'].median())

#* Extract numeric value from 'tax_horsepower' (e.g., "6 CV" -> 6.0)
df['tax_horsepower'] = df['tax_horsepower'].str.extract(r"(\d+)").astype(float)
df['tax_horsepower'] = df['tax_horsepower'].fillna(df['tax_horsepower'].median())


#* Calculate car age from model year
df['car_age'] = 2025 - df['model_year']

#* Translate french values to english
df["transmission"] = df["transmission"].replace("Automatique","automatic").replace("Manuelle","manual")
df["fuel_type"] = df["fuel_type"].replace("Essence","Gasoline").replace("Electrique","Electric").replace("Hybride","Hybrid")
df["origin"] = df["origin"].replace("WW au Maroc","Morocco WW").replace("Dédouanée","Customs cleared").replace("Importée neuve","Newly imported").replace("Pas encore dédouanée","Not yet customs cleared")

#* export cleaned up dataset
if EXPORT:
    df.to_csv(EXPORT_FILE,index=False)