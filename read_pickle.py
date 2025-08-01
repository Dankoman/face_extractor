import pickle

# Ange sökvägen till din .pkl-fil
pkl_file = "face_knn_arcface.pkl"

# Läs in filen
with open(pkl_file, "rb") as f:
    data = pickle.load(f)

# Hämta LabelEncoder
label_encoder = data.get("label_encoder")

# Visa labels och räkna hur många som finns
if label_encoder:
    labels = label_encoder.classes_  # Klassen innehåller alla unika etiketter
    print("Labels i modellen:")
    print(labels)
    print(f"Antal unika labels: {len(labels)}")
else:
    print("Ingen LabelEncoder hittades i filen.")