import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# Définir le chemin où le modèle a été sauvegardé par train.py
model_path = "./models/sentiment_model"

if not os.path.exists(model_path):
    print(f"Erreur : Le dossier du modèle '{model_path}' n'existe pas. Assurez-vous d'avoir entraîné le modèle avec train.py d'abord.")
    exit(1)

print("Chargement du tokenizer et du modèle entraîné...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
print("Modèle et tokenizer chargés.")

# Chargement du jeu de données de test (le même que celui utilisé pour l'évaluation dans train.py)
print("Chargement du jeu de données de test...")
dataset = load_dataset("imdb")
small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(200)) # Même taille que dans train.py
print("Jeu de données de test chargé.")

# Fonction de prétraitement des données
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

print("Tokenisation des données de test...")
tokenized_eval_dataset = small_eval_dataset.map(tokenize_function, batched=True)
print("Données de test tokenisées.")

# Définition des métriques d'évaluation (doit être la même que dans train.py)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="weighted")
    accuracy = accuracy_score(labels, predictions)
    return {"f1_score": f1, "accuracy": accuracy}

# Configuration du Trainer pour l'évaluation
# Pas besoin de TrainingArguments aussi complets pour l'évaluation seule
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    eval_dataset=tokenized_eval_dataset,
    compute_metrics=compute_metrics,
)

# Évaluation du modèle
print("Début de l'évaluation du modèle...")
metrics = trainer.evaluate()
print(f"Résultats de l'évaluation : {metrics}")

# Récupérer le F1-score pour le pipeline CI/CD
f1_score_value = metrics.get("eval_f1_score") # Notez le préfixe 'eval_'
accuracy_score_value = metrics.get("eval_accuracy")

if f1_score_value is not None:
    print(f"F1-score final pour le pipeline : {f1_score_value}")
    # Optionnel : sauvegarder le score dans un fichier JSON pour une récupération facile par GitHub Actions
    with open("evaluation_results.json", "w") as f:
        json.dump({"f1_score": f1_score_value, "accuracy": accuracy_score_value}, f)
    print("Résultats de l'évaluation sauvegardés dans evaluation_results.json")
else:
    print("Le F1-score n'a pas pu être récupéré.")
    exit(1) # Quitter avec un code d'erreur si le score n'est pas trouvé