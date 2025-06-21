import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch # Assurez-vous que torch est bien installé pour le fonctionnement avec transformers

# 1. Chargement des données
# Nous allons utiliser un petit sous-ensemble du jeu de données "imdb" pour la démonstration.
# Pour un projet réel, vous pourriez vouloir un jeu de données plus équilibré ou plus grand.
print("Chargement du jeu de données...")
# Charger un petit échantillon pour un entraînement rapide
dataset = load_dataset("imdb")
# Réduire la taille du dataset pour un entraînement plus rapide localement et sur GitHub Actions
# Pour un exemple réel, utilisez plus de données.
small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000)) # 1000 exemples
small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(200))   # 200 exemples
print("Jeu de données chargé.")

# 2. Chargement du tokenizer et du modèle
model_name = "distilbert-base-uncased"
print(f"Chargement du tokenizer et du modèle {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) # 2 labels pour positif/négatif
print("Tokenizer et modèle chargés.")

# 3. Fonction de prétraitement des données
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

print("Tokenisation des données...")
tokenized_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = small_eval_dataset.map(tokenize_function, batched=True)
print("Données tokenisées.")

# 4. Définition des métriques d'évaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="weighted")
    accuracy = accuracy_score(labels, predictions)
    return {"f1_score": f1, "accuracy": accuracy}

# 5. Configuration de l'entraînement
output_dir = "./models"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",  # Ajoutez cette ligne !
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        report_to="none"
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 6. Entraînement du modèle
print("Début de l'entraînement du modèle...")
trainer.train()
print("Entraînement terminé.")

# 7. Sauvegarde du modèle entraîné
final_model_path = os.path.join(output_dir, "sentiment_model")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Modèle sauvegardé dans : {final_model_path}")

# Évaluation finale sur le jeu d'évaluation
results = trainer.evaluate()
print(f"Résultats de l'évaluation finale : {results}")

# Sauvegarder les métriques dans un fichier pour que evaluate.py puisse les lire
# Ou pour que le workflow GitHub Actions puisse les récupérer directement
# Pour l'instant, on va juste imprimer, evaluate.py fera la lecture.