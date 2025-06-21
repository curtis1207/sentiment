import os
import json
from huggingface_hub import HfApi, HfFolder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Chemin où le modèle entraîné est sauvegardé
model_path = "./models/sentiment_model"
# Fichier où les résultats d'évaluation sont stockés
evaluation_results_file = "evaluation_results.json"

# Récupérer la clé API Hugging Face à partir des variables d'environnement
# Cette variable sera définie via GitHub Secrets dans le workflow CI/CD
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    print("Erreur : La variable d'environnement 'HF_API_KEY' n'est pas définie.")
    print("Veuillez la configurer comme un GitHub Secret ou localement pour le test.")
    exit(1)

# Récupérer le seuil de performance à partir des variables d'environnement
# Cette variable sera définie via GitHub Secrets
THRESHOLD_SCORE_STR = os.getenv("THRESHOLD_SCORE")
if not THRESHOLD_SCORE_STR:
    print("Erreur : La variable d'environnement 'THRESHOLD_SCORE' n'est pas définie.")
    print("Veuillez la configurer comme un GitHub Secret ou localement pour le test.")
    exit(1)

try:
    THRESHOLD_SCORE = float(THRESHOLD_SCORE_STR)
except ValueError:
    print(f"Erreur : La valeur de THRESHOLD_SCORE '{THRESHOLD_SCORE_STR}' n'est pas un nombre valide.")
    exit(1)

print(f"Seuil de performance défini : {THRESHOLD_SCORE}")

# 1. Lire les résultats d'évaluation
if not os.path.exists(evaluation_results_file):
    print(f"Erreur : Le fichier de résultats d'évaluation '{evaluation_results_file}' est introuvable.")
    print("Veuillez vous assurer que 'evaluate.py' a été exécuté avec succès.")
    exit(1)

with open(evaluation_results_file, "r") as f:
    evaluation_results = json.load(f)

f1_score_model = evaluation_results.get("f1_score")

if f1_score_model is None:
    print("Erreur : Le F1-score n'a pas pu être récupéré des résultats d'évaluation.")
    exit(1)

print(f"F1-score du modèle obtenu : {f1_score_model}")

# 2. Comparer le score avec le seuil
if f1_score_model >= THRESHOLD_SCORE:
    print(f"Le F1-score ({f1_score_model:.4f}) atteint ou dépasse le seuil ({THRESHOLD_SCORE:.4f}). Déploiement en cours...")

    # Assurer que le modèle est disponible
    if not os.path.exists(model_path):
        print(f"Erreur : Le dossier du modèle '{model_path}' n'existe pas. Impossible de déployer.")
        exit(1)

    # 3. Se connecter à Hugging Face Hub et pousser le modèle
    try:
        # Sauvegarder la clé API pour cette session
        HfFolder.save_token(HF_API_KEY)
        api = HfApi()

        # Votre nom d'utilisateur Hugging Face
        # Vous pouvez le récupérer dynamiquement ou le définir.
        # Pour l'intégration CI/CD, utiliser le nom d'utilisateur associé à la clé API est souvent suffisant.
        # Sinon, vous pouvez le passer comme une autre variable d'environnement.
        # Par simplicité, on utilisera le nom du repo pour le modèle sur HF Hub
        repo_id =  f"curtis1207/sentiment-distilbert" # Si votre nom d'utilisateur est curtis" # REMPLACER votre_nom_utilisateur_HF

        # Vous pouvez créer un repo directement ou le laisser la fonction push_to_hub le faire
        # api.create_repo(repo_id=repo_id, private=False, exist_ok=True)

        print(f"Chargement du modèle et du tokenizer pour le push...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        print(f"Push du modèle vers Hugging Face Hub : {repo_id}...")
        model.push_to_hub(repo_id=repo_id)
        tokenizer.push_to_hub(repo_id=repo_id)
        print("Déploiement réussi sur Hugging Face Hub !")

        # Optionnel: Mettre à jour un message de succès
        with open("deployment_status.json", "w") as f:
            json.dump({"status": "success", "f1_score": f1_score_model, "repo_id": repo_id}, f)

    except Exception as e:
        print(f"Échec du déploiement sur Hugging Face Hub : {e}")
        with open("deployment_status.json", "w") as f:
            json.dump({"status": "failure", "f1_score": f1_score_model, "error": str(e)}, f)
        exit(1) # Quitter avec un code d'erreur

else:
    print(f"Le F1-score ({f1_score_model:.4f}) est inférieur au seuil ({THRESHOLD_SCORE:.4f}). Déploiement annulé.")
    with open("deployment_status.json", "w") as f:
        json.dump({"status": "rejected", "f1_score": f1_score_model}, f)
    exit(0) # Quitter avec un code de succès, car l'action était de ne pas déployer