# Utiliser une image Python 3.11.6
FROM python:3.11.6-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier uniquement les fichiers nécessaires pour l'application
COPY app/ ./app/
COPY models/ ./models/
COPY requirements.txt ./
COPY photo.jpg ./

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port utilisé par Streamlit
EXPOSE 8501

# Commande pour lancer l'application
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]