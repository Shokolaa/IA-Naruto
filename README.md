# **Reconnaissance de mudra naruto avec Pytorch et MediaPipe**

Système de reconnaissance en temps réel des mudras (signes ninja) de Naruto via webcam, combinant le tracking des articulations de la main par MediaPipe Hands et une classification ultra-rapide par un réseau de neurones profond sous PyTorch accéléré par CUDA.

## **Architecture du Modèle**

Entrée (Vecteur de taille 126\)  
   │  
   ├── Linear(126 \-\> 256\) ── BatchNorm1d ── ReLU ── Dropout(0.3)  
   │  
   ├── Linear(256 \-\> 128\) ── BatchNorm1d ── ReLU ── Dropout(0.3)  
   │  
   ├── Linear(128 \-\> 64\)  ── BatchNorm1d ── ReLU  
   │  
   └── Linear(64 \-\> N\_CLASSES) ── CrossEntropyLoss / Softmax

## **1\. Prérequis & Installation**

> * **Python 3.10 ou 3.11** (MediaPipe a du mal avec Python 3.12)  
> * **Carte graphique Nvidia** avec support CUDA (testé avec CUDA 12.1+)  
> * **Une webcam fonctionnelle**

## **2\. Configuration de l'environnement virtuel**

\# Cloner le dépôt  
git clone 
cd "IA Naruto"

\# Créer l'environnement virtuel avec Python 3.11  
py \-3.11 \-m venv .venv

\# Activer l'environnement  
.\\.venv\\Scripts\\Activate.ps1

## **3\. Installation des dépendances**

\# Mise à jour de pip  
python \-m pip install \--upgrade pip

\# Installation de PyTorch avec support CUDA  
python \-m pip install torch torchvision \--index-url https://download.pytorch.org/whl/cu124

\# Dépendances de vision par ordinateur & analyse  
python \-m pip install "mediapipe\<0.10.15" opencv-python numpy pandas scikit-learn

## **Guide d'Utilisation**

### **Étape 1 : Collecter vos propres données (collect\_data.py)**

Lancez le script de capture :  
python collect\_data.py

**Raccourcis clavier :**

> * **ESPACE** : Activer / désactiver l'enregistrement continu.  
> * **TAB** : Basculer vers le signe suivant (Tigre, Serpent, Dragon, Oiseau).  
> * **ECHAP** : Sauvegarder et fermer l'application.

**Astuce de collecte :** Visez entre 200 et 300 poses par signe en inclinant et en déplaçant légèrement vos mains pour maximiser la robustesse.

### **Étape 2 : Entraîner le réseau (train.py)**

Une fois le fichier naruto\_dataset.csv généré, lancez l'entraînement :  
\# Entraînement standard (40 epochs par défaut)  
python train.py

\# Ou avec des hyperparamètres personnalisés  
python train.py \--epochs 50 \--batch\_size 32 \--lr 0.001

### **Étape 3 : Détection en direct (detect.py)**

Lancez l'inférence temps réel sur votre webcam :  
python detect.py

## **Performances Obtenues**

> * **Validation Accuracy** : \~98.9%  
> * **Validation Loss** : \~0.04  
> * **Temps d'inférence** : \< 15 ms par image sur GPU Nvidia.

## **Licence**

Ce projet est sous licence MIT.
