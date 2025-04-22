
# 🧠 Predicting Credit Risk

A machine learning project to predict credit risk, helping financial institutions reduce loan defaults, fraud, and financial losses.

---


## ❓ Problem Statement

Financial institutions, payment providers, and fintech companies need to accurately assess the risk associated with funding consumer purchases and onboarding merchants. A failure to correctly estimate risk can lead to increased loan defaults, fraud, and financial losses.

---

## 📁 Project Structure

```
📦 root
 ┣ 📂 credits         # Core logic and model pipeline
 ┣ 📂 notebook        # Jupyter Notebooks for analysis
 ┣ 📂 static          # CSS and assets
 ┣ 📂 templates       # HTML frontend
 ┣ 📜 app.py          # Flask Web Application
 ┣ 📜 demo.py         # CLI Prediction Demo
 ┣ 📜 Dockerfile
 ┣ 📜 requirements.txt
 ┗ 📜 setup.py
```

---

## 🧪 Run Locally

### 🔧 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🚀 Launch the App

```bash
python app.py
```

---

## 🔐 Environment Variables (PowerShell)

```powershell
$env:MONGODB_URL = "your_mongodb_connection_string"
$env:AWS_ACCESS_KEY_ID = "<Your_AWS_ACCESS_KEY_ID>"
$env:AWS_SECRET_ACCESS_KEY = "<Your_AWS_SECRET_ACCESS_KEY>"
```

---

## ☁️ AWS CI/CD Deployment (GitHub Actions)

### 🛠 Setup

1. IAM User with:
   - AmazonEC2FullAccess
   - AmazonEC2ContainerRegistryFullAccess

2. Create ECR repo  
   URI: `438465169815.dkr.ecr.eu-west-3.amazonaws.com/credit4`

3. Launch EC2 (Ubuntu) & install Docker

---

### 📦 Workflow

```bash
# Build and push
docker tag credit-risk:latest 438465169815.dkr.ecr.eu-west-3.amazonaws.com/credit4
docker push 438465169815.dkr.ecr.eu-west-3.amazonaws.com/credit4

# On EC2
docker pull 438465169815.dkr.ecr.eu-west-3.amazonaws.com/credit4
docker run -d -p 80:5000 credit-risk
```

---

## 🤝 Contributing

Contributions are welcome!  
Fork → Improve → PR ✅

---

## 📄 License

**MIT License**
