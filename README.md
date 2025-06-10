# 🧠 MNIST Digit Recognizer with Streamlit, PyTorch, and PostgreSQL Logging

This project is a web-based image classifier built using **PyTorch**, **Streamlit**, and **PostgreSQL**. It allows users to draw digits (0–9) on an interactive canvas and get real-time predictions from a trained neural network. 
The app also logs each prediction into a PostgreSQL database for analysis and audit.

MNIST Digit Recognizer

---

## 📦 Features

- ✍️ Interactive canvas to draw handwritten digits
- 🔮 Real-time predictions using a trained PyTorch model
- 🧮 Confidence score displayed for each prediction
- 🗃️ Logging of predictions and actual labels to a PostgreSQL database
- 🧾 Display of recent predictions in a dynamic table
- 🐳 Fully containerized with **Docker** and orchestrated using **Docker Compose**
- 🌐 Ready to deploy to any server with Docker support

---

## 🚀 Live Demo

👉 [**Live App URL**](#) — *https://mnist-digit-recognizer-lk3kqsoygvrji4rnczq75k.streamlit.app/*

---

## 🧰 Technologies Used

- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [PostgreSQL](https://www.postgresql.org/)
- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [Psycopg2](https://www.psycopg.org/docs/) for PostgreSQL integration
- [Pil](https://pil.readthedocs.io/) for image processing

---

## 🧠 Model Overview

The PyTorch model is a simple fully connected neural network trained on the MNIST dataset: 

```python
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc(x)
      )
```

🖥️ Local Development
```LD
Clone the Repository:
git clone https://github.com/Bashaito1/mnist-digit-recognizer.git
cd mnist-digit-recognizer
```

Set Up Environment
```LD
Create a Python virtual environment (optional but recommended):
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run Streamlit App
```LD
streamlit run app.py
```

🐳 Run with Docker Compose
```LD
Build and Run
docker-compose up --build
```

This will:
```LD
Build the Streamlit app image
Start a PostgreSQL container
Serve your app on http://localhost:8505
```

Stop and Remove Containers
```LD
docker-compose down
```

🗄️ PostgreSQL Database
```LD
Container name: db

Default credentials (set via docker-compose.yml or .env):

DB_NAME=mnistdb
DB_USER=mnistuser
DB_PASSWORD=mnistpass
```
The app logs predictions to the predictions table:

```SQL
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    predicted_digit INTEGER,
    true_label INTEGER,
    timestamp TIMESTAMP
);

```
📂 Folder Structure
```SQL
mnist-digit-recognizer/
│
├── app.py                    # Main Streamlit application
├── mnist_model.pth           # Trained PyTorch model
├── Dockerfile                # Docker image definition for Streamlit app
├── docker-compose.yml        # Multi-container setup
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview
└── .env                      # Environment variables (optional)
```
```SQL
🛡️ Security Notes
Do not hardcode DB credentials in app.py. Use environment variables or .env.
Ensure your personal access token (PAT) is never committed to GitHub.
```
```SQL
✨ Future Improvements
Add user authentication
Train a more advanced CNN model for higher accuracy
Visual analytics dashboard for predictions
Deploy with HTTPS and custom domain
```
🧑‍💻 Author
Emeka Onwubalili



