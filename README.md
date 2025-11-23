📊 AI Finance Tracker
OCR-Powered Expense Extraction + Smart Dashboard + AI Budget Assistant + Voice Input

The AI Finance Tracker is a personal finance management web application that automatically extracts spending from SMS screenshots using OpenAI Vision, categorizes them, builds month-wise summaries, compares spending trends, and provides AI-driven budgeting advice through a built-in assistant (with optional voice input).

This project is ideal for college submissions, personal use, or real-world learning.

🚀 Features
🔍 1. Screenshot-Based OCR

Upload:

A single screenshot

OR a full folder of SMS/notification screenshots

The system uses OpenAI Vision to extract:

Date

Description

Amount

Transaction type (credit/debit)

🧠 2. Automatic Categorization

Rule-based classifier assigns transactions to categories:

Rent

Groceries

Food Delivery

Shopping

Transport

Other

📈 3. Interactive Dashboard

Includes:

Monthly spending overview

Category progress bars

Notifications when budgets are exceeded or nearing limits

Spending comparison: Current vs Last Month

Automatic suggested budgets for next month

Full transaction table

Toast alerts

💬 4. AI Budget Assistant

Chat with the assistant to ask:

"How much did I spend on groceries?"

"Am I overspending anywhere?"

"Give me goals for next month."

AI also auto-saves goals into the system.

🎤 5. Voice Input (Speech-to-Text)

Press the mic button to speak your query.
Includes:

Voice recording animation

Auto-insert into chat box

Works seamlessly with OpenAI assistant

🌓 6. Beautiful UI with Dynamic Gradient Background

Fully custom CSS

Expanding/collapsing assistant panel

Responsive and polished layout

🛠️ Tech Stack
Frontend

HTML / CSS

JavaScript (with Web Speech API for voice input)

Chart.js for interactive graphs

Backend

Python + Flask

Pandas for data processing

OpenAI (GPT + Vision models)

dotenv for API keys

CSV + JSON storage

📁 Project Structure
finance-tracker/
│── app.py
│── ocr.py
│── requirements.txt
│── Procfile
│── .env
│── data/
│   ├── transactions.csv
│   ├── goals.json
│── templates/
│   ├── index.html
│   ├── dashboard.html
│   ├── assistant.html
│   ├── base.html
│── static/
│   ├── style.css
│   ├── mic-listen.gif
│   ├── script.js (if added)
│── images/ (for OCR testing)

🧪 Sample SMS Format

OCR extracts values from messages like:

BANK ALERT: A purchase of ₹450.00 at DMART was made on your card ending 1234.
BANK ALERT: Your card ending 1234 was used for ₹320.00 at SWIGGY.
BANK ALERT: Rent payment of ₹12000.00 was debited today.

⚙️ Setup Instructions
1️⃣ Clone the repo
git clone https://github.com/Rose-Bhardwaj/finance-tracker.git
cd finance-tracker

2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac
venv\Scripts\activate      # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add your OpenAI API key

Create .env file:

OPENAI_API_KEY=your_key_here
FLASK_SECRET_KEY=some_random_secret

5️⃣ Run the app
python app.py

🌐 Deployment (Render)

Push project to GitHub

Create new Render Web Service

Set:

Start command: gunicorn app:app

Build: pip install -r requirements.txt

Add environment variables

Deploy!

🛡️ Security Notes

API keys are loaded securely via .env

No database needed — uses CSV/JSON for simplicity

Works offline after first load (except AI calls)

📌 Future Enhancements

Potential improvements:

Spending prediction based on ML

Multi-user support

Export full financial report to PDF

Mobile optimized version

Full voice assistant (TTS + STT)

🤝 Contributing

Pull requests and feature suggestions are always welcome.

📜 License

This project is for educational use. Free to modify and share.

❤️ Acknowledgements

Built using:

OpenAI GPT & Vision

Flask

Pandas

Chart.js
