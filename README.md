# 🌿 Agri-Chat: AI for Optimal Crop Production (Kisaan Mitra AI)

## 🌟 Project Title and Description

[cite_start]Agri-Chat AI ek low-latency, Dual-API-powered chatbot solution hai jo agricultural users ke liye design kiya gaya hai[cite: 1, 2]. [cite_start]Iska core function **soil-based crop production** aur **nutritional deficiency (N, P, K)** ki turant pehchaan karna hai[cite: 2, 8, 13]. [cite_start]Yeh farmers ko personalized fasal ki salaah (recommendations) aur khaad (fertilizer) use karne ki guidance deta hai[cite: 14, 25].

## ❓ Problem Statement

Aaj bhi bahut se kisaan, khaaskar rural areas mein, sahi aur timely agricultural salah tak nahi pahunch paate, jiske kai karan hain:

* Limited Access to Expert Guidance:** Many farmers lack timely expert guidance on soil health and optimal crop choices, leading to poor yields.
* Complexity of Management:** Understanding vital soil nutrient levels (Nitrogen, Phosphorus, Potassium - N, P, K) requires technical knowledge often unavailable to typical farmers.
* Inefficient Traditional Services:** Existing advisory systems are often slow, expensive, or inaccessible, particularly for small-scale farmers without smart sensor technology.
* Need for Accessible AI Solution:** There is a strong need for an affordable solution that can quickly help farmers understand their soil problems and give them good advice.

## ✅ Solution Approach

1.  Instant Visual Diagnosis (Core Function):** Leaf images (Visual Diagnosis) aur text input ke madhyam se **N, P, K deficiencies** aur fasal ki samasyao ki turant pehchaan karna[cite: 23, 13].
2.  Dual-API Optimization:Fast aur reliable response ke liye **OpenAI (Fastest chat)** aur **Gemini (Vision/Fallback)** ka use karte hue **API Racing** strategy apply karna.
3.  Natural Language Interaction:** LLM ka use karke kisaanon ke sawalon ka jawab aasan, natural language mein dena.
4.  Bridging the Gap: Dependency on expensive soil testing aur sensors kam karna, aur interactive, image-assisted advice dena.

## 🛠️ Technology Stack

|Component | Technology | Role |

| **Backend Framework** | Python, **FastAPI** | High performance, asynchronous API handling. |
| **AI Models** | **Gemini 2.5 Flash** (Vision/Fallback), **OpenAI GPT-3.5 Turbo** (Low-Latency Chat) | Fastest processing of images and text generation. |
| **Database** | SQLite (via **Aiosqlite/Databases** & **SQLAlchemy**) | User authentication and prediction history storage. |
| **Asynchronicity** | **Asyncio** (using `asyncio.to_thread`) | Concurrent processing of API calls and CPU-heavy tasks. |
| **Frontend** | HTML5, Tailwind CSS, Vanilla JavaScript | Responsive, advanced gradient UI, non-blocking fetch calls. |

## ⚙️ Setup and Installation Instructions
### Prerequisites
* Python 3.10+
* Git

## 📁 Download Link

[Click here to access the file on Google Drive]
https://drive.google.com/file/d/1VadJ2_8-Gk-Vo0qWfnPGcK0hhiFGkuzw/view?usp=sharing)

[PPT link]
https://drive.google.com/file/d/1srMFmjsUZLh5VyRHl50bG9Nrzp9PekKb/view?usp=sharing


### 1. Dependencies Confirm Aur Install Karein
```bash
# Ensure you have the requirements.txt file with all dependencies
pip install -r requirements.txt
after installing all the dependencies final run in terminal-> uvicorn app:app --reload


