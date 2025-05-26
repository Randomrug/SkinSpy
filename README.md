# 🕵️‍♀️ SkinSpy

"Not Every Spot Is Just a Dot — Let SkinSpy Decide"  
Catch the Cancer with CNN.

SkinSpy is a web application that uses a deep learning model to classify suspicious skin spots as benign, malignant, insect bite, or no concern. It aims to promote skin cancer awareness by helping users distinguish between harmless skin issues and potentially dangerous ones.

───────────────────────────────────────────────

💻 How to Run This Project

🧰 Prerequisites:
- Python 3.8 or higher
- VS Code (recommended)
- pip (Python package installer)
- trained models

───────────────────────────────────────────────

✅ Step-by-Step Setup

1. Clone the Repository:
   git clone https://github.com/Randomrug/SkinSpy.git
   cd SkinSpy

2. Open the Project in VS Code:
   → Open VS Code
   → File > Open Folder > Select the SkinSpy folder
   → Install Python extension if prompted

3. Create a Virtual Environment:
   python -m venv venv

4. Activate the Virtual Environment:
   On Windows:
     venv\Scripts\activate
   On macOS/Linux:
     source venv/bin/activate

5. Install the Required Libraries:
   pip install numpy tensorflow pillow torch torchvision flask werkzeug

6. Download the Trained Model Files:
   The trained models are too large to upload to GitHub.
   Download them from Google Drive:
   👉 https://drive.google.com/drive/folders/1vL74I7Xdcap_HLgQLyNuZGrj09H3-Iif?usp=drive_link 

7. Place the Model Files:
   After downloading, move them into a `models/` folder in the project root:
   SkinSpy/
   ├── app.py
   ├── static/
   ├── templates/
   ├── models/
       ├── skin_classification.h5
       └── insect_modelformal.pt

8. Run the App:
   python app.py

9. Open the App in Your Browser:
   Go to: http://127.0.0.1:5000

───────────────────────────────────────────────

🧠 What This App Does

- Uploads a suspicious skin image.
- Uses a trained CNN to classify the image.
- Shows confidence percentages for:
  - Benign
  - Malignant
  - Insect Bite - sub prediction into ( ts and abcf classes )
      - ts - ticks or spider bites
      - abcf - ants/bugs/chiggers/fleas 
  - No Bites
- Displays a dynamic result verdict with suggestions and links on the medical information regarding the predicted outcome.

───────────────────────────────────────────────

📝 Notes

⚠️ The trained model files are hosted externally due to size constraints.
📥 Download from Google Drive and place in the /models directory.

🧪 This project is for **awareness and educational use** only. It is **not a diagnostic tool**. Always consult a real doctor.

───────────────────────────────────────────────

🙌 Credits to myself(😂😄)

Made with 💜 by Randomrug  
GitHub: https://github.com/Randomrug

───────────────────────────────────────────────

📄 License

MIT License
