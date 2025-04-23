![img_car](<img/car_img.png>)

##  📌 Index
-  [About the Project](#-about-the-project)  
-  [Main Features](#-main-features)  
-  [Current Issues](#-current-issues)
-  [Folder Structure](#-folder-structure)
-  [Possible Improvements](#-possible-improvements)   
-  [Technologies Used & Dependencies](#-technologies-used--dependencies)   
-  [Installation and Usage](#-installation-and-usage)   
-  [Collaborators](#-collaborators)   

---

##  🚗 About the Project  

<div align="justify">

**Car Prices Project** is an educational regression project that predicts the price of used cars based on various user-selected features. The data comes from [this Kaggle competition](https://www.kaggle.com/competitions/playground-series-s4e9/overview), focused on machine learning practice.

The main goal was to apply exploratory data analysis (EDA) techniques and build a prediction model that can be consulted through an interactive interface built with Streamlit.

</div>

---

##  🔍 Main Features  
✅ Complete EDA with visualizations to understand variable relationships.  
✅ Trained regression model to predict used car prices.  
✅ Streamlit visual interface for predictions.  
✅ Well-structured project by functionality.  

---

##  🐞 Current Issues  
❌ The dataset could be enriched with external sources.  

---

##  💡 Possible Improvements  
✅ Add and compare new models (XGBoost, CatBoost, etc.).  
✅ Implement more robust cross-validation.  
✅ Improve the UI and navigation in the Streamlit app.  

---

##  📁 Folder Structure

```bash
# Car_prices_project_equipo2
📂 Car-Prices-Project/  
├── 📂 .venv/                   
├── 📂 app/                   
│   └── app.py              
├── 📂 data/   
│   └── clean_data_car.csv 
│   └── train.csv                
├── 📂 eda/                   
│   └── eda_car_prices.ipynb  
├── 📜 .gitignore  
├── 📜 requirements.txt  
├── 📜 README.md  

🧪 Technologies Used & Dependencies



---

## ⚙️ Installation and Usage

1️⃣ Clone the repository
```bash
git clone [https://github.com/your-username/Car-Prices-Project.git](https://github.com/your-username/Car-Prices-Project.git)
cd Car-Prices-Project

2️⃣ Create and activate the virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Linux/MacOS
.venv\Scripts\activate     # On Windows


3️⃣ Install dependenci```bash

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the code
```bash
jupyter notebook eda/eda.ipynb
```

### 5️⃣ Start the Streamlit 

```bash
streamlit run app/app.py
```


### 🧑‍💻 Collaborators
This project was developed by the following contributors:  
- [Yael Parra](https://www.linkedin.com/in/yael-parra/)  
- [Orlando Alcalá](https://www.linkedin.com/in/orlando-david-71417411b/)   
- [Noheli Salazar](https://www.linkedin.com/in/nhoeli-salazar/)   
- [Fernando García](https://www.linkedin.com/in/fernandogarciacatalan/)  

If you have suggestions or feedback, feel free to contact us!

