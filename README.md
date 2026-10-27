# Student Service Request Monitoring System

Streamlit decision-support application based on the supplied midterm notebook and synthetic 330-record service request dataset.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

## Pages

- Home
- Dataset upload and validation
- Data Analysis (statistics and interactive charts)
- Prediction (saved Decision Tree pipeline)
- Dashboard (risk flags and review queue)
- System Design (requirements, actors, use case diagram and workflow)

## Model details

The fitted pipeline uses the same ten input features and Decision Tree settings in the provided notebook. The target `SLA_Breached` is defined for completed records with both `Processing_Days` and `SLA_Days` available. The model is saved to `models/trained_model.joblib`; model evaluation metrics are recorded in `models/model_metadata.json`.

## Important limitations

The dataset is synthetic practice data. Model metrics are for a held-out split of this dataset and do not guarantee real-world performance. Predictions are decision support only. Authorized staff must review source information and make final decisions. Do not publish personal or sensitive student records.
