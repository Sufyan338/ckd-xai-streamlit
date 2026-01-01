ckd-xai-streamlit/
├─ app.py
├─ requirements.txt
├─ README.md
├─ models/
│  ├─ binary_pipeline.joblib          (optional pretrained)
│  ├─ binary_label_encoder.joblib     (optional pretrained)
│  ├─ stage_pipeline.joblib           (optional pretrained)
│  └─ stage_label_encoder.joblib      (optional pretrained)
└─ src/
   ├─ __init__.py
   ├─ config.py
   ├─ data_io.py
   ├─ preprocess.py
   ├─ models.py
   ├─ evaluate.py
   ├─ xai.py
   └─ persistence.py
