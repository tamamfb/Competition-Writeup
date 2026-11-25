# Data ROGII Wellbore Geology Prediction

Dataset tidak disimpan di repo karena ukurannya sekitar 1,3 GB.

- Unduh: [Google Drive](GDRIVE_LINK_ROGII)
- Sumber asli: [Kaggle, ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data)

Setelah diunduh, susunannya seperti ini:

```
data/
├── sample_submission.csv                       # id,tvt (14.151 baris)
├── AI_wellbore_geology_prediction_task_en.pptx # deskripsi tugas dari penyelenggara
├── train/
│   ├── <well_id>__horizontal_well.csv   # MD,X,Y,Z,ANCC,ASTNU,ASTNL,EGFDU,EGFDL,BUDA,TVT,GR,TVT_input
│   ├── <well_id>__typewell.csv          # TVT,GR,Geology
│   └── <well_id>.png                    # pratinjau sumur dari penyelenggara
└── test/
    ├── <well_id>__horizontal_well.csv   # MD,X,Y,Z,GR,TVT_input
    └── <well_id>__typewell.csv          # TVT,GR
```

Train berisi 773 sumur. Folder test hanya berisi 3 sumur placeholder; sumur test sebenarnya disediakan Kaggle saat notebook dijalankan ulang.
