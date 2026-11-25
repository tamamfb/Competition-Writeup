# Data Datathon UI Task 1

Dataset tidak disimpan di repo karena ukurannya sekitar 189 MB.

- Unduh: [Google Drive](GDRIVE_LINK_DATATHON_T1)

Setelah diunduh, susunannya seperti ini:

```
data/
├── sample_submission.csv              # id,speed (2.041.200 baris)
├── static/
│   ├── Roads1260.json                 # daftar link per ruas jalan
│   ├── matrix.npy                     # adjacency 1260×1260 berarah
│   └── active_mask.npy                # mask grid 36×36
├── train/
│   ├── train_speed_m1_1_11160.npy     # 11160×1260 float32
│   ├── train_speed_m2_1_5039.npy      # 5039×1260 float32
│   ├── train_text_m1_1_11160.json     # teks kejadian per langkah waktu
│   └── train_text_m2_1_5039.json
└── test/
    ├── test_X_hist.npy                # 540×15×1260
    └── test_texts.json
```
