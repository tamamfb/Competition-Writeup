# Data Data Slayer 4.0

Dataset tidak disimpan di repo karena ukurannya sekitar 6,4 GB dan berisi rekaman wajah peserta.

- Unduh: [Google Drive](GDRIVE_LINK_DATASLAYER)

Setelah diunduh, susunannya seperti ini:

```
data/
├── sample_submission.csv          # id,label (1.515 baris)
├── train/
│   ├── normal/subject_N/{front,side}_{Writing_Answers,Thinking,Looking_Around,Checking_Smartwatch}.MOV
│   └── cheat/subject_N/{front,side}_{Looking_Under_the_Desk,Reading_Hidden_Notes,Repeated_Peeking,Using_Smartwatch}.MOV
└── test/
    └── test_seq_XXXXX/01.jpg … 30.jpg   # 1.515 folder, 640×360
```
