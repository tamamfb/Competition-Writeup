# Competition Writeup

Kumpulan writeup lomba data science dan machine learning yang saya ikuti selama 2026, saat kuliah S1 di Institut Teknologi Sepuluh Nopember. Setiap writeup membahas masalahnya, temuan dari data, pendekatan yang dipakai, hasil beserta ablation, dan hal-hal yang tidak berhasil. Gambar diagram dan grafik dibuat dengan gaya paper.

## Daftar lomba

Diurutkan dari yang paling awal.

| Waktu | Lomba | Tugas | Pendekatan utama | Hasil |
|---|---|---|---|---|
| Apr 2026 | [DAC: Face Anti-Spoofing](DAC-Face-Anti-Spoofing) | Klasifikasi gambar 6 kelas (wajah asli vs 5 jenis serangan) | CLIP ViT-L/14 + agregasi lintas lapisan (I-FAS), pseudo-label | Macro-F1 validasi 0.9888 |
| Mei 2026 | [IMITCOMP 2026](IMITCOMP-2026) | Prediksi gagal bayar pinjaman P2P | XGBoost, SHAP, skor risiko | AUC-ROC test 0.9496 |
| Jul 2026 | [Datathon UI](DATATHON-UI) | Playground tabular, prakiraan lalu lintas, prediksi klik Wikispeedia | TabPFN; MLP spasial-temporal berbasis graf; rekonstruksi graf dari screenshot + LightGBM | Task 2 public LB 0.890 |
| Jul 2026 | [ROGII: Wellbore Geology](ROGII-Wellbore-Geology) | Prediksi TVT sepanjang sumur horizontal | Estimator struktural dari sumur tetangga + koreksi forward-backward | Public LB RMSE 12.995 |
| Sep 2026 | [Hology: HoloMine](Hology) | Klasifikasi mammogram; harga properti dari teks iklan | Ensemble model mammografi publik zero-shot; DeBERTa + stacking | Task 1 public LB 0.75 |
| Okt 2026 | [DATASCAPE 2026](Datascape-2026) | Klasifikasi kondisi paket e-commerce | 8 encoder beku, forensik duplikat, koreksi jalan pintas | **Juara 1** private LB (0.96760, 91 tim) |
| Okt 2026 | [DiceFest 2026](DiceFest-2026) | Regresi konsumsi energi, 5 target | Identitas eksak + 16 model GBDT/MLP, blend NNLS | **Peringkat 6** private LB, finalis |
| Okt 2026 | [Name That LLM](Name-That-LLM) | Menebak LLM penulis teks dari 18 model | LM beku + LDA, decoding per grup prompt | CV 0.7365, public LB 0.434 |
| Okt-Nov 2026 | [ACTION 2026](ACTION-2026) | Klasifikasi komentar layanan publik: relevansi lalu sentimen per pilar | De-obfuscation teks, IndoBERT + Sailor2-3B LoRA | **Finalis**; private LB final 0.76116 |
| Nov 2026 | [Data Slayer 4.0](Dataslayer-4.0) | Deteksi kecurangan ujian dari video | Stitching klip, DINOv2 + V-JEPA 2, assignment per subjek | Public LB 1.0 |

Lomba diikuti bersama tim **sistemhitam** (ACTION, DATASCAPE), **doremifasolasido** (Datathon UI, ROGII), **Hbat** (DiceFest), dan **nasi bungkus** (Hology).

## Struktur repo

Setiap folder lomba berisi berkas yang sama:

```
<Lomba>/
├── README.md        # writeup
├── notebook.ipynb   # notebook final
├── submission.csv   # submission final
└── data/            # data lomba, atau README berisi link unduh jika datanya besar
```

Lomba yang punya beberapa task atau babak (ACTION, Datathon UI, Hology) memakai subfolder per task dengan isi yang sama, sedangkan README writeup-nya tetap satu di tingkat lomba.

```
assets/
├── <lomba>/         # gambar yang dipakai di writeup
└── src/             # skrip matplotlib pembuat gambar
```

Semua gambar bisa dibuat ulang dengan `py -3.12 assets/src/fig_<lomba>.py`. Gaya diagram dan grafiknya diatur di [`assets/src/paperfig.py`](assets/src/paperfig.py).

## Catatan

- Dataset yang besar (gambar, video, atau di atas sekitar 50 MB) tidak disimpan di repo. Link unduhnya ada di `data/README.md` masing-masing lomba.
- Path data di notebook sudah diarahkan ke folder `data/`. Beberapa notebook juga membutuhkan cache atau artefak hasil run sebelumnya; kebutuhan ini dijelaskan di bagian "Menjalankan ulang" pada tiap writeup.
- Data lomba adalah milik penyelenggara masing-masing. Lisensi MIT di repo ini hanya berlaku untuk kode dan tulisan.
