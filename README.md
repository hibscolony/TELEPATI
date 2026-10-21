# TELEPATI — Paddy Disease Detection

**YOLO26n FAST T4×2 • Supervised object detection • 11 kelas**

Notebook submission utama berasal dari **hasil eksekusi Kaggle yang diunggah pemilik repository**, bukan notebook template kosong.

## Submission notebook (terbaru)

**[TELEPATI_YOLO26n_FAST_T4x2.ipynb](./TELEPATI_YOLO26n_FAST_T4x2.ipynb)**

Notebook memuat:
- Pemetaan 11 kelas dari anotasi COCO ke YOLO dan audit sampel label;
- Tahap boot (20% train), fine-tuning cepat (65% train), dan full-data fine-tuning (100% train);
- Pelatihan supervised YOLO26n pada Kaggle GPU T4 ×2;
- Evaluasi validation mAP dan Macro F1, pemilihan confidence pada validation;
- Evaluasi test dengan confidence yang telah dipilih.

## Hasil evaluasi yang tercatat di output notebook

| Metrik | Validation | Test |
| --- | ---: | ---: |
| mAP@50 | 65.32% | **67.06%** |
| Macro F1 @ IoU 0.50 | 63.97% | **65.54%** |

Confidence optimal di validation: **0.25**. Macro F1 mengikuti implementasi matching berbasis IoU yang ada di notebook. Angka di atas disalin dari **output run** pada notebook yang diunggah, bukan hasil eksekusi ulang di luar Kaggle.

## Reproduksi

1. Siapkan dataset TELEPATI dengan anotasi COCO untuk train, valid, dan test.
2. Jalankan notebook di Kaggle dengan GPU T4 ×2, mengikuti konfigurasi dataset pada notebook.
3. Simpan hasil model akhir dari `/kaggle/working/telepati_abf/exports/fast_t4x2/final_best.pt`.
4. Gunakan validation untuk memilih threshold; test hanya untuk evaluasi final.
5. Simpan bobot model secara persisten (Kaggle Save Version/download); runtime Kaggle bisa dihapus.

**Perhatian:** pemetaan alias `paddy` dan `Rice-Leaf-Diseasee` ke `bacterial_leaf_blight` masih perlu diverifikasi terhadap label asli, sebagaimana dicatat notebook.

## Model weights — belum sinkron dengan notebook terbaru

[Release **YoloABF**](https://github.com/hibscolony/TELEPATI/releases/tag/YoloABF) masih menyimpan `best_yolo26n_abf_kd.pt` dari percobaan **ABF + Knowledge Distillation sebelumnya**, **bukan** bobot dari notebook FAST T4×2 yang baru diunggah.

Untuk melengkapi submission versi FAST, unggah `final_best.pt` hasil run Kaggle ke release baru, misalnya `FAST-T4x2-v1`, dan gunakan tautan release tersebut di formulir.

## Arsip eksperimen

Notebook [TELEPATI_YOLO26n_ABF.ipynb](./TELEPATI_YOLO26n_ABF.ipynb) dipertahankan sebagai riwayat metode lama.
