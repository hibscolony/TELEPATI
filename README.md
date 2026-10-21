# TELEPATI — Paddy Disease Detection

Deteksi 11 kelas kondisi/penyakit padi menggunakan **supervised object detection**, dengan evaluasi **mAP@50** dan **Macro F1 @ IoU 0.50**.

## Notebook

| Eksperimen | Notebook | Keterangan |
| --- | --- | --- |
| **YOLO26n FAST T4×2 (terbaru)** | [TELEPATI_YOLO26n_FAST_T4x2.ipynb](./TELEPATI_YOLO26n_FAST_T4x2.ipynb) | Kaggle T4 ×2; tahap boot 1 epoch, tahap 1 empat epoch, tahap 2 enam epoch; evaluasi validation/test; `gc` import sudah diperbaiki |
| YOLO26n ABF + Knowledge Distillation (arsip) | [TELEPATI_YOLO26n_ABF.ipynb](./TELEPATI_YOLO26n_ABF.ipynb) | Pendekatan lama; tidak sama dengan FAST T4×2 |

## Menjalankan FAST T4×2 di Kaggle

1. Tambahkan dataset TELEPATI yang memiliki anotasi COCO untuk **train**, **valid**, dan **test**.
2. Aktifkan accelerator **GPU T4 ×2**. Pastikan dependencies dan pretrained weights tersedia.
3. Jalankan notebook FAST berurutan. Tahap boot menghasilkan checkpoint awal, lalu tahap 1 dan tahap 2 melakukan fine-tuning YOLO26n.
4. Simpan checkpoint penting melalui **Save Version** Kaggle atau unduh, karena file di `/kaggle/working` tidak dijamin bertahan jika runtime hilang.
5. Ambil bobot akhir dari `/kaggle/working/telepati_abf/exports/fast_t4x2/final_best.pt`, setelah tahap 2 selesai.
6. Gunakan **validation** untuk pemilihan confidence threshold; lakukan evaluasi **test** sesudah model dan threshold ditetapkan.

> Pelatihan, mAP, dan Macro F1 pada notebook FAST belum diverifikasi di GitHub. Jangan mencantumkan skor tanpa output evaluasi yang sesuai. Tidak ada zero-shot pada pipeline ini.

## Bobot model di GitHub

[Release **YoloABF**](https://github.com/hibscolony/TELEPATI/releases/tag/YoloABF) berisi bobot **`best_yolo26n_abf_kd.pt` dari pendekatan ABF/KD lama**. Bobot tersebut **bukan** hasil training FAST T4×2.

**Bobot FAST T4×2 belum diunggah ke repo/release.** Setelah file `final_best.pt` tersedia dari Kaggle, buat release terpisah (misalnya `FAST-T4x2-v1`) agar bobot lama tidak tertukar.

## Catatan data

Pemetaan label lama menyatukan `paddy` dan `Rice-Leaf-Diseasee` ke `bacterial_leaf_blight`. Mapping ini perlu diperiksa secara manual terhadap anotasi asli, karena mungkin memengaruhi validitas hasil.

Repositori tidak berisi seluruh dataset karena ukurannya dan kemungkinan ketentuan lisensinya.
