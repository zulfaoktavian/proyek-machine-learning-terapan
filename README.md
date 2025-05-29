# Laporan Proyek Machine Learning - Zulfaa Dwi Oktavian
## Domain Proyek
Darah merupakan jaringan ikat yang mengandung sel-sel khusus yang mengapung dalam cairan yang disebut plasma. Sel-sel darah tersebut terdiri atas tiga jenis utama, yaitu eritrosit (sel darah merah), leukosit (sel darah putih), dan trombosit (platelet). (Ananta Dwi Prayoga Alwy et al., 2023).  Eritrosit bertanggung jawab atas transportasi oksigen, sementara leukosit berperan dalam  pembuatan antibodi untuk  menangkal virus dan  bakteri. Selain itu, darah juga  membawa  bahan kimia  yang  diproduksi  melalui  metabolisme  dan  mengalirkan  zat  serta  oksigen  yang  diperlukan  tubuh. Karena peranannya yang sangat vital, darah memegang peranan penting dalam identifikasi penyakit dalam tubuh,  dimana  banyak  kondisi  penyakit  berkaitan  langsung  dengan  keseimbangan  darah (Boly & Akbar, 2024). Sel darah putih merupakan komponen penting dalam sistem peredaran darah yang berfungsi melindungi tubuh dari berbagai infeksi dan penyakit. Berdasarkan jenisnya, sel darah putih terbagi menjadi lima tipe utama, yaitu neutrofil, limfosit, monosit, eosinofil, dan basofil. Setiap jenis memiliki karakteristik yang berbeda, baik dari segi warna maupun bentuk. Neutrofil memiliki ciri khas berwarna merah kebiruan dengan tiga inti sel yang bentuknya bervariasi. Limfosit tampak berwarna biru pucat dan memiliki bentuk yang kaku karena tidak bergerak bebas. Monosit memiliki warna biru dan bentuk yang memanjang. Eosinofil ditandai dengan warna merah dan tampilan berbintik, sedangkan basofil memiliki warna biru dengan bentuk bintik serupa. Perbedaan morfologi ini menjadi dasar dalam identifikasi jenis sel darah putih melalui pengamatan mikroskopik (Prasthio et al., 2022). Sementara itu, eritrosit berperan dalam mengangkut oksigen ke seluruh tubuh melalui kandungan hemoglobin, sedangkan trombosit berfungsi dalam proses hemostasis dengan membantu pembekuan darah saat terjadi luka atau cedera pada pembuluh darah.

Untuk mengetahui jenis sel darah, pemeriksaan masih banyak dilakukan secara manual melalui serangkaian uji laboratorium, yang umumnya memerlukan waktu relatif lama. Proses ini melibatkan penggunaan preparat darah yang telah diberi pewarnaan khusus, sehingga perbedaan antara sel darah merah dan sel darah putih dapat diamati lebih jelas di bawah mikroskop. Namun, metode manual ini memiliki keterbatasan dalam hal akurasi karena sangat bergantung pada kondisi fisik, tingkat pengetahuan, ketelitian, dan konsentrasi petugas medis. Faktor-faktor tersebut dapat mempengaruhi hasil diagnosis, terutama ketika terjadi perbedaan interpretasi antara satu tenaga medis dengan yang lain. Selain itu, kompleksitas dalam mendeteksi jumlah dan jenis kelainan darah juga turut mempengaruhi ketepatan hasil analisis (Ardina, R., & Rosalinda, S., 2018). Kemajuan teknologi yang berkembang pesat telah mendorong transformasi signifikan di berbagai sektor, termasuk di bidang medis (Febrianti, 2017). Dalam praktik medis, analisis darah merupakan salah satu prosedur diagnostik yang esensial untuk mengidentifikasi berbagai jenis penyakit. Prosedur ini umumnya dilakukan dengan cara pengamatan langsung terhadap morfologi, jumlah, serta tipe sel darah pasien menggunakan mikroskop oleh tenaga ahli, seperti dokter atau analis laboratorium. Informasi yang diperoleh dari hasil pengamatan tersebut sangat krusial dalam menentukan kondisi kesehatan pasien secara menyeluruh. Namun, pendekatan konvensional ini memiliki keterbatasan dari segi efisiensi, akurasi, serta potensi subjektivitas dalam interpretasi hasil.

Sebagai solusi atas permasalahan tersebut, penerapan teknologi _deep learning_, khususnya melalui pendekatan _Transfer Learning_, dapat menjadi alternatif yang efektif. Transfer Learning memungkinkan pemanfaatan model _deep learning_ yang telah dilatih sebelumnya (pretrained models) pada dataset besar, kemudian disesuaikan (fine-tuned) untuk tugas klasifikasi sel darah. Beberapa arsitektur model pretrained yang dapat digunakan antara lain MobileNet, ResNet50, VGG16, dan EfficientNet. Model-model ini telah terbukti unggul dalam ekstraksi fitur dari citra dan memiliki performa tinggi dalam klasifikasi gambar. Dengan menggunakan pendekatan ini, sistem klasifikasi sel darah dapat dikembangkan dengan lebih cepat, akurat, serta efisien, bahkan dengan jumlah data pelatihan yang terbatas, sehingga berpotensi besar dalam mendukung proses diagnosis medis secara otomatis dan konsisten.

## Bussiness Understanding 
Berdasarkan latar belakang yang disampaikan, dapat didefinisikan hal yang ingin diselesaikan pada analisis ini sebagai berikut :
### Problem Statements
- Pernyataan Masalah 1
Proses identifikasi jenis sel darah masih banyak dilakukan secara manual melalui pengamatan mikroskopik oleh tenaga medis. Metode ini bersifat subjektif dan sangat bergantung pada tingkat keahlian, konsentrasi, dan kondisi fisik pemeriksa, yang dapat menyebabkan variasi hasil diagnosis antar individu.
- Pernyataan Masalah 2
Pemeriksaan manual terhadap preparat darah yang telah diwarnai memerlukan waktu yang cukup lama dan berpotensi menurunkan efisiensi layanan laboratorium, terutama pada rumah sakit atau fasilitas kesehatan dengan keterbatasan sumber daya manusia dan alat diagnostik.
- Pernyataan Masalah 3
Terdapat kompleksitas dalam membedakan jenis-jenis sel darah secara akurat karena variasi morfologi yang dimiliki oleh setiap tipe sel, seperti warna, bentuk inti, dan tekstur. Hal ini menyulitkan klasifikasi visual secara konsisten, terutama pada kasus jumlah data besar.

### Goals
- Tujuan 1
Mengembangkan sistem klasifikasi otomatis berbasis citra mikroskopik yang mampu mengidentifikasi jenis sel darah secara akurat dan konsisten menggunakan pendekatan deep learning.
- Tujuan 2
Mengurangi ketergantungan pada pemeriksaan manual yang memakan waktu dan rentan terhadap kesalahan manusia dengan memanfaatkan teknologi kecerdasan buatan, khususnya Transfer Learning.
- Tujuan 3
Mengevaluasi performa beberapa arsitektur model pretrained seperti MobileNet, ResNet50, VGG16, dan EfficientNet dalam melakukan klasifikasi sel darah putih berdasarkan citra, guna menemukan model yang paling efisien dan akurat untuk diterapkan dalam sistem pendukung keputusan klinis.

### Solution Statements 
- Solution Statement 1
Mengimplementasikan model Transfer Learning dengan menggunakan dua arsitektur pretrained yang berbeda, yaitu MobileNetV2 dan ResNet50, untuk melakukan klasifikasi jenis sel darah berdasarkan citra mikroskopik. Kedua model akan dibandingkan performanya berdasarkan metrik evaluasi seperti akurasi, presisi, recall, dan F1-score. Tujuannya adalah untuk menentukan model mana yang paling optimal dalam hal akurasi dan efisiensi komputasi.
- Solution Statement 2
Melakukan tuning hyperparameter pada model terbaik yang diperoleh dari Solution Statement 1, dengan memodifikasi parameter seperti learning rate, jumlah epoch, batch size, dan teknik augmentasi data. Eksperimen ini bertujuan untuk meningkatkan performa model dalam mengenali variasi visual dari citra sel darah. Efektivitas tuning diukur menggunakan akurasi validasi dan confusion matrix pada data uji.
- Solution Statement 3 (Opsional untuk penguatan)
Menambahkan data augmentation dan early stopping sebagai bagian dari strategi pelatihan untuk menghindari overfitting dan meningkatkan generalisasi model terhadap data baru. Teknik augmentasi seperti rotasi, flipping, dan zoom digunakan untuk mensimulasikan variasi pengambilan gambar di dunia nyata. Efektivitas pendekatan ini akan diukur dari perbandingan metrik evaluasi sebelum dan sesudah augmentasi diterapkan.

## Data Understanding

Dataset yang dimanfaatkan adalah sebuah koleksi citra sel darah yang terdiri dari delapan kelas berbeda. Fitur utama dalam dataset ini adalah 'image' dengan tipe data gambar, dan 'label' yang merupakan klasifikasi dari gambar tersebut. Label ini dikategorikan ke dalam delapan kelas yaitu: monocyte, ig (immature granulocyte), neutrophil, basophil, lymphocyte, erythroblast, eosinophil, dan platelet.
Dataset ini terbagi menjadi satu bagian utama yaitu 'train', yang memiliki total 17.092 contoh gambar. Ukuran unduhan dataset ini adalah sekitar 281.24 MB, sedangkan ukuran dataset setelah diekstrak adalah kurang lebih 312.69 MB. Informasi ini mengindikasikan bahwa dataset ini cukup substansial untuk melatih model klasifikasi citra sel darah.
Dataset ini terdapat di platform Hugging Face di tautan berikut: https://huggingface.co/datasets/Docty/Blood-Cells. Dataset ini memiliki lisensi MIT, yang memperbolehkan penggunaan secara luas.

Variabel-variabel pada Blood Cells
- monocyte:0
- ig (immature granulocyte):1
- neutrophil:2
- basophil:3
- lymphocyte:4
- erythroblast:5
- eosinophil:6
- platelet:7

Untuk dataset citra sel darah seperti yang Anda miliki ("Docty/Blood-Cells"), berikut adalah beberapa teknik visualisasi dan Exploratory Data Analysis (EDA) yang bagus untuk memahami data lebih dalam sebelum melatih model:

1. Distribusi Jumlah Sampel per Kelas (Class Distribution):
Visualisasi: Diagram batang (bar chart) yang menunjukkan jumlah gambar untuk setiap kelas sel darah (monocyte, ig, neutrophil, basophil, lymphocyte, erythroblast, eosinophil, platelet).
Tujuan: Ini adalah langkah EDA yang sangat penting untuk mengidentifikasi apakah ada ketidakseimbangan kelas (class imbalance). Jika beberapa kelas memiliki jumlah sampel yang jauh lebih sedikit daripada yang lain, model mungkin cenderung lebih baik dalam memprediksi kelas mayoritas dan buruk pada kelas minoritas. Mengetahui ini di awal membantu dalam merencanakan strategi penanganan seperti oversampling, undersampling, atau penggunaan metrik evaluasi yang tepat.

2. Menampilkan Contoh Gambar dari Setiap Kelas (Sample Image Display):
Visualisasi: Menampilkan beberapa gambar acak (misalnya, 3-5 gambar) untuk masing-masing dari delapan kelas sel darah.
Tujuan: Memberikan pemahaman visual tentang bagaimana setiap jenis sel darah terlihat. Ini membantu memverifikasi apakah label sesuai dengan gambar, mengidentifikasi potensi masalah kualitas gambar (misalnya, buram, artefak), dan mendapatkan intuisi tentang fitur visual yang membedakan antar kelas.

## Data Preparation

Tahapan Persiapan Data untuk Klasifikasi Sel Darah
Persiapan data adalah langkah krusial dalam membangun model machine learning yang akurat dan andal. Tujuannya adalah untuk membersihkan, mentransformasi, dan mengatur data mentah agar siap digunakan untuk melatih model. Untuk dataset citra sel darah, berikut adalah tahapan yang umumnya dilakukan:

1. Pemuatan Data (Data Loading)
- Proses: Memuat dataset citra sel darah (misalnya, dari direktori lokal atau platform seperti Hugging Face) ke dalam lingkungan kerja Anda (misalnya, notebook Python). Ini melibatkan pembacaan file gambar dan labelnya.
- Alasan: Langkah awal yang fundamental. Tanpa memuat data, kita tidak bisa melakukan analisis atau pemrosesan lebih lanjut.
2. Eksplorasi Data Awal (Initial Data Exploration)
- Proses: Seperti yang telah dibahas sebelumnya, ini mencakup:
   - Memeriksa jumlah gambar per kelas (distribusi kelas).
   - Menampilkan beberapa contoh gambar dari setiap kelas.
   - Memeriksa dimensi gambar dan tipe data.
- Alasan: Memahami karakteristik dasar dataset, mengidentifikasi potensi masalah seperti ketidakseimbangan kelas, kualitas gambar, atau kesalahan pelabelan awal. Informasi ini akan memandu tahapan persiapan data selanjutnya.
3. Pembersihan Data (Data Cleaning)
- Proses:
Penanganan Data Hilang (Missing Data): Memeriksa apakah ada gambar atau label yang hilang. Jika ada, tentukan strategi penanganannya (misalnya, menghapus sampel tersebut jika jumlahnya sedikit, atau melakukan imputasi jika memungkinkan, meskipun untuk gambar ini jarang terjadi).
Penanganan Duplikat: Mengidentifikasi dan menghapus gambar yang identik jika ada.
Koreksi Label (jika teridentifikasi): Jika selama eksplorasi ditemukan gambar yang salah label, idealnya label tersebut diperbaiki. Jika tidak memungkinkan, gambar tersebut bisa disisihkan.
- Alasan: Data yang bersih dan konsisten adalah fondasi untuk model yang baik. Data yang hilang, duplikat, atau salah label dapat mengganggu proses pelatihan dan menghasilkan model yang tidak akurat.
4. Pra-pemrosesan Gambar (Image Preprocessing)
Ini adalah salah satu tahapan terpenting untuk data citra.
- Proses:
Pengubahan Ukuran (Resizing): Menyeragamkan ukuran semua gambar ke dimensi yang sama (misalnya, 128x128 piksel, 224x224 piksel). Sebagian besar arsitektur model deep learning (seperti CNN) memerlukan input dengan ukuran yang konsisten.
Normalisasi Piksel (Pixel Normalization): Mengubah rentang nilai piksel gambar. Cara umum adalah mengubah skala nilai piksel dari rentang [0, 255] menjadi [0, 1] (dengan membagi semua nilai piksel dengan 255.0) atau [-1, 1].
Standardisasi (Standardization): Mengubah distribusi nilai piksel sehingga memiliki rata-rata (mean) 0 dan standar deviasi 1.
(Opsional) Konversi Mode Warna: Memastikan semua gambar memiliki mode warna yang sama (misalnya, RGB). Dataset sel darah biasanya sudah dalam format RGB.
- Alasan:
Resizing: Model memerlukan input dengan dimensi yang seragam.
Normalisasi/Standardisasi: Membantu proses pelatihan model agar lebih stabil dan konvergen lebih cepat. Nilai piksel yang besar dapat memperlambat pelatihan atau menyebabkan ketidakstabilan numerik. Ini juga memastikan bahwa fitur (piksel) dengan rentang nilai yang lebih besar tidak mendominasi fitur lainnya.
5. Pemisahan Data (Data Splitting)
- Proses: Membagi dataset menjadi tiga bagian yang terpisah:
  - Data Latih (Training Set): Bagian terbesar dari data yang digunakan untuk melatih model. Model akan belajar pola dan fitur dari data ini.
  - Data Validasi (Validation Set): Digunakan untuk mengevaluasi performa model selama proses pelatihan dan untuk melakukan penyesuaian hyperparameter. Ini membantu mencegah overfitting.
  - Data Uji (Test Set): Digunakan untuk menguji performa akhir model setelah pelatihan selesai. Data ini tidak boleh dilihat oleh model selama proses pelatihan atau validasi untuk memberikan evaluasi yang objektif.
Proporsi yang digunakan 80% latih, 10% validasi, 10% uji.
- Alasan: Penting untuk mengevaluasi seberapa baik model dapat melakukan generalisasi pada data baru yang belum pernah dilihat sebelumnya. Tanpa pemisahan yang benar, kita tidak bisa menilai performa model secara objektif.
6. Augmentasi Data (Data Augmentation)
- Proses: Membuat variasi baru dari gambar yang ada di data latih secara artifisial. Teknik umum meliputi:
  - Rotasi (memutar gambar dengan sudut tertentu).
  - Geser horizontal atau vertikal (horizontal/vertical shift).
  - Balik horizontal atau vertikal (horizontal/vertical flip).
  - Zoom (memperbesar atau memperkecil bagian gambar).
  - Perubahan kecerahan (brightness adjustment).
- Alasan:
Meningkatkan Jumlah Data Latih: Sangat berguna jika dataset asli tidak terlalu besar.
Mengurangi Overfitting: Dengan melihat lebih banyak variasi gambar, model menjadi lebih tangguh (robust) dan kurang cenderung menghafal data latih. Ini meningkatkan kemampuan generalisasi model pada data baru.
Membuat model tidak terlalu sensitif terhadap variasi posisi, orientasi, atau kondisi pencahayaan objek dalam gambar.
7. Pemformatan Data untuk Model (Data Formatting/Batching)
- Proses:
Mengubah gambar dan label ke dalam format yang sesuai untuk framework deep learning yang digunakan (misalnya, Tensor untuk TensorFlow atau PyTorch).
Mengelompokkan data ke dalam batch (kumpulan kecil data). Model akan dilatih dengan memproses satu batch data pada satu waktu.
- Alasan: Framework deep learning memiliki persyaratan format input tertentu. Pelatihan dengan batch lebih efisien secara komputasi dan dapat membantu proses optimasi (misalnya, gradient descent) menjadi lebih stabil.
