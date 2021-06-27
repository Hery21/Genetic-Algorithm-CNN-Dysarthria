# Genetic-Algorithm-CNN-Dysarthria
IMPLEMENTASI

5.1 Persiapan Data
Data yang akan digunakan terlebih dahulu dipraproses menggunakan aplikasi Audacity dan program dari kode python. Data yang telah dipraproses ini kemudian akan disimpan dalam Google Drive. Kode pemuatan pustaka yang digunakan dapat dilihat pada Gambar 5.1

Gambar 5.1 Pustaka yang digunakan pada penelitian ini

5.1.1 Prapemrosesan
Prapemrosesan dilakukan dengan menggunakan aplikasi Audacity sesuai pada subbab 4.2.
    1. Menggunakan efek Noise Reduction untuk mengurangi noise pada data suara agar suara ucapan menjadi lebih jelas.
    2. Menggunakan efek Normalize untuk memperjelas dan mengurangi perbedaan tiap sesi rekaman.
    3. Mencari bagian awal dan akhir ucapan menggunakan fitur Sound Finder.
    4. Menambahkan Silence sebelum dan sesudah ucapan secara seimbang sehingga panjang data menjadi sama 1870 ms dan suara ucapan berada di tengah.
Pada Gambar 5.2 dapat dilihat tangkapan layar beberapa data suara yang dimuat pada aplikasi Audacity dan Gambar 5.3 adalah tangkapan layar dari data suara ini setelah dilakukan prapemrosesan. Pada Gambar 5.2 masih dapat terlihat noise pada beberapa data suara dan juga panjang rekaman tiap data suara yang berbeda menjadi tidak ber-noise dan memiliki ukuran rekaman yang sama pada Gambar 5.3.

Gambar 5.2 Beberapa contoh data suara sebelum prapemrosesan


Gambar 5.3 Beberapa contoh data suara setelah prapemrosesan

5.1.2 Ekstraksi Fitur
Ekstraksi fitur dilakukan dengan menggunakan pustaka python_speech_features yaitu fungsi mfcc untuk mengekstrak fitur MFCC dan delta untuk mendapatkan turunan dan turunan dari turunan dari fitur MFCC. Pustaka scipy digunakan untuk membaca data suara yang memiliki format wav. Kode implementasi ekstraksi fitur dapat dilihat pada Gambar 5.4.

Gambar 5.4 Kode ekstraksi fitur mfcc, delta, dan delta-delta

5.1.3 Pemisahan Data Latih dan Data Uji
	Setelah diekstraksi fiturnya, data kemudian dibagi menjadi dua bagian untuk data latih dan data uji. Pembagian data latih dan data uji berdasarkan pada sesi rekaman yaitu ada tiga sesi. Perbandingan antara data latih dan data uji adalah 2:1 dengan sesi rekaman pertama dan kedua sebagai data latih dan sesi rekaman ketiga sebagai data uji. Implementasi pemisahan data latih dan data uji dapat dilihat pada Gambar 5.5.

Gambar 5.5 Kode pemisahan data latih dan data uji

5.1.4 Standarisasi Fitur
	Standarisasi fitur dilakukan setelah data yang diekstraksi fiturnya dibagi menjadi data latih dan data uji. Mean dan standar deviasi dihitung dari data latih dan kemudian data latih dan data uji distandarisasi dengan cara menguranginya dengan mean dan kemudian dibagi dengan standar deviasinya.
Data kemudian disimpan dalam format .pkl dan kemudian disimpan pada Google Drive agar mudah diakses menggunakan layanan Google Colabolatory. Jenis file .pkl digunakan untuk memudahkan penyimpanan dan akses data yang akan diproses pada bahasa python dengan memanfaatkan pustaka pickle. Implementasi dari standarisasi fitur dan penyimpanan dalam bentuk .pkl ini dapat dilihat pada Gambar 5.6.

Gambar 5.6 Kode standarisasi fitur menggunakan data latih

5.2 Implementasi Algoritma Genetika
	Implementasi algoritma genetika ini memanfaatkan pustaka numpy dan random dan CNN-nya memanfaatkan pustaka keras. Dikarenakan penggunaan layanan Google Drive untuk mengakses data yang akan diproses pada layanan Google Colaboratory maka dilakukan akses terlebih dahulu menggunakan pustaka google.colab. Kode pemuatan data ini dapat dilihat pada Gambar 5.7.

Gambar 5.7 Kode pemuatan data dari Google Drive ke Google Collaboratory

5.2.1 Pembangkitan Populasi Awal
	Proses pembangkitan populasi awal dimulai dengan menentukan batasan-batasan untuk gen. Kemudian beberapa kromosom dibentuk dari batasan-batasan ini yang kemudian dimasukkan ke dalam sebuah populasi. Kromosom-kromosom yang dibangkitkan dibuat agar tidak sama satu sama lainnya agar memperluas ruang pencarian. Implementasi fungsi pembangkitan populasi ini dapat dilihat pada Gambar 5.8.

Gambar 5.8 Kode pembangkitan populasi

5.2.2 Pembangkitan Model CNN untuk Nilai Fitness
	Individu-individu pada setiap populasi masing-masing merepresentasikan satu model CNN tersendiri yang nilai akurasinya akan digunakan sebagai nilai fitness untuk algoritma genetikanya. Model dibangkitkan dengan mengambil informasi ukuran dan jumlah filter dari kromosom dan menambahkan max-pooling layer setelah tiap layer konvolusi, kemudian ditambahkan fully-connected layer dengan 10 node untuk melakukan klasifikasi. Model CNN yang dibangkitkan ini dilatih menggunakan data latih. Setelah dilatih, pada model ini kemudian dilakukan pengujian yaitu dilakukan feedforward data uji untuk menilai keakuratan prediksi. Prediksi dilakukan pada 10 kelas yaitu kelas 0 sampai 9 untuk ucapan 0 sampai 9. Prediksi yang dilakukan oleh model dibandingkan dengan label data sebenarnya dan dijumlahkan prediksi yang sesuai dengan labelnya. Prediksi yang sesuai ini kemudian dirasiokan dengan jumlah data uji sehingga didapatkan nilai akurasi. Nilai akurasi atau fitness ini kemudian akan disimpan pada array populasi sesuai pada individunya. Untuk membuat layer-layer model CNN ini dimanfaatkan pustaka keras yaitu fungsi Sequential, rmsprop, Conv2D, MaxPooling2D, Dense, dan Flatten. Setiap layer konvolusi diikuti oleh layer MaxPooling2D. Implementasi fungsi penghitungan nilai fitness dengan membangun model dan menghitung akurasinya dapat dilihat pada Gambar 5.9.

Gambar 5.9 Kode penghitungan nilai fitness individu dengan menghitung akurasi CNN

5.2.3 Implementasi Crossover dan Mutasi
	Crossover pada penelitian ini mengimplementasikan one-point crossover sehingga hanya satu titik potong yang ditentukan secara acak antara dua kromosom yang dipilih secara acak dari populasi. Nilai probabilitas crossover dikalikan dengan jumlah individu pada populasi dan dibagi dua untuk menghasilkan individu sesuai dengan probabilitasnya. Pada penelitian ini, proses mutasi dapat digabungkan pada fungsi crossover dikarenakan mutasi diterapkan hanya pada kromosom hasil crossover. Batasan-batasan untuk mutasi mengikuti batasan-batasan pada pembangkitan populasi. Berdasarkan pada batasan-batasan ini mutasi kemudian diimplementasikan secara acak pada gen sebanyak probabilitas mutasi dikalikan jumlah gen pada kromosom. Jumlah individu yang dihasilkan pada fungsi ini bersesuaian dengan jumlah individu hasil crossover. Implementasi fungsi untuk melakukan crossover dapat dilihat pada Gambar 5.10 dan mutasi pada Gambar 5.11.

Gambar 5.10 Kode implementasi crossover


Gambar 5.11 Kode implementasi mutasi

5.2.4 Seleksi untuk Populasi Baru
	Pada individu-individu baru yang dihasilkan melalui crossover dan mutasi kemudian diimplementasikan perhitungan fitness dan dimasukkan ke dalam populasi yang telah ada dan kemudian diurutkan berdasarkan pada nilai fitnessnya. Pengambilan individu untuk populasi baru ini menggunakan sistem elitism sehingga hanya individu sebanyak jumlah populasi yang telah ditentukan akan menjadi bagian dari populasi generasi berikutnya. Untuk mengatasi individu-individu yang memiliki nilai fitness yang sama, maka ditentukan lagi parameter kedua untuk mengurutkan individu-individu yaitu jumlah dari perkalian nilai ukuran layer dan jumlah layer. Hal ini dilakukan agar individu yang lebih efisien atau membutuhkan komputasi yang lebih rendah yang akan berlanjut pada generasi berikutnya.

5.2.5 Implementasi Pengurutan Algoritma Genetika
	Fungsi-fungsi yang telah didefinisikan kemudian disusun dan diurutkan untuk membentuk suatu algoritma genetika. Pertama-tama dibangkitkan populasi awal kemudian dihitung nilai fitness dari tiap individu ini. Lalu sebanyak jumlah generasi, dilakukan pembaruan populasi dengan penambahan individu hasil crossover dan mutasi dengan nilai fitness-nya berdasarkan pada nilai probabilitas crossover dan mutasi. Implementasi penyusunan kode fungsi untuk membentuk satu fungsi algoritma genetika dapat dilihat pada Gambar 5.12.
