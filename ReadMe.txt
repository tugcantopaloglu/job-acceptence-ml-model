# Makine Öğrenmesi (BLM5110) dersi kapsamında geliştirilen Logistic Regression ödevi. 

# Hazırlayan - Tuğcan Topaloğlu

## Proje Açıklaması
Bu projede Logistic Regression algoritmasını kullanarak sınıflandırma işlemlerini gerçekleştirmektedir. Proje boyunca L2 regularization, early stopping, ve threshold optimizasyonu gibi yöntemler uygulanmıştır. Amaç modelin performansını artırırken overfitting'i önlemektir.

## Dosya Yapısı
- data/:
  - hw1Data.txt: Ham veri dosyası.
  - train_data.txt: Eğitim veri seti.
  - validate_data.txt: Doğrulama veri seti.
  - test_data.txt: Test veri seti.
- results/:
  - epoch_loss_output.txt: Her epoch için loss değerleri.
  - weights.txt: Modellerin ağırlıkları.
  - scores.txt: Modellerin performans skorları (Accuracy, Precision, Recall, F1-score).
  - Data_Plot.png: Eğitim verilerinin scatter plot görselleştirmesi.
  - Train_Before_Optimization.png: Eğitim loss grafiği (optimizasyon öncesi).
  - Validate_Before_Optimization.png: Doğrulama loss grafiği (optimizasyon öncesi).
  - Train_After_Optimization.png: Eğitim loss grafiği (optimizasyon sonrası).
  - Validate_After_Optimization.png: Doğrulama loss grafiği (optimizasyon sonrası).
- main.py: Ana çalışma dosyası. Veri işleme, model eğitimi ve test işlemleri bu dosyada gerçekleştirilir.
- DataHandler.py: Veri yükleme, bölme ve görselleştirme işlemlerini yapan modül.
- LogisticRegression.py: Logistic Regression algoritmasını içeren modül.

## Kullanılan Algoritmalar ve Teknikler
1. Logistic Regression
2. L2 Regularization
3. Early Stopping
4. Threshold Optimizasyonu

## Projeyi Çalıştırma
1. Gerekli kütüphaneleri yükleyin:
   - NumPy
   - Pandas
   - Matplotlib
   - Scikit-learn

   - "pip install -r requirements.txt" ile gerekli kütüphaneler yüklenebilir.

2. Verileri hazırlayın:
- `data/` klasöründeki veriler kullanıma hazırdır. Eğer veriler bölünmemişse, `main.py` dosyası çalıştırıldığında bölme işlemi otomatik olarak yapılır.

3. Ana dosyayı çalıştırın:
   - "python main.py"

4. Sonuçları inceleyin:
   - Performans sonuçlarını `results/scores.txt` dosyasından kontrol edebilirsiniz. Bu dosyada yoksa program tarafından otomatik oluşturulur. Varsa sonuçlar dosya sonuna eklenir.
   - Grafikler `results/` klasöründe bulunmaktadır.

## Notları
- Bu programda tüm çıktı dosyalarında dosyada yoksa program tarafından otomatik oluşturulur. Varsa sonuçlar dosya sonuna eklenir
- Bu proje Python 3.8 veya üzeri bir sürüm gerektirir. Python 3.10 ve 3.12 ile test edilmiş ve çalıştırılmıştır.
- Veriler `./data/` dizininde bulunmalıdır.
- Sonuçlar `./results/` dizinine kaydedilmektedir.
- İlgili kütüphanelerin son sürümleri kullanılabilir ancak bu projede uygulanmış sürümleri de aşağıda eklenmiştir.
- Programı 2 ortamda sorunsuz çalıştırabildim bir problemle rastlarsanız bana iletmenizi rica ederim.

## Requirements sürümler:
matplotlib==3.9.2
numpy==2.1.3
scikit-learn==1.5.2
pandas==2.2.3
os (default pakettir python içerisinde bulunur)
