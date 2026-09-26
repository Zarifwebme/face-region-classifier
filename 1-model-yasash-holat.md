# Yuzni Regionlarga Tasniflovchi Deep Learning Modeli: Yaratish va O'qitish Hisoboti
**Fayl:** `1_model_training_report.md`

## 1. Loyiha Maqsadi
Ushbu bosqichning asosiy maqsadi inson yuzini 5 ta turli mintaqa (American, African, Asian, Indian, European) bo'yicha farqlovchi va tasniflovchi aniqligi yuqori bo'lgan Konvolyutsion Neyron Tarmog'i (CNN) modelini yaratish va o'qitishdan iborat edi. Loyiha Google Colab bulutli platformasida amalga oshirildi.

## 2. Ma'lumotlarni Tayyorlash va Infratuzilma (Google Colab)
* **Xotira Optimizatsiyasi:** Dastlab dataset to'g'ridan-to'g'ri Google Drive orqali ulanib o'qitilganda jarayon juda sekin kechdi. Buni tezlashtirish maqsadida arxivlangan ma'lumotlar to'plami Colab'ning lokal SSD xotirasiga (`/content/local_dataset`) ko'chirildi. Natijada I/O (o'qish/yozish) tezligi sezilarli oshdi va o'qitish vaqti qisqardi.
* **Data Augmentation:** Rasmlardagi xilma-xillikni oshirish va model yodlab olishining (overfitting) oldini olish uchun Pytorch `transforms` orqali tasvirlarni qirqish, burish va rangini biroz o'zgartirish (Color Jitter) kabi usullar qo'llanildi. Rasmlar model uchun standart `224x224` o'lchamga keltirilib, normallashtirildi.

## 3. Model Arxitekturasi va O'qitish Strategiyasi
* **Baza Model:** Pytorch kutubxonasidagi pre-trained **ResNet18** arxitekturasi tanlandi. Modelning oxirgi qatlami (`fc`) 5 ta sinfga moslashtirib o'zgartirildi.
* **Class Imbalance (Sinflar disbalansi):** Dastlabki 5 epochlik bazaviy o'qitishda model 77% test aniqligini ko'rsatdi. Tahlil natijasida *European* va *Indian* sinflarida ma'lumotlar kamligi yoki o'xshashligi sababli aniqlik past ekani aniqlandi. Buni hal qilish uchun **Weighted Cross-Entropy Loss** (har bir sinfning datasetdagi soniga qarab turlicha og'irlik beruvchi yo'qotish funksiyasi) qo'llanildi.
* **Giperparametrlar va Optimizatsiya:** 
  * Optimizer sifatida **Adam** ishlatildi.
  * O'qitish tezligini avtomatik moslashtirish uchun **CosineAnnealingLR** rejalashtirgichidan (scheduler) foydalanildi.
  * Modelning barcha qatlamlari (Full Fine-Tuning) yangi dataset asosida qayta o'qitildi.

## 4. Olingan Natijalar
Yuqoridagi optimizatsiya va texnikalar natijasida modelning umumiy test aniqligi **80% ga ko'tarildi**. Muvaffaqiyatli o'qitilgan model parametrlari keyingi bosqichlarda lokal kompyuterda ishlatish uchun `region_model_optimized.pth` nomi bilan saqlab olindi.