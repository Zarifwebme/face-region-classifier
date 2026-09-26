# Yuzni Regionlarga Tasniflovchi Model: Lokal Muhitda va Kamerada Ishlatish Hisoboti
**Fayl:** `2_model_deployment_report.md`

## 1. Loyiha Maqsadi
Google Colab'da o'qitilgan `region_model_optimized.pth` modelini lokal Windows kompyuteriga (laptop) tushirish, kameradan olingan real vaqt (real-time) videokadrlar asosida inson yuzini aniqlash va uning qaysi mintaqaga tegishli ekanligini bashorat qilishni ta'minlovchi `app.py` skriptini yaratish.

## 2. Lokal Muhitni Sozlash (VS Code & Virtual Environment)
Dastur boshqa global paketlar bilan to'qnashmasligi uchun loyiha papkasida alohida Python virtual muhiti (`.venv`) yaratildi va VS Code orqali ulandi. Kerakli kutubxonalarni o'rnatish jarayonida quyidagi qaramlik (dependencies) versiyalari `requirements.txt` faylida qat'iy belgilab olindi:
* `numpy>=1.24.0,<2.0.0`
* `opencv-python==4.8.1.78`
* `torch>=2.0.0`, `torchvision>=0.15.0`, `pillow>=9.5.0`

## 3. Duch Kelingan Texnik Muammolar va Ularning Yechimi
Lokal muhitni sozlashda bir nechta ziddiyatlar yuzaga keldi va muvaffaqiyatli bartaraf etildi:
* **OpenCV C++ DLL / CascadeClassifier Xatoligi:** Windows tizimida OpenCV'ning so'nggi (4.9/4.10) versiyalari to'liq yuklanmay, `CascadeClassifier` topilmaslik xatosi (`AttributeError`) berdi. Bu OpenCV versiyasini barqaror **4.8.1.78** versiyasiga tushirish orqali hal qilindi.
* **NumPy 2.x To'qnashuvi:** OpenCV 4.8.1.78 kutubxonasi NumPy'ning yangi 2.0+ versiyasi bilan mos kelmay qoldi va `ImportError: numpy.core.multiarray failed to import` xatosi kelib chiqdi. Yechim sifatida NumPy paketini 1.x versiyasiga tushirish (`pip install "numpy<2"`) orqali konflikt to'liq bartaraf qilindi.
* **Haar Cascade Yo'li:** `cv2.data.haarcascades` tizim yo'llarida muammo qilmasligi uchun XML fayli to'g'ridan-to'g'ri OpenCV GitHub omboridan lokal papkaga yuklab olinib, o'qiladigan qilib sozlandi.

## 4. Dastur Mantiqiy Ketma-ketligi (`app.py`)
Model lokal kamerada ishga tushganda quyidagi qadamlar siklik ravishda bajariladi:
1. **Kadr Olish:** `cv2.VideoCapture(0)` yordamida veb-kameradan kadr olinadi.
2. **Yuzni Aniqlash:** Olingan kadr kulrang (Grayscale) formatga o'tkazilib, Haar Cascade usulida (`haarcascade_frontalface_default.xml`) yuz sohasi aniqlanadi.
3. **Qayta Ishlash (Preprocessing):** Yuz qirqib olinib (Crop), BGR dan RGB formatga, so'ngra PIL rasmiga aylantiriladi. Colab'dagi transformatsiyalar (224x224 ga o'zgartirish, Tenzorga o'tkazish, Normallashtirish) qo'llaniladi.
4. **Bashorat Qilish (Inference):** Tenzor PyTorch modeliga uzatiladi. `Softmax` funksiyasi yordamida ehtimolliklar hisoblanadi va eng yuqori ehtimollikka ega bo'lgan region (American, African, Asian, Indian, European) tanlanadi.
5. **Natijani Ekranga Chiqarish:** Kadrda yuz atrofida to'rtburchak chiziladi va uning ustida taxmin qilingan mintaqa nomi hamda ishonch foizi (`%`) ko'rsatiladi. Dastur `q` tugmasi bosilgunga qadar real vaqtda ishlaydi.