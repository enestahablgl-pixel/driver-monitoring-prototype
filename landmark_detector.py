import math

class LandmarkDetector:
    def __init__(self, ear_threshold=0.21, mar_threshold=0.5):
        # Bu değerleri duruma göre (kamera açısı vs.) daha sonra hassaslaştırabiliriz
        self.EAR_THRESHOLD = ear_threshold
        self.MAR_THRESHOLD = mar_threshold

        # MediaPipe yüz ağındaki spesifik göz ve ağız nokta indeksleri
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [78, 308, 13, 14] # Sol dış, Sağ dış, Üst iç, Alt iç

    def calculate_distance(self, p1, p2):
        """İki nokta (x,y) arasındaki Öklid uzaklığını hesaplar."""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def calculate_ear(self, eye_points, landmarks):
        """Göz için EAR hesaplaması"""
        # Dikey mesafeler
        v1 = self.calculate_distance(landmarks[eye_points[1]], landmarks[eye_points[5]])
        v2 = self.calculate_distance(landmarks[eye_points[2]], landmarks[eye_points[4]])
        # Yatay mesafe
        h = self.calculate_distance(landmarks[eye_points[0]], landmarks[eye_points[3]])
        
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def calculate_mar(self, landmarks):
        """Ağız için MAR hesaplaması"""
        # İç dudak dikey mesafesi
        v = self.calculate_distance(landmarks[self.MOUTH[2]], landmarks[self.MOUTH[3]])
        # Dış dudak yatay mesafesi
        h = self.calculate_distance(landmarks[self.MOUTH[0]], landmarks[self.MOUTH[1]])
        
        mar = v / (h + 1e-6) # Sıfıra bölünme hatasını önlemek için küçük bir değer ekliyoruz
        return mar

    def analyze_face(self, face_landmarks):
        """Tüm hesaplamaları yapıp durumu döndüren ana fonksiyon"""
        left_ear = self.calculate_ear(self.LEFT_EYE, face_landmarks)
        right_ear = self.calculate_ear(self.RIGHT_EYE, face_landmarks)
        
        # İki gözün ortalama açıklığını alıyoruz
        avg_ear = (left_ear + right_ear) / 2.0
        mar = self.calculate_mar(face_landmarks)

        # Eşik değerlerine göre durum analizi
        eyes_open = avg_ear > self.EAR_THRESHOLD
        mouth_open = mar > self.MAR_THRESHOLD

        return {
            "ear_degeri": round(avg_ear, 3),
            "mar_degeri": round(mar, 3),
            "gozler_acik": eyes_open,
            "agiz_acik": mouth_open
        }