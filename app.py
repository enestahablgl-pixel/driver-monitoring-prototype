import cv2
import mediapipe as mp
import numpy as np
import math
from face_detector import FaceDetector
from landmark_detector import LandmarkDetector

# =======================================================
# 1. YÜZ VE EL MODÜLLERİNİ BAŞLAT
# =======================================================
face_detector = FaceDetector(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
landmark_detector = LandmarkDetector(ear_threshold=0.21, mar_threshold=0.2)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# --- SİSTEM DURUM DEĞİŞKENLERİ ---
alarm_aktif = False
mevcut_risk_skoru = 0.0
noktalari_goster = False  # Yüz noktalarını açıp kapatmak için (Geliştirici Modu)

# --- EL İŞARETİ KONTROL FONKSİYONU ---
def el_acik_mi(hand_landmarks):
    fingers_tips = [8, 12, 16, 20]
    fingers_pips = [6, 10, 14, 18]
    acik_parmak_sayisi = 0
    for tip, pip in zip(fingers_tips, fingers_pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            acik_parmak_sayisi += 1
    return acik_parmak_sayisi >= 3

# =======================================================
# 2. EPFL 3B YÜZ MODELİNİ YÜKLE VE HİZALA
# =======================================================
txt_path = r"D:\staj\facedetector\epfl\epfl_3b_mean_model.txt"
tum_3b_noktalar = np.loadtxt(txt_path, dtype=np.float64, skiprows=1)

face_3d_model = np.array([
    tum_3b_noktalar[2],   # Burun Ucu
    tum_3b_noktalar[9],   # Çene Ucu
    tum_3b_noktalar[13],  # SOL GÖZ
    tum_3b_noktalar[10],  # SAĞ GÖZ
    tum_3b_noktalar[4],   # SOL DUDAK
    tum_3b_noktalar[3]    # SAĞ DUDAK
], dtype=np.float64)

burun_koordinati = face_3d_model[0].copy()
face_3d_model -= burun_koordinati

sol_goz = face_3d_model[2]
sag_goz = face_3d_model[3]
vektor = sag_goz - sol_goz

angle_y = np.arctan2(vektor[2], vektor[0])
R_y = np.array([
    [np.cos(angle_y), 0, np.sin(angle_y)],
    [0, 1, 0],
    [-np.sin(angle_y), 0, np.cos(angle_y)]
])
face_3d_model = np.dot(face_3d_model, R_y.T)

vektor_yeni = face_3d_model[3] - face_3d_model[2]
angle_z = np.arctan2(-vektor_yeni[1], vektor_yeni[0])
R_z = np.array([
    [np.cos(angle_z), -np.sin(angle_z), 0],
    [np.sin(angle_z),  np.cos(angle_z), 0],
    [0, 0, 1]
])
face_3d_model = np.dot(face_3d_model, R_z.T)

face_3d_model[:, 0] *= -1  
face_3d_model[:, 1] *= -1  

# --- PNP HAFIZA DEĞİŞKENLERİ ---
prev_rvec = None
prev_tvec = None

# =======================================================
# 3. CANLI KAMERA DÖNGÜSÜ
# =======================================================
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    
    # --- A. EL İŞLEME VE RESET KONTROLÜ ---
    results_hands = hands.process(rgb_image)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            if el_acik_mi(hand_landmarks) and alarm_aktif:
                alarm_aktif = False
                mevcut_risk_skoru = 0.0
                cv2.putText(image, "RESET SINYALI ALINDI!", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
    # --- B. YÜZ VE KAFA POZU (HEAD POSE) ---
    face_landmarks_list = face_detector.get_landmarks(image)
    
    if face_landmarks_list:
        for face_landmarks in face_landmarks_list:
            
            # =======================================================
            # YENİ: YÜZ NOKTALARINI ÇİZME (GELİŞTİRİCİ MODU)
            # =======================================================
            if noktalari_goster:
                # Tüm noktaları minik yeşil çiz
                for (px, py) in face_landmarks:
                    cv2.circle(image, (int(px), int(py)), 1, (0, 255, 0), -1)
                
                # Sadece matematiğe giren kritik noktaları büyük Kırmızı çiz
                kritik_noktalar = [1, 152, 33, 263, 61, 291, 10] # Burun, Çene, Gözler, Dudaklar ve Alın
                for idx in kritik_noktalar:
                    kx, ky = face_landmarks[idx]
                    cv2.circle(image, (int(kx), int(ky)), 4, (0, 0, 255), -1)

            # 1. EAR / MAR Analizi
            analiz = landmark_detector.analyze_face(face_landmarks)
            ear_degeri = analiz['ear_degeri']
            
            goz_durumu = "ACIK" if analiz["gozler_acik"] else "KAPALI"
            agiz_durumu = "ACIK" if analiz["agiz_acik"] else "KAPALI"
            renk_goz = (0, 255, 0) if analiz["gozler_acik"] else (0, 0, 255)
            cv2.putText(image, f"Goz: {goz_durumu} (EAR: {ear_degeri:.3f})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk_goz, 2)
            cv2.putText(image, f"Agiz: {agiz_durumu} (MAR: {analiz['mar_degeri']:.3f})", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if analiz["agiz_acik"] else (0,0,255), 2)
            
            # 2. SOLVEPNP - 2B Noktaları Ayarlama
            image_points = np.array([
                face_landmarks[1],    
                face_landmarks[152],  
                face_landmarks[33],   
                face_landmarks[263],  
                face_landmarks[61],   
                face_landmarks[291]   
            ], dtype=np.float64)
            
            focal_length = w
            cam_matrix = np.array([[focal_length, 0, w / 2],
                                   [0, focal_length, h / 2],
                                   [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)
            
            # 3. PnP Çözümü
            if prev_rvec is None:
                success_pnp, rvec, tvec = cv2.solvePnP(face_3d_model, image_points, cam_matrix, dist_coeffs)
            else:
                success_pnp, rvec, tvec = cv2.solvePnP(face_3d_model, image_points, cam_matrix, dist_coeffs, 
                                                       rvec=prev_rvec, tvec=prev_tvec, useExtrinsicGuess=True)
            
            if success_pnp:
                prev_rvec = rvec
                prev_tvec = tvec
                
                rmat, _ = cv2.Rodrigues(rvec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                
                pitch = angles[0] 
                yaw = angles[1]   
                roll = angles[2] 
                
                if pitch < -90: pitch += 180
                elif pitch > 90: pitch -= 180
                
                if roll < -90: roll += 180
                elif roll > 90: roll -= 180
                
                cv2.putText(image, f"Pitch (Yukari/Asagi): {int(pitch)}", (w - 280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(image, f"Yaw (Saga/Sola): {int(yaw)}", (w - 280, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(image, f"Roll (Yana Yatma): {int(roll)}", (w - 280, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # =======================================================
                # 3 EKSENLİ ORTAK MATEMATİKSEL RİSK SKORU
                # =======================================================
                ear_risk = max(0.0, min(1.0, (0.30 - ear_degeri) / 0.15))
                
                sapma_acisi = math.sqrt(pitch**2 + yaw**2 + roll**2)
                kafa_risk = max(0.0, min(1.0, sapma_acisi / 40.0))
                
                toplam_risk = math.sqrt(ear_risk**2 + kafa_risk**2)
                anlik_skor = min(100.0, toplam_risk * 100)
                
                # Asimetrik Filtre
                if anlik_skor > mevcut_risk_skoru:
                    mevcut_risk_skoru = (mevcut_risk_skoru * 0.98) + (anlik_skor * 0.02)
                else:
                    mevcut_risk_skoru = (mevcut_risk_skoru * 0.80) + (anlik_skor * 0.20)
                
                if mevcut_risk_skoru > 80.0:
                    alarm_aktif = True

                # Görselleştirme: Tehlike Barı
                bar_genislik = 300
                bar_doluluk = int((mevcut_risk_skoru / 100) * bar_genislik)
                
                if mevcut_risk_skoru < 40: bar_renk = (0, 255, 0)
                elif mevcut_risk_skoru < 75: bar_renk = (0, 255, 255)
                else: bar_renk = (0, 0, 255)
                
                cv2.rectangle(image, (20, h - 60), (20 + bar_genislik, h - 40), (255, 255, 255), 2) 
                cv2.rectangle(image, (20, h - 60), (20 + bar_doluluk, h - 40), bar_renk, -1)        
                cv2.putText(image, f"RISK SKORU: %{int(mevcut_risk_skoru)}", (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_renk, 2)

                # =======================================================
                # EKSEN ÇİZİMİ
                # =======================================================
                axis_length = 60.0
                axis_3d = np.float32([[-axis_length, 0, 0], [0, -axis_length, 0], [0, 0, -axis_length]])
                
                imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, cam_matrix, dist_coeffs)
                origin_proj, _ = cv2.projectPoints(np.float32([[0, 0, 0]]), rvec, tvec, cam_matrix, dist_coeffs)
                
                ox, oy = int(origin_proj[0][0][0]), int(origin_proj[0][0][1])
                
                vx_x, vy_x = int(imgpts[0][0][0]) - ox, int(imgpts[0][0][1]) - oy
                vx_y, vy_y = int(imgpts[1][0][0]) - ox, int(imgpts[1][0][1]) - oy
                vx_z, vy_z = int(imgpts[2][0][0]) - ox, int(imgpts[2][0][1]) - oy
                
                alin_x, alin_y = face_landmarks[10]
                
                cv2.line(image, (alin_x, alin_y), (alin_x + vx_x, alin_y + vy_x), (0, 0, 255), 3) 
                cv2.line(image, (alin_x, alin_y), (alin_x + vx_y, alin_y + vy_y), (0, 255, 0), 3) 
                cv2.line(image, (alin_x, alin_y), (alin_x + vx_z, alin_y + vy_z), (255, 0, 0), 3) 

    # --- C. ALARM ARAYÜZÜ ---
    if alarm_aktif:
        cv2.rectangle(image, (0, 0), (w, h), (0, 0, 255), 10) 
        cv2.putText(image, "TEHLIKE: DIKKAT DAGINIKLIGI VEYA UYKU!", (int(w*0.05), h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(image, "Susturmak icin elini kaldir", (int(w*0.2), (h//2) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
    cv2.imshow('DMS: Ortak Risk Skoru Sistemi', image)
    
    # KLAVYE KONTROLLERİ
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):  # N tuşuna basıldığında noktaları aç/kapat
        noktalari_goster = not noktalari_goster

cap.release()
cv2.destroyAllWindows()