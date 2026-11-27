import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self, static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5):
        # MediaPipe Face Mesh modülünü standart yoldan başlatıyoruz
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True, # Göz ve dudak çevresi için ekstra hassasiyet
            min_detection_confidence=min_detection_confidence
        )

    def get_landmarks(self, image):
        """
        Görüntüyü alır, MediaPipe'a sokar ve yüz landmarklarını döndürür.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        h, w, c = image.shape
        
        face_landmarks_list = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]
                face_landmarks_list.append(landmarks)
                
        return face_landmarks_list