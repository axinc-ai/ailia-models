# mediapipe version of facemesh for verification

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

import sys
args = sys.argv
if len(args) >= 2:
  file = args[1]
else:
  file = "man.jpg"

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,  # False for blazeface
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5) as face_mesh:

  image = cv2.imread(file)
  results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

  annotated_image = image.copy()
  for face_landmarks in results.multi_face_landmarks:
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(
                      color=(0,255,0), thickness=1))
  cv2.imwrite('output_mediapipe.png', annotated_image)
