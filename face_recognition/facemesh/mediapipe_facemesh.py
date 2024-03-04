# mediapipe version of facemesh for verification

import sys
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Get input file name
args = sys.argv
if len(args) >= 2:
  file = args[1]
else:
  file = "man.jpg"

# Video mode
if ".mp4" in file:
  capture = cv2.VideoCapture(file)
  f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
  f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
  writer = cv2.VideoWriter("mediapipe.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), frameSize=(f_w, f_h), fps=20)

# Open mediapipe
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,  # True for attention model
    min_detection_confidence=0.5) as face_mesh:

  if ".mp4" in file:
    # Video mode
    while(True):
      ret, image = capture.read()
      if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
          break
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
      writer.write(annotated_image)
      cv2.imshow('frame', annotated_image)
    capture.release()
    if writer is not None:
        writer.release()
  else:
    # Image mode
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

