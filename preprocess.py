import cv2,sys
import mediapipe as mp
import numpy as np
#import iris_tools
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


input_file = sys.argv[1]
output_file = sys.argv[2]

cap = cv2.VideoCapture(input_file)


out = cv2.VideoWriter(output_file,cv2.VideoWriter_fourcc(*"MJPG"), 25, (640,360))

with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=True) as holistic:
    while cap.isOpened():
        ret, image = cap.read()
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks != None and results.face_landmarks != None:
            print(
                f'Nose coordinates: ('
                f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
                f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
            )
            # iris_tools.iris(
            #     image, results.face_landmarks.landmark)
        annotated_image = image.copy()
        if None in np.stack((results.segmentation_mask,) * 3, axis=-1):
            continue
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        # bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)
        mp_drawing.draw_landmarks(
            annotated_image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.
            get_default_pose_landmarks_style())
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)

        # cv2.imshow("123", annotated_image)
        # cv2.waitKey(1) & 0xff == ord('q')
        annotated_image = cv2.resize(annotated_image, (640, 360))
        out.write(annotated_image)
        cv2.imshow("123", annotated_image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
cap.release()
out.release()

cv2.destroyAllWindows()