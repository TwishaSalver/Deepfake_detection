import preprocessing
from PIL import Image
import numpy as np
import traceback
print('MEDIAPIPE_OK', preprocessing._MEDIAPIPE_OK)
print('face_mesh attr', hasattr(preprocessing, '_mp_face_mesh'))
print('face_mesh before', preprocessing._mp_face_mesh)
img = Image.open(r'C:\Users\Twisha Salver\Downloads\Deepfake_detection-main\Deepfake_detection-main\data\raw\real\real_0000.jpg').convert('RGB')
rgb = np.array(img)
try:
    fm = preprocessing._get_face_mesh()
    print('fm', fm)
    res = fm.process(rgb)
    print('res', res)
    print('multi_face_landmarks', res.multi_face_landmarks)
    if res.multi_face_landmarks:
        print('landmark count', len(res.multi_face_landmarks[0].landmark))
except Exception:
    traceback.print_exc()
