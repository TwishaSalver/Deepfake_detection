import json
from pathlib import Path
from ensemble import predict_deepfake
from models import models_available

root = Path('.')
real_dir = root / 'data' / 'raw' / 'real'
paths = sorted(real_dir.glob('*.jpg'))[:5]
print('models available:', models_available())
for p in paths:
    print('\n===', p.name)
    result = predict_deepfake(str(p), demo_mode=False)
    print(json.dumps(result, indent=2))
