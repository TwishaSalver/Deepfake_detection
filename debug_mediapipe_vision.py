import os
import mediapipe as mp
import pkgutil
import mediapipe.tasks.python.vision as v
print('vision attrs', [name for name in dir(v) if not name.startswith('_')])
print('vision modules', [m.name for m in pkgutil.iter_modules(v.__path__)])
