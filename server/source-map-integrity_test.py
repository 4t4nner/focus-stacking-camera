import cv2, numpy as np, os

session = "debug_output/session_20260618_154158_363480"
fm = cv2.imread(os.path.join(session, "focusmap_raw.png"), cv2.IMREAD_GRAYSCALE)

total = fm.size
# unfocused-пиксели = пиксели, не назначенные ни одному источнику.
# В вашем пайплайне после fill/Вороного таких быть не должно.
# Проверяем распределение по зонам:
vals, counts = np.unique(fm, return_counts=True)
for v, c in zip(vals, counts):
    print(f"zone value {v}: {c} px ({100*c/total:.2f}%)")
# Если используете отдельную маску unfocused — подставьте её сюда.
print("Total pixels:", total)