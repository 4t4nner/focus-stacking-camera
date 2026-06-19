import cv2, numpy as np, glob, os, json

def sharpness(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())

session = "debug_output/session_20260618_154158_363480"
inputs = sorted(glob.glob(os.path.join(session, "input_*.jpg")))
result = os.path.join(session, "composite_final.jpg")

s_inputs = [sharpness(p) for p in inputs]
s_result = sharpness(result)

print("Per-input sharpness:", [round(x,1) for x in s_inputs])
print("Mean input:", round(np.mean(s_inputs),1))
print("Max input :", round(max(s_inputs),1))
print("Result    :", round(s_result,1))
print("Gain vs mean: x%.2f" % (s_result/np.mean(s_inputs)))
print("Gain vs max : x%.2f" % (s_result/max(s_inputs)))