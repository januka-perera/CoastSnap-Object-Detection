import cv2, json
img = cv2.imread("reference.jpg")
with open("reference.json") as f:
    kps = json.load(f)
for name, (x, y) in kps.items():
    cv2.circle(img, (int(x), int(y)), 15, (0, 0, 255), -1)
    cv2.putText(img, name, (int(x)+10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
cv2.imwrite("reference_check.png", img)