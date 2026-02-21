from ultralytics import YOLO
import cv2 as cv 

# 1. Carrega o modelo pré-treinado YOLOv8
model = YOLO("yolov8n.pt")  # versão leve e rápida

# 2. Carrega a imagem
img = cv.imread("data/person.png")  # substitua pelo nome correto da imagem

# 3. Realiza a inferência
results = model.predict(img, conf=0.40, verbose=False)

# 4. Conta quantas detecções são da classe "person"
person_count = 0

for result in results:
    for cls in result.boxes.cls:
        if int(cls) == 0:  # classe 0 no COCO = person
            person_count += 1

print(f"Pessoas detectadas: {person_count}")

# 5. Desenha bounding boxes verdes
result_img = img.copy()

for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])

        if cls == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv.putText(
    result_img,
    f"Pessoas: {person_count}",
    (10, 30),
    cv.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)


# 5. Salva imagem anotada
cv.imshow("Resultado YOLO", result_img)
cv.imwrite("outputs/project_2/person_anotado.png", result_img)
cv.waitKey(0)
cv.destroyAllWindows()