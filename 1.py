import cv2 as cv

#1 Carrega imagem
img = cv.imread("data\graos.png")

#2 Escala de cinza
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#3 Filtro Gaussiano
blurred = cv.GaussianBlur(gray, (5, 5), 0)

#4 Binarização
_, binary = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

#5 Encontra contornos
contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#6 Suaviaza contornos
contours = [cnt for cnt in contours if cv.contourArea(cnt) > 100]

#7 Desenha contornos
result = img.copy()
cv.drawContours(result, contours, -1, (0, 255, 0), 2)
cv.putText(result, f"count: {len(contours)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#8 Resultado
print(f"Nº de objetos encontrados: {len(contours)}")

# Exibição e salvamento
cv.imshow("Original", img)
cv.imshow("Binarizada", binary)
cv.imshow("Contornos", result)
cv.imwrite("outputs/project_1/graos_anotado.png", result)

cv.waitKey(0)
cv.destroyAllWindows()