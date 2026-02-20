import cv2

# Carrega imagem
img = cv2.imread("data\graos.png")

# Escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Filtro Gaussiano
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Binarização
_, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

# Encontra contornos
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Desenha contornos
result = img.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# Resultado
print(f"Nº zero de objetos encontrados: {len(contours)}")

# Mostra
cv2.imshow("Original", img)
cv2.imshow("Binarizada", binary)
cv2.imshow("Contornos", result)
cv2.imwrite("resultado_anotado.png", result)

cv2.waitKey(0)
cv2.destroyAllWindows()