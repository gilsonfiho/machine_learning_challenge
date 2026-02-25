# Machine Learning Challenge

Repositório com implementações de desafios de visão computacional e machine learning.

---

## Questão 1: Detecção de Grãos por Contornos

**Arquivo:** [1.py](1.py)

**Descrição:**
Implementação de um sistema de detecção automática de grãos em uma imagem usando técnicas de processamento de imagem e análise de contornos.

**Metodologia:**
- Carregamento da imagem original
- Conversão para escala de cinza
- Aplicação de filtro Gaussiano para suavização
- Binarização automática com Otsu
- Detecção de contornos externos
- Filtragem de contornos por área mínima
- Visualização e anotação dos resultados

**Resultado:**
![Grãos Detectados](outputs/project_1/graos_anotado.png)

---

## Questão 2: Detecção de Pessoas com YOLOv8

**Arquivo:** [2.py](2.py)

**Descrição:**
Implementação de um sistema de detecção de pessoas em imagens usando o modelo YOLOv8 (pré-treinado no dataset COCO).

**Metodologia:**
- Carregamento do modelo YOLOv8 nano (versão otimizada)
- Inferência na imagem com confiança mínima de 40%
- Filtragem de detecções da classe "person" (classe 0 no COCO)
- Desenho de bounding boxes verdes ao redor das pessoas
- Contagem e anotação dos resultados

**Resultado:**
![Pessoas Detectadas](outputs/project_2/person_anotado.png)

---