import numpy as np
import cv2

# carregando a imagem e convertendo para escala de cinza
image = cv2.imread('1d.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# calculo dos gradientes

gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

# subtracao dos gradientes
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)


# borramento e binarizacao da imagem
blurred = cv2.blur(gradient, (5, 5))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

# morfologia, para destacar elementos em formato retangular
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# erosoes e dilatacoers
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)

##### imagens do processo#####
def exibir_resultados(gx,gy,b,t,c):
    comparacao = cv2.vconcat((gradX,gradY))
    cv2.imshow('GradX x GradY',cv2.resize(comparacao,None, fx=0.3, fy=0.3))
    cv2.waitKey(0)

    comparacao2 = cv2.vconcat((blurred,thresh))
    cv2.imshow('Borrada x Binarizada',cv2.resize(comparacao2,None, fx=0.3, fy=0.3))
    cv2.waitKey(0)

    cv2.imshow('Fechamento + Erosao', cv2.resize(closed, None, fx=0.3, fy=0.3))
    cv2.waitKey(0)



# procurnando contornos na imagem binarizada e retornando o de maior tamanho
cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

#cv.findContours(image, modo, metodo)
#metodos: CHAIN_APPROX_SIMPLE e CHAIN_APPROX_NONE
#CHAIN_APPROX_NONE:armazena TODOS os pontos do contorno
#CHAIN_APPROX_NONE:remove os pontos redundantes e comprime o contorno

# calculo da caixa delimitadora para o maior contorno
#minAreaRect() -> encontra a menor area rotacionada de um retangulo
#retorno: (center(x, y), (width, height), angle of rotation)
rect = cv2.minAreaRect(c)
#vetor coordenadas da caixa:
box = np.int64(cv2.boxPoints(rect)) # rever esse variavel(tamanho)

print('(centro, largura, altura, angulo)): ',rect)
print('coordenadas da caixa: ',box)

# desenhar a caixa ao redor do codigo de barras
# cv2.drawContours(imagem, vetor coordenadas, se negativo:todos os contornos sao desenhados,cor, espessura)

###########exibir_resultados(gradX,gradY,blurred,thresh,closed)

cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
cv2.imshow("Image", cv2.resize(image, None, fx=0.7, fy=0.7))
cv2.waitKey(0)
