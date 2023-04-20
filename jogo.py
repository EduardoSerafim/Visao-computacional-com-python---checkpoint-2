import cv2
import os,sys, os.path
from cv2 import Mat
import numpy as np
import math

#importes para emular precionamento de teclas
from pynput.keyboard import Key, Controller
import time
import random

#Inicializa o controle 
keyboard = Controller()

font = cv2.FONT_HERSHEY_SIMPLEX

#PONTUAÇÃO DE CADA JOGADOR
pontuação_j1 = 0
pontuação_j2 = 0

#DEIXAR A IMAGEM MENOR
def ajustarImagem(img):
    return cv2.resize(img, (0, 0), None, 0.5, 0.5)

#IMPORTANDO AS IMAGEMS
template_papel = ajustarImagem(cv2.imread("papel.png", 0))
template_tesoura = ajustarImagem(cv2.imread("tesoura.png", 0))
template_pedra = ajustarImagem(cv2.imread("pedra.png", 0))


#INVERTENDO AS IMAGENS PARA O IDENTIFICAR O JOGADOR 2
papel_j2 = cv2.flip(template_papel, -1)
tesoura_j2 = cv2.flip(template_tesoura, -1)
pedra_j2 = cv2.flip(template_pedra, -1)

jogada_anterior_j1 = ""
jogada_anterior_j2 = ""
resultado = ""

#LÓGICA DA PARTIDA DE JOKENPO
def jankenpo(jogada_j1, jogada_j2):
    global pontuação_j1
    global pontuação_j2
    
   
    if jogada_j1 == "papel" and jogada_j2 == "pedra" or jogada_j1 == "tesoura" and jogada_j2 == "papel" or jogada_j1 == "pedra" and jogada_j2 == "tesoura": #VITÓRIA DO JOGADOR 1
        pontuação_j1 += 1
        return  "JOGADOR 1 VENCEU"
    elif jogada_j2 == "papel" and jogada_j1 == "pedra" or jogada_j2 == "tesoura" and jogada_j1 == "papel" or jogada_j2 == "pedra" and jogada_j1 == "tesoura": #VITÓRIA DO JOGADOR 2
        pontuação_j2 += 1
        return "JOGADOR 2 VENCEU"
    else:
        return "EMPATE"



#IDENTIFICAR MÃO DO JOGADOR 1
def jogada_jogador1(img_gray, img_rgb):
    
    #PAPEL
    res_papel = cv2.matchTemplate(img_gray,template_papel,cv2.TM_SQDIFF_NORMED)
    
    #MIN VAL PARA VERIFICAR O NÌVEL DE COMPATIBILIDADE DO TEMPLATE COM A IMAGEM PARA EVITAR FALSOS POSITVOS
    min_val_pa,_, min_loc_pa, _ = cv2.minMaxLoc(res_papel)
    _, altura_pa = template_papel.shape[::-1]
    if min_val_pa < 0.019: 
        cv2.putText(img_rgb, "Papel", (min_loc_pa[0] , min_loc_pa[1] + altura_pa + 30), font,1,(255,0,0),1,cv2.LINE_AA)
        return "papel"    
    
    #TESOURA
    res_tesoura = cv2.matchTemplate(img_gray,template_tesoura,cv2.TM_SQDIFF_NORMED)
    min_val_te, _, min_loc_te, _ = cv2.minMaxLoc(res_tesoura)
    _, altura_te = template_tesoura.shape[::-1]
    if min_val_te < 0.030:
        cv2.putText(img_rgb, "Tesoura", (min_loc_te[0] , min_loc_te[1] + altura_te + 30), font,1,(255,0,0),1,cv2.LINE_AA)
        return "tesoura"
 
    #PEDRA
    res_pedra = cv2.matchTemplate(img_gray,template_pedra,cv2.TM_SQDIFF_NORMED)
    min_val_pe, _, min_loc_pe, _ = cv2.minMaxLoc(res_pedra)
    _, altura_pe = template_pedra.shape[::-1]
    if min_val_pe < 0.0098:        
        cv2.putText(img_rgb, "Pedra", (min_loc_pe[0] , min_loc_pe[1] + altura_pe + 30), font,1,(255,0,0),1,cv2.LINE_AA)
        return "pedra"


#IDENTIFICAR MÃO DO JOGADOR 2
def jogada_jogador2(img_gray, img_rgb):
    
    #PAPEL  
    res_papel_j2 = cv2.matchTemplate(img_gray,papel_j2,cv2.TM_SQDIFF_NORMED)
    min_val_pa2, _, min_loc_pa2, _ = cv2.minMaxLoc(res_papel_j2)
    _, altura_pa2 = papel_j2.shape[::-1]
    if min_val_pa2 < 0.019:
        cv2.putText(img_rgb, "Papel", (min_loc_pa2[0] , min_loc_pa2[1] + altura_pa2 + 30), font,1,(255,0,0),1,cv2.LINE_AA)
        return "papel"
  
    #TESOURA
    res_tesoura_j2 = cv2.matchTemplate(img_gray,tesoura_j2,cv2.TM_SQDIFF_NORMED)
    min_val_te2, _, min_loc_te2, _ = cv2.minMaxLoc(res_tesoura_j2)
    _, altura_te2 = tesoura_j2.shape[::-1]
    if min_val_te2 < 0.030:
        cv2.putText(img_rgb, "Tesoura", (min_loc_te2[0] , min_loc_te2[1] + altura_te2 + 30), font,1,(255,0,0),1,cv2.LINE_AA)
        return "tesoura"


    #PEDRA
    res_pedra_j2 = cv2.matchTemplate(img_gray,pedra_j2,cv2.TM_SQDIFF_NORMED)
    min_val_pe2, _, min_loc_pe2, _ = cv2.minMaxLoc(res_pedra_j2)
    _, altura_pe2 = pedra_j2.shape[::-1]
    if min_val_pe2 < 0.011: 
        cv2.putText(img_rgb, "Pedra", (min_loc_pe2[0] , min_loc_pe2[1] + altura_pe2 + 30), font,1,(255,0,0),1,cv2.LINE_AA)
        return "pedra"
 


def image_da_webcam(img):
    global jogada_anterior_j1
    global jogada_anterior_j2
    global resultado
    
    #RECEBENDO O FRAME
    resized = ajustarImagem(img)
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    jogada1 = jogada_jogador1(img_gray, img_rgb)
    jogada2 = jogada_jogador2(img_gray, img_rgb)

 

    if jogada_anterior_j1 != jogada1 or jogada_anterior_j2 != jogada2 :
        jogada_anterior_j1 = jogada1
        jogada_anterior_j2 = jogada2
        resultado = jankenpo(jogada1, jogada2)
       

    #CAPTURA O RESULTADO DA PARTIDA
    cv2.putText(img_rgb, resultado, (100 , 100), font,1,(0,0,0),1,cv2.LINE_AA)

    cv2.putText(img_rgb, f"JOGADOR 1: {pontuação_j1}"  , (img_rgb.shape[1] - img_rgb.shape[1] + 10, img_rgb.shape[0] - 20), font,1,(0,150,0),1,cv2.LINE_AA)
    cv2.putText(img_rgb, f"JOGADOR 2: {pontuação_j2}" , (img_rgb.shape[1] - 300, img_rgb.shape[0] - 20), font,1,(255,0,0),1,cv2.LINE_AA)


    #EXIBE A PONTUAÇÃO     
    print(jogada_anterior_j1)
    print(jogada_anterior_j2)


    return img_rgb

   

cv2.namedWindow("preview")
# define a entrada de video para webcam
vc = cv2.VideoCapture("pedra-papel-tesoura.mp4")


#configura o tamanho da janela 
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    
    img = image_da_webcam(frame) # passa o frame para a função imagem_da_webcam e recebe em img imagem tratada
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("preview", img_rgb)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break



#EXIBE A PONTUAÇÃO FINAL
print("----FIM DE JOGO----")
print("Pontuação final")
print(f"Jogador 1 : {pontuação_j1}")
print(f"Jogador 2 : {pontuação_j2}")


cv2.destroyWindow("preview")
vc.release()
