from PIL import Image
import glob
import matplotlib.pyplot as plt
import os
import skimage
from skimage.color import rgb2gray 
import numpy as np
from skimage import feature, transform, filters, morphology
import scipy
import cv2
from skimage.feature import canny
from scipy import ndimage
import math
import sys
import json
import cv2
import random
import csv
from argparse import ArgumentParser
import argparse
from pathlib import Path

def boolean_to_rgb(boolean_image, object_color=[255, 255, 255], background_color=[0, 0, 0]):
    # Konvertuje boolean image do RGB
    
    object_color = np.array(object_color)
    background_color = np.array(background_color)
    
    # Prázdná schránka pro RGB obrázek
    rgb_image = np.zeros(boolean_image.shape + (3,), dtype=np.uint8)
    
    # Určení pixelu podle boolean obrazku
    rgb_image[boolean_image] = object_color
    rgb_image[~boolean_image] = background_color
    
    return rgb_image

def ZDO(output, *arg2):
    path ='zdo2024/images/incision_couples'
    temata_vstup='zdo2024/annotations.xml'
    
    # Zpracovani argumentu
    argumenty = []
    if(len(arg2) > 0):
        for i in arg2[0]:
            argumenty.append(i)
        vizualizace = False
        if argumenty[0] == '-v':
            vizualizace = True
            a = argumenty.pop(0)
    else:
        vizualizace = False

    # Pocet stehu podle anotaci
    # Nacteni anotaci
    f=open(temata_vstup,'r')
    tema=f.read()
    f.close()

    # Uprava anotaci
    rozdeleni=tema.split('</meta>')
    obrazky=rozdeleni[1].split('</image>')
    obrazky.pop()
    stitch_list = {}

    # Urceni poctu stehu podle anotaci
    rez='Incision'
    steh='label="Stitch"'
    for i in obrazky:
        rozdel=i.split()
        nazev=rozdel[2].split('/')[1].replace('"','')
        try:
            ano_rez=i.index(rez)
            stitch_list[nazev]=len([a for a, x in enumerate(rozdel) if x == steh])
        except:
            stitch_list[nazev]=-1
    # -----------------------------------------------------------------

    # Nacteni a ulozeni obrazku v odstínech šedi
    image_list = {}
    image_list_orig = {}
    nazvy_obrazku=[]
    pocet_stehu={}
    for f in os.listdir(path):
        obrazek=rgb2gray(np.array(Image.open(os.path.join(path,f))))
        obrazek_orig=np.array(Image.open(os.path.join(path,f)))
        image_list[f]=obrazek
        image_list_orig[f]=obrazek_orig
        nazvy_obrazku.append(f)
        pocet_stehu[f]=0  
    # -------------------------------------------------------------

    right_decision = 0
    obr_steh = {}
    
    # Velky cyklus pro nacteni a praci s jednotlivymi obrazky
    for i in nazvy_obrazku:
        # Nacteni obrazku
        obr = image_list[i]
        original = obr
        plt.subplot(341)
        plt.imshow(image_list_orig[i])
        plt.title(f"Originální obrázek")
        '''plt.show()'''
        plt.subplot(342)
        plt.imshow(obr,cmap='gray')
        plt.title(f"Obrázek v odstínech šedi")
        '''plt.show()'''
        
        
        # Odstraneni stinu
        # Rozdeleni na casti v pripade vicevrstveho obrazku
        rgb_planes = cv2.split(obr)

        # Uprava obrazku
        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 1)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            
        # Sjednoceni vrstev obrazku
        result = cv2.merge(result_planes)
        obr = result
        
        plt.subplot(343)
        plt.imshow(obr,cmap='gray')
        plt.title(f"Obrázek s odstraněnými stíny")
        '''plt.show()'''
        
        # Oddeleni pozadi obrazku podle Otsu filtru
        image_list[i]=obr < filters.threshold_otsu(obr)
        obr = image_list[i]
        plt.subplot(344)
        plt.imshow(obr,cmap='gray')
        plt.title(f"Obrázek oddělený na objekt a pozadí")
        '''plt.show()'''
        #------------------------------------------------------------------------------
        
        # Mazani rohu obrazku
        # 3 pixely z kazde strany krome leve, kde mazeme 5% z šířky obrazku
        leva_strana = math.floor((len(obr[0]) / 100) * 5)
        
        mazani_rohu = obr[1:3, :]
        mazani_rohu[True] = False
        obr[1:3, :] = mazani_rohu
        mazani_rohu = obr[(len(obr)-2):len(obr), :]
        mazani_rohu[True] = False
        obr[(len(obr)-2):len(obr), :] = mazani_rohu
        mazani_rohu = obr[:, 0:leva_strana]
        mazani_rohu[True] = False
        obr[:, 0:leva_strana] = mazani_rohu
        mazani_rohu = obr[:, (len(obr[0])-2):len(obr[0])]
        mazani_rohu[True] = False
        obr[:, (len(obr[0])-2):len(obr[0])] = mazani_rohu
        # -------------------------------------------------
        
        # Nalezeni rezu, eroze a dilatace o jinych jadrech
        kernel_big = np.ones((1,15), np.uint8)
        image_list[i] = skimage.morphology.binary_erosion(obr, kernel_big)
        obr_copy1 = image_list[i]
        kernel_big = np.ones((5,120), np.uint8)
        image_list[i] = skimage.morphology.binary_dilation(obr_copy1, kernel_big)
        obr_copy1 = image_list[i]
        
        # Nalezeni stehu, eroze a dilatace o jinych jadrech
        help_delka_vertical = len(obr) / 60
        kernel_big = np.ones((math.floor(19*help_delka_vertical),1), np.uint8)
        image_list[i] = skimage.morphology.binary_erosion(obr, kernel_big)
        obr_copy2 = image_list[i]
        kernel_big = np.ones((50,3), np.uint8)
        image_list[i] = skimage.morphology.binary_dilation(obr_copy2, kernel_big)
        obr_copy2 = image_list[i]
        
        # Sjednoceni rezu a stehu
        obr = obr_copy1 + obr_copy2
        
        plt.subplot(345)
        plt.imshow(obr,cmap='gray')
        plt.title(f"Řez a stehy pomocí morfologie")
        '''plt.show()'''
        # ------------------------------------------------------------
        
        # Skeletonizace obrazku
        image_list[i] = skimage.morphology.skeletonize(obr)
        obr = image_list[i]
        plt.subplot(346)
        plt.imshow(obr,cmap='gray')
        plt.title(f"Skeletonizovaný obrázek")
        '''plt.show()'''
        
        # Dilatace skeletonizovaného obrazku
        kernel_big = np.ones((3,3), np.uint8)
        image_list[i] = skimage.morphology.binary_dilation(obr, kernel_big)
        obr = image_list[i]
        plt.subplot(347)
        plt.imshow(obr,cmap='gray')
        plt.title(f"Dilatace skeletonizovaného obrázku")
        '''plt.show()'''
        
        # Mazani rohu obrazku pro pripad, kdyby predchozi dilatace narazila na kraj obrazku
        mazani_rohu = obr[1:3, :]
        mazani_rohu[True] = False
        obr[1:3, :] = mazani_rohu
        mazani_rohu = obr[(len(obr)-2):len(obr), :]
        mazani_rohu[True] = False
        obr[(len(obr)-2):len(obr), :] = mazani_rohu
        mazani_rohu = obr[:, 1:4]
        mazani_rohu[True] = False
        obr[:, 1:4] = mazani_rohu
        mazani_rohu = obr[:, (len(obr[0])-2):len(obr[0])]
        mazani_rohu[True] = False
        obr[:, (len(obr[0])-2):len(obr[0])] = mazani_rohu
        #--------------------------------------------------------------------------
        
        # Segmentace obrazku pomoci Canny
        image_list[i] = canny(obr)
        obr = image_list[i]
        plt.subplot(348)
        plt.imshow(obr,cmap='gray')
        plt.title(f"Segmentovaný obrázek")
        '''plt.show()'''
        #----------------------------------------------------------------------------
        
        # Prevedeni bool obrazku do rgb
        obr_rgb = boolean_to_rgb(obr)
        obr_rgb2 = boolean_to_rgb(obr)
        #--------------------------------------------------------------------------------
        
        # Tvorba Freemanova retezce
        # List moznych smeru
        directions = [0, 2, 4, 6, 1, 3, 5, 7]
        # Mozne posuny od prave prohledavaneho bodu
        dir_values = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
        # Slovnik smeru, kam se muzeme vydat z aktualniho bodu
        dir_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
        
        dir_rgb = [(255,0,0), (255,165,0), (255,255,0), (0,255,0), (0,0,255), (75,0,130), (148,0,211), (255,192,203)]
        
        # Zjisteni prvniho bodu pro Freemanuv retezec
        start_point = {}
        for l, row in enumerate(obr):
            for j, value in enumerate(row):
                # prvni pixel nejvice vlevo ale dale od kraje nez 9 pixelu
                if value == True and j > 9:
                    start_point[l] = j
                    break
                else:
                    start_point[l] = 1000
                    continue
        start_Freeman_chain_help = min(start_point, key=start_point.get)
        # Prvni bod pro start Freemanova kodu
        start_Freeman_chain = (start_Freeman_chain_help, start_point[start_Freeman_chain_help])
        
        # Freemanuv retezovy kod
        chain_code = []
        current_point = start_Freeman_chain
        current_dir = 0
        chain_code_point = []

        # Cyklus pro objeti rezu a stehu
        while True:
            Nalezen_soused = False
            for j in directions:
                next_dir = j
                dx, dy = dir_values[j]
                next_point = (current_point[0] + dx, current_point[1] + dy)
                # Pokud se nasledujici bod nachazi v obrazku, je True a nenachazi se jiz v retezci
                if (0 <= next_point[0] < obr.shape[0] and 0 <= next_point[1] < obr.shape[1] and obr[next_point] and next_point not in chain_code_point):
                    chain_code.append(dir_map[next_dir])
                    chain_code_point.append(next_point)
                    current_point = next_point
                    obr_rgb2[current_point] = dir_rgb[next_dir]
                    Nalezen_soused = True
                    break
            if Nalezen_soused == False:
                break
        # Vypsani pocatecniho bodu a pote Freemanova retezce
        #print('--------------')
        #print(start_Freeman_chain)
        #print(chain_code)
        
        plt.subplot(349)
        plt.imshow(obr_rgb2)
        plt.title(f"Obrys podle směrů Freemanova kódu")
        '''plt.show()'''
        
        # --------------------------------------------------------
        
        # Vyhodnoceni poctu stehu
        consecutive_number = 1
        prev_number = 10
        num_stitches = 0
        cycle = 0
        consecutive_number_list = []
        # Cyklus pro vyhodnoceni
        # Pokud jsou splneny podminky aktualniho zkoumaneho cisla a predchoziho zkoumaneho cisla, pricte se hodnota do consecutive_number
        # Pokud jiz nejsou splneny podminky, ale consecutive_number je alespon 5, program nasel nejspise rostouci hranu stehu, zapise ji a vykresli ji do rgb obrazku
        for j in chain_code:
            curr_number = j
            if (((curr_number == 2 or curr_number == 1) and (prev_number == 2 or prev_number == 1)) or (curr_number == 0 and prev_number == 2) or (curr_number == 2 and prev_number == 0) or (curr_number == 3 and prev_number == 0) or (curr_number == 0 and prev_number == 3) or (curr_number == 3 and prev_number == 2) or (curr_number == 2 and prev_number == 3) or (curr_number == 0 and prev_number == 2) or (curr_number == 2 and prev_number == 0) or (curr_number == 3 and prev_number == 0) or (curr_number == 0 and prev_number == 3)):
                consecutive_number = consecutive_number + 1
                consecutive_number_list.append(chain_code_point[cycle])
            else:
                if consecutive_number >= 5 and (prev_number == 2 or prev_number == 1 or prev_number == 0):
                    num_stitches = num_stitches + 1
                    for k in consecutive_number_list:
                        obr_rgb[k] = [255, 0, 0]
                consecutive_number = 1
                consecutive_number_list = []
            prev_number = curr_number
            cycle = cycle + 1
            
        hrany_nahoru = num_stitches
        # Vydeleni rostoucich hran stehu dvema. Predpokladame, ze kazdy steh ma dve rostouci hrany, jednu nad rezem a jednu pod rezem
        num_stitches = math.floor((num_stitches) / 2)
        # ------------------------------------------------------------
        
        # Neplatny obrazek, pokud je Freemanuv kod prilis kratky
        if len(chain_code) < 230:
            num_stitches = -1
        
        # Vypocet spravne vyhodnocenych obrazku
        if num_stitches == stitch_list[i]:
            right_decision = right_decision + 1
        
        # Vykresleni rgb zobrazeni hran, podle kterych se rozhoduje o poctu stehu
        plt.subplot(3,4,10)
        plt.imshow(obr_rgb)
        plt.title(f"Zbarvení hran určující počet stehů")
        '''plt.show()'''
        
        # Vypsani nazvu obrazku, poctu nalezenych rostoucich hran, urcenych stehu a poctu stehu podle anotace
        #print('Nazev obrazku: ', i)
        #print('Pocet hran nahoru: ', hrany_nahoru)
        #print('Pocet stehu: ', num_stitches)
        #print('Spravny pocet stehu: ', stitch_list[i])
    
        obr_steh[i] = num_stitches
        # Zobrazeni obrazku, ktery graficky ukazuje prubeh algoritmu. Pokud bylo v argumenty napsano -v, zobrazi ty obrazky, jejichz nazvy byly napsane v argumentu
        if vizualizace and i in argumenty:
            plt.suptitle(i)
            plt.show()
    # ------------------------------------------------------------------------------------

    # Po celem algoritmu vypiseme pocet rozhodnuti, ktere se shoduji s anotacemi
    # Ne vsechny anotace jsou spravne anotovany, viz napr. obrazek: SA_20220620-110036_vp424rn48961_incision_crop_0.jpg
    #print('--------------------')
    #print('Pocet správných rozhodnutí: ', right_decision, ' z ', len(stitch_list.keys()))
    #print('Procento správného rozhodnutí: ', (right_decision / len(stitch_list.keys())) * 100)
        
    print('Konec')

    # Program napsán ve Visual Studio Code
    
    with open(output, 'w', newline='') as csvfile:
            zapis = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            zapis.writerow(['filename ', ' n_stiches'])
    
            if len(argumenty) > 0:
                    for i in argumenty:
                        zapis.writerow([i+' ', ' '+str(obr_steh[i])])
            else:
                    for i in obr_steh:
                        zapis.writerow([i+' ', ' '+str(obr_steh[i])])
                        
  
  
def main():
    parser = ArgumentParser()
    parser.add_argument('csv_file')
    parser.add_argument('images', nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    
    csv_file = args.csv_file
    img = args.images

    if img:
        ZDO(csv_file,img)
    else:
        ZDO(csv_file)

if __name__ == '__main__':
    main()

#ZDO("output","-v", "SA_20220620-103348_3imoxskpwsvo_incision_crop_0.jpg", "SA_20230222-130540_1qil9lfd57zi_incision_crop_0.jpg")