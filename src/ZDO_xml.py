temata_vstup='zdo2024/annotations.xml'

f=open(temata_vstup,'r')
tema=f.read()
f.close()

rozdeleni=tema.split('</meta>')
obrazky=rozdeleni[1].split('</image>')
obrazky.pop()
image_list = {}

rez='Incision'
steh='label="Stitch"'
for i in obrazky:
    rozdel=i.split()
    nazev=rozdel[2].split('/')[1].replace('"','')
    try:
        ano_rez=i.index(rez)
        image_list[nazev]=len([a for a, x in enumerate(rozdel) if x == steh])
    except:
        image_list[nazev]=-1
    
print(image_list)

