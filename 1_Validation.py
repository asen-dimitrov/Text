#!/usr/bin/env python
# coding: utf-8

# In[2]:


def val(a):
    a=str(a)
    a=a.strip()
    if not a.isdigit():
        return "greshen"
    else:
        l=len(str(a))
        if l == 10:
            result = val_egn(a)
        elif l==9:
            result = val_eik9(a)
        elif l==13:
            result = val_eik13(a)
        else: 
            return "greshen"
    return result


# In[3]:


def val_egn(a):
    tegla = [2, 4, 8, 5, 10, 9, 7, 3, 6]
    sumata = 0
    for i in range(len(str(a))-1):
        sumata+=(int(a[i])*int(tegla[i]))
        
    if sumata%11<10:
        kc=sumata%11
    elif sumata%11==10:
        kc=0
        
    if int(a[9])==kc:
        result="EGN"
    else:
        result="greshen"
    return result    


# In[4]:


def val_eik9(a):
    tegla = [1, 2, 3, 4, 5, 6, 7, 8]
    tegla2= [3, 4, 5, 6, 7, 8, 9, 10]
    sumata = 0
    for i in range(len(str(a))-1):
        sumata+=(int(a[i])*int(tegla[i]))
        
    if sumata%11<10:
        kc=sumata%11
    elif sumata%11==10:
        sumata=0
        for i in range(len(str(a))-1):
            sumata+=(int(a[i])*int(tegla2[i]))
        if sumata%11<10:
            kc=sumata%11
        elif sumata%11==10:
            kc=0
            
    if int(a[8])==kc:
        result="EIK9"
    else:
        result="greshen"
    return result    


# In[5]:


def val_eik13(a):
    tegla3= [2,7,3,5]
    tegla4= [4,9,5,7]
    sumata = 0
    
    if val_eik9(a[0:9])=="greshen":
        return "greshen"
    else:
        for i in range(9,13):
            sumata+=(int(a[i])*int(tegla3[i-9]))
        if sumata%11<10:
            kc=sumata%11
        elif sumata%11==10:
            sumata = 0
            for i in range(9,13):
                sumata+=(int(a[i])*int(tegla4[i-9]))
            if sumata%11<10:
                kc=sumata%11
            elif sumata%11==10:
                kc=0
    if int(a[8])==kc:
        result="EIK13"
    else:
        result="greshen" 
    return result    


# In[6]:


def ext_egn(a):
    if val(a) != "EGN":
        return "greshen"
    
    #####
    if int(a[8])%2:
        pol="жена"
    else:
        pol="мъж"
            
    #####
    godina, mesec, den =a[0:2], a[2:4], a[4:6]
    if int(mesec) > 40:
        godina = int(godina) + 2000
        mesec = int(mesec) - 40
    elif int(mesec) > 20:
        godina = int(godina) + 1800
        mesec = int(mesec) - 20
    else:
        godina = int(godina) + 1900
    rden = str(den)+"."+str(mesec)+"."+str(godina)  
    
    
    ####
    result = pol + " " + rden
    return result


# In[7]:


def ext_egnsreg(a):
    if val(a) != "EGN":
        return "greshen"
    
    #####
    if int(a[8])%2:
        pol="жена"
    else:
        pol="мъж"
            
    #####
    godina, mesec, den =a[0:2], a[2:4], a[4:6]
    if int(mesec) > 40:
        godina = int(godina) + 2000
        mesec = int(mesec) - 40
    elif int(mesec) > 20:
        godina = int(godina) + 1800
        mesec = int(mesec) - 20
    else:
        godina = int(godina) + 1900
    rden = str(den)+"."+str(mesec)+"."+str(godina)  
    
    ####
    ereg = [43,93,139,169,183,217,233,281,301,319,341,377,395,435,501,527,555,575,601,623,721,751,789,821,843,871,903,925,999]
    reggr = ["Благоевград", "Бургас", "Варна", "Велико Търново", "Видин", "Враца", "Габрово", "Кърджали", "Кюстендил", "Ловеч", "Монтана", "Пазарджик", "Перник", "Плевен", "Пловдив", "Разград", "Русе", "Силистра", "Сливен", "Смолян", "София - град", "София - окръг", "Стара Загора", "Добрич (Толбухин)", "Търговище", "Хасково", "Шумен", "Ямбол", "Друг/Неизвестен" ]
    for e in ereg:
        if int(a[6:9]) <= e:
            region = reggr[ereg.index(e)]
            break
            
    ####
    result = pol + " " + rden + " " + region
    return result


# In[ ]:




