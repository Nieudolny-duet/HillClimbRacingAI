import pyautogui
import time
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
#stałe
NUM_OF_Cars = 34
NUM_OF_Maps = 32


def Go_To_Maps():
    """
    Funkcja nawigująca do sekcji Map w aplikacji poprzez naciśnięcia klawiszy 'down', 'right' i 'enter'.
    """
    hold_key('down',1)
    hold_key('left',1)
    hold_key('enter',1)


def press(key,sleeps):
    pyautogui.keyDown(key)
    time.sleep(sleeps)  # Krótkie przytrzymanie klawisza
    pyautogui.keyUp(key)

def hold_key(key, times, delay=0.6):
    """
    Kliknij określony klawisz określoną liczbę razy z ustalonym opóźnieniem między kliknięciami.

    :param key: Klawisz do kliknięcia (np. 'left' dla strzałki w lewo)
    :param times: Liczba kliknięć
    :param delay: Czas opóźnienia między kliknięciami (w sekundach)
    """
    for _ in range(times):
        press(key, 0.1)
        time.sleep(delay)

# Użycie funkcji hold_key do naciskania strzałki w lewo 34 razy


def find_points(matrix, structure, step, widht, height, len_of_structure):
    matches = []
    #check first structure 
    prev = 0
    for y in range(height,len_of_structure,-1):
        if (matrix[(y-len_of_structure):y,0,:] == structure).all():
            matches.append((0,y))    
    return matches

def find_points_in_highway(matrix):
    indexes = np.argwhere(matrix[:,:,1] == 0)
    indexes_sorted_by_Y = indexes[np.argsort(indexes[:, 1])]
    return indexes_sorted_by_Y


def Get_List_Of_highway(list_of_points, matrix ,width, height ,step):
    fkfk = 4
    idx = 0
    pointList = []
    for i in range(0,width,step):
        for a in range(idx,len(list_of_points)-1):
            current = list_of_points[a]
            if current[1] == i:
                if current[0] < height-1 and is_gray_color(matrix[current[0]+1,current[1],:]):
                    pointList.append(current)
                    idx = a
                    break
                if current[0] < height-1:
                    idx = a
                    break
            if list_of_points[a][1] > i:
                idx = a
                break
    return np.array(pointList)
            

def is_gray_color(array):
    return 255 > array[0] == array[1] == array[2] > 0 
                

    
        


def isSimilar(current, prev, significant):
    # Dodanie małej wartości epsilon, aby uniknąć dzielenia przez zero
    epsilon = 1e-10
    
    # Obliczenie różnicy względnej tam, gdzie wartości są znaczące
    relative_diff = np.abs((current - prev) / (np.abs(prev) + epsilon))
    
    # Użyj bezpośredniego porównania różnic, gdy wartości są bardzo małe
    absolute_diff = np.abs(current - prev)
    
    # Warunek podobieństwa: względna różnica mniejsza niż próg lub absolutna różnica mniejsza niż epsilon
    return np.all((relative_diff < significant) | (absolute_diff < epsilon))




def checkFuel(imageMatrix, row=22, col1=129, col2=406, significant=0.05):
    a = 0
    for column in range(col1+1,col2):
        curr_submatrix = imageMatrix[row, column, :]
        pre_submatrix = imageMatrix[row, column-1, :]
        
        if not isSimilar(curr_submatrix,pre_submatrix,significant):
            break
        a += 1
    return a/len(imageMatrix[row,col1:col2,1])




# Główna funkcja
def main():

    for i in range(200,217,1):
        img_path = f'C:/Users/zapar/Python/HillClimbRacing/screenshot_{i}.png'
        image = cv2.imread(img_path)
        image1_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(checkFuel(image1_rgb))
        image = cv2.resize(image1_rgb, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(f'C:/Users/zapar/Python/HillClimbRacing/KOT{i}.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))



    structure = np.array([[ [0, 0, 0],]])

#main()

for _ in range(200, 217, 1):
    img = f'C:/Users/zapar/Python/HillClimbRacing/Kot{_}.png'
    image = cv2.imread(img)
    if image is None:
        print(f"Image at {img} not found.")
        continue

    image1_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    potential_point = find_points_in_highway(image1_rgb)
    listofpoints = Get_List_Of_highway(potential_point, image1_rgb, image1_rgb.shape[1], image1_rgb.shape[0], 3)

    # Tworzenie przykładowej tablicy NumPy (np. macierzy 108x192x3 wypełnionej wartościami 1)
    array = np.ones((image1_rgb.shape[0], image1_rgb.shape[1], 3))

    for g in listofpoints:
        array[g[0], g[1], :] = np.array([0, 0, 0])

    # Stworzenie rysunku z tablicy NumPy i oryginalnego obrazu
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(array, interpolation='none')
    axs[0].set_title('Przetworzona tablica NumPy w RGB')

    axs[1].imshow(image1_rgb)
    axs[1].set_title('Oryginalny obraz RGB')

    plt.show()




#kekw = cv2.resize(image1_rgb, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)


#np.savetxt("C:/Users/zapar/Python/HillClimbRacing/Autostrada1.csv",image1_rgb[:,:,0])
#np.savetxt("C:/Users/zapar/Python/HillClimbRacing/Autostrada2.csv",image1_rgb[:,:,1])
#np.savetxt("C:/Users/zapar/Python/HillClimbRacing/Autostrada3.csv",image1_rgb[:,:,2])


"""
time.sleep(3)
press('left',5)
position = random.randint(1,NUM_OF_Cars)
print(position)
hold_key('right', position, 0.2)
Go_To_Maps()
press('left',5)
position = random.randint(1,NUM_OF_Maps)
print(position)
hold_key('right',position,0.2)
"""

