from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import os


pathOut = './videos/play.mp4'
path_src = './images'

def save_images(html):

    # Setting Chrome driver
    driver_path = '/Users/users/Downloads/chromedriver'
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(driver_path, chrome_options = chrome_options)
    driver.implicitly_wait(3)
    driver.get(html)

    # Make image folder path (wanna delete)
    image_path = './images'
    os.makedirs(image_path, exist_ok=True)

    # Get timestep
    xpath_ts = '/html/body/div/div[4]/span'
    board_ts = driver.find_element(By.XPATH, xpath_ts)

    # Get images of every timestep
    xpath_bd = '/html/body/div'
    board_bd = driver.find_element(By.XPATH, xpath_bd)

    xpath_bt = '/html/body/div/div[4]/button'
    board_bt = driver.find_element(By.XPATH, xpath_bt)


    for i in range(20):
        board_bt.send_keys(Keys.LEFT)

    pre_t = -1
    _, max_t = board_ts.text.split('/')
    max_t = int(max_t.strip())

    board_bt.send_keys(Keys.SPACE)
    while pre_t != max_t:
        curr_t, _ = board_ts.text.split('/')
        curr_t = int(curr_t.strip())
        if pre_t != curr_t:
            board_bd.screenshot('./images/tmp_%03d.png' % curr_t)
            pre_t = curr_t

    driver.close()

def img2mp4(paths, pathOut, fps) :
    import cv2
    frame_array = []
    for idx , path in enumerate(paths) : 
        img = cv2.imread(path)
        height, width, layers = img.shape
        size = (width, height)
        frame_array.append(img)
        fourcc = cv2.VideoWriter_fourcc(*'FMP4')
        out = cv2.VideoWriter(pathOut, fourcc = fourcc, fps = fps, frameSize = size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


paths = sorted(os.listdir(path_src))
paths = [os.path.join(path_src, path) for path in paths]

filename = 'file://' + os.getcwd() + '/' + 'render_file.html'
img2mp4(paths, pathOut, fps=3)