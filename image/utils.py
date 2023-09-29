import matplotlib.pyplot as plt
import cv2


def show(image, figsize=(10, 10)):
    colored_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(colored_image)
    plt.axis('off')
    plt.show()


def draw_rects(image, rects):
    colored_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(colored_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return colored_image


def draw_contours(image, contours):
    colored_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.drawContours(colored_image, contours, -1, (0, 255, 0), 1)
    return colored_image
