import cv2
import keras
import numpy as np

# Model to be used for digit prediction.
model = keras.models.load_model("models/cnn_model.keras")

# Image size of the MNIST dataset (28x28).
mnist_image_size = 28

# Image size which the model has been trained. It should be a multiple of the
# MNIST image size.
model_image_size = 84


def crop_image_to_content(image: np.array) -> np.array:
    # Computes the (x,y) coordinates of the non-blank pixels (image content).
    non_blank_pixels = np.argwhere(image < 255)

    # Gets the coordinates of the 'bounding box' around the non-blank pixels.
    y1, x1 = np.min(non_blank_pixels, axis=0)
    y2, x2 = np.max(non_blank_pixels, axis=0)

    # Crops the image based on the 'bounding box'.
    cropped_image = image[y1:y2+1, x1:x2+1]
    return cropped_image


def pad_image_to_mnist_image_size_multiple(image: np.array) -> np.array:
    """
    Pads an array image to ensure its dimensions are a multiple of the MNIST
    image size.
    """
    source_height, source_width = image.shape[0], image.shape[1]

    # Computes the size for a square image (required by the model) and pads it
    # to match a multiple of the MNIST image size, if needed.
    target_size = max(source_height, source_width)
    target_size_remainder = (target_size % mnist_image_size)
    if target_size_remainder > 0:
        padding_size = mnist_image_size - target_size_remainder
        target_size += padding_size

    # Computes the target (padded) image heigth and width.
    target_height = target_size - source_height
    target_width = target_size - source_width

    # Computes the variables to place the source image in the center of the
    # target (padded) image.
    half_target_height = target_height // 2
    half_target_width = target_width // 2
    target_x1 = half_target_width
    target_x2 = half_target_width + source_width
    target_y1 = half_target_height
    target_y2 = half_target_height + source_height

    # Creates a new blank image with the target (padded) size.
    padded_image = np.ones((target_size, target_size), dtype=np.uint8) * 255

    # Places the original image in the center of the target (padded) image.
    padded_image[target_y1:target_y2, target_x1:target_x2] = image
    return padded_image


def compatibilize_image_with_model(image: np.array) -> np.array:
    # Crops the image to the content.
    cropped_image = crop_image_to_content(image)

    # Pads the image to match a MNIST image size multiple.
    padded_image = pad_image_to_mnist_image_size_multiple(cropped_image)

    # Resizes the image to match the model's image size.
    resized_image = cv2.resize(
        padded_image, (model_image_size, model_image_size), interpolation=cv2.INTER_AREA)

    # Inverts the image colors because MNIST dataset has inverted colors, i.e.,
    # a black background instead of a white one.
    inverted_image = cv2.bitwise_not(resized_image)
    return inverted_image


def normalize_image(image: np.array) -> np.array:
    """
    Converts the pixel intensities of the image into model weights.
    """
    normalized_image = image.astype(float) / 255.0
    return normalized_image


def prepare_image_for_prediction(image: np.array) -> np.array:
    compatible_image = compatibilize_image_with_model(image)
    normalized_image = normalize_image(compatible_image)

    # Reshapes the image to fit it into the model.
    reshaped_image = normalized_image.reshape(1, model_image_size, model_image_size, 1)
    return reshaped_image


def predict(digit_image):
    prepared_digit_image = prepare_image_for_prediction(digit_image)
    prediction_values = model.predict(prepared_digit_image, batch_size=1)
    prediction = str(np.argmax(prediction_values))
    return prediction
