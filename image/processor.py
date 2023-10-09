import os
import cv2
import numpy as np
import image.utils as image_utils


def resize_image_based_on_height(image, new_height):
    """
    Resizes the image based on a new height, computing the new width, in order
    to maintain the aspect ratio.
    """
    height, width = image.shape[:2]
    aspect_ratio = float(new_height) / height
    new_width = int(width * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


def prepare_image(image):
    # Removes some noise.
    smoothed_image = cv2.GaussianBlur(image, (3, 3), 0)

    # Converts the image to grayscale.
    gray_image = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2GRAY)

    # Creates a new binarized image.
    binarized_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 17)

    # Resizes the image to keep it in a moderately 'known' domain.
    new_height = 2500
    binarized_image_resized = resize_image_based_on_height(binarized_image, new_height)
    gray_image_resized = resize_image_based_on_height(gray_image, new_height)

    return binarized_image_resized, gray_image_resized


def clear_irrelevant_image_regions(image):
    """
    Blanks irrelevant regions of the image, such as logos and signatures areas;
    also removing the 'page edges' that may arise during scanning.
    """
    # Computes the number of rows and columns to blank:
    # - ~18% of the top and bottom.
    # - ~5% of the left and right sides.
    height, width = image.shape[0:2]
    num_rows_to_blank = int(0.18 * height)
    num_columns_to_blank = int(0.05 * width)

    # Creates a black mask with the same size as the image.
    mask = np.zeros_like(image)

    # Blanks the top and bottom sides of the mask.
    mask[:num_rows_to_blank, :] = 255
    mask[-num_rows_to_blank:, :] = 255

    # Blanks the left and rigth sides of the mask.
    mask[:, :num_columns_to_blank] = 255
    mask[:, -num_columns_to_blank:] = 255

    # Applies the mask to the image, by using the 'or' operator.
    masked_image = cv2.bitwise_or(image, mask)

    return masked_image


def find_rects_in_image(image):
    """
    Returns:
        adjusted_rects: rectangles drawn around the detected contours.
        contour_rects: original detected contours.
    """
    # Finds contours in the image.
    edges_image = cv2.Canny(image, 100, 150)
    contours, _ = cv2.findContours(edges_image.copy(),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Draws rectangles around the detected contours. To avoid mistakenly
    # detecting areas between characters, filters out rectangles whose area is
    # less than 5% of the total image area. This approach avoids fixed-size
    # criteria (e.g., w > 100 and h > 100), being agnostic about image size.
    area_threshold = 0.05
    adjusted_rects = []
    contour_rects = []
    image_area = image.shape[0] * image.shape[1]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if ((w*h)/image_area) >= area_threshold:
            contour_rects.append(contour)
            adjusted_rects.append((x, y, w, h))
    return adjusted_rects, contour_rects


def get_tentative_grades_rect(rects):
    """
    Gets the 'tentative' grades rectangle.
    """
    # Obtains the three rectangles with the maximum areas. In theory, each of
    # those rectangles, corresponds, respectively to the 'students', 'grades'
    # and 'grades in letter form' rectangles.
    rects.sort(key=lambda rect: rect[2] * rect[3], reverse=True)
    rects = rects[:3]

    # Now that there are three rectangles, selects the smallest one (which
    # should correspond to the 'grades' boundary).
    (x, y, w, h) = rects[2]
    return (x, y, w, h)


def change_image_orientation_if_required(image, grades_rect):
    """
    Note: this function assumes that the image is in vertical form.
    """
    # Computes the x-coordinate for the middle of the image.
    middle_x_image = image.shape[1] // 2

    # Computes the x-coordinate for the middle of the grades rectangle.
    middle_x_grades_rect = (grades_rect[0] + grades_rect[3]) // 2

    # If the middle of the grades rectangle is in the left side of the image...
    if (middle_x_grades_rect < middle_x_image):
        # Rotates the image.
        return cv2.rotate(image, cv2.ROTATE_180), True
    return image, False


def align_image_if_required(image, rect_contour):
    # Computes the minimum rectangle of the specified rect contour.
    rect = cv2.minAreaRect(rect_contour)

    # Gets the angle of the rectangle.
    angle = rect[-1]

    # Computes the center of the image.
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    # IMPORTANT! Given that the rectangle must be aligned with respect to the
    # image, is required to compute a rotation matrix based on the angle of the
    # rectangle (which is computed by 'minAreaRect'). However, when considering
    # 'y1' and 'y2' as the coordinates located at left of the x-axis, and 'y3'
    # and 'y4' the coordinates located at the right side of the x-axis, there
    # are two cases (note: graphing a rectangle for each case is a more visual
    # aid!):
    # 1. y1 < y2: in this scenario, the rectangle is initially oriented to the
    # left, so the original angle was computed based on the segment (y2, y4).
    # Then, the alignment must be performed to the right, so the rotation angle
    # must be computed as 'angle - 90'.
    # 2. y1 > y2: in this case, the rectangle is currently oriented to the
    # right, so the original angle was computed based on the segment (y1, y3).
    # So, the alignment must be performed conforming the original angle of the
    # rectangle.

    # In order to get 'y1' and 'y2', it is required to obtain the vertices,
    # sorting it by their 'x' values.
    vertices = cv2.boxPoints(rect)
    vertices = sorted(vertices, key=lambda point: point[0])
    y1, y2 = vertices[0][1], vertices[1][1]

    rotation_angle = 0
    if (y1 < y2):
        # Case 1: moves to left.
        rotation_angle = angle - 90
    else:
        # Case 2: moves to right.
        rotation_angle = angle

    if rotation_angle:
        # Aligns the image.
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
        image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return image, bool(rotation_angle)


def crop_grades_image_to_content(image):
    # Removes some noise.
    aux_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Applies binary thresholding to improve cropping based on white pixels.
    aux_image = cv2.adaptiveThreshold(
        aux_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 19)

    # Gets the coordinates of the 'bounding box' the around non-blank pixels.
    non_blank_pixels = np.argwhere(image < 255)
    y1, x1 = np.min(non_blank_pixels, axis=0)
    y2, x2 = np.max(non_blank_pixels, axis=0)

    # Crops the original image.
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image


def extract_grades_image(image, grades_rect):
    """
    Extracts the grades image, removing borders and keeping digits only.
    """
    # Ensures the image width is 2500 pixels, which is useful for subsequent
    # statements.
    assert (image.shape[0] == 2500)

    # Defines a margin slack for the grades image extraction. This tries to
    # prevent potential over-adjustments given the specified rectangle and
    # ensures that the extracted table maintains consistent border sizes, which
    # is useful when removing them.
    margin_slack = 5

    # Computes the coordinates for the region of interest.
    x, y, w, h = grades_rect
    x1 = x - margin_slack
    x2 = x + w + margin_slack
    y1 = y - margin_slack
    y2 = y + h + margin_slack

    # Extracts the grades image.
    grades_image = image[y1:y2, x1:x2]

    # Defines a normalized height for the grades image. '1005' is chosen
    # because it's a multiple of 15, aligning with the number of rows,
    # simplifying the computation of integer results for each cell rectangle.
    normalized_heigth = 1005

    # Resizes the grades table.
    resized_image = resize_image_based_on_height(
        grades_image, normalized_heigth)

    # Crops the image to content.
    # cropped_image = crop_grades_image_to_content(resized_image)

    # Note: it attempted to remove borders using white dilation, but, given the
    # 'known domain,' it is more effective to remove them based on their size.
    # kernel = np.ones((6, 6), np.uint8)
    # dilated = cv2.dilate(binarized_image, kernel)

    # Now, the borders can be removed based on a fixed size. Note that this
    # border size has been optimized for a height of 1005 pixels. The 'assert'
    # statement must be changed when modifying the last one.
    assert (normalized_heigth == 1005)
    border_size = 18
    resized_image = resized_image[border_size:-
                                  border_size, border_size:-border_size]

    # Resizes the image again, and recomputes the grades rectangle.
    resized_image = resize_image_based_on_height(resized_image, normalized_heigth)
    y, x = resized_image.shape[0:2]
    new_grades_rect = (0, 0, (x), (y))
    return resized_image, new_grades_rect


def compute_digit_cells_rects(grades_rect):
    students_per_rect_count = 15
    cells_per_student_count = 2

    rect_x, rect_y, rect_w, rect_h = grades_rect
    cell_width = rect_w // cells_per_student_count
    cell_height = rect_h // students_per_rect_count

    curr_y_position = rect_y

    digit_rects = []
    for _ in range(0, students_per_rect_count):
        digit1_cell_rect = (rect_x, curr_y_position, cell_width, cell_height)
        digit2_cell_rect = (rect_x + cell_width,
                            curr_y_position, cell_width, cell_height)

        digit_rects.append(digit1_cell_rect)
        digit_rects.append(digit2_cell_rect)

        # Moves to the next row.
        curr_y_position += cell_height

    return digit_rects


def extract_content_from_cell(image, cell_rect):
    # 1. Extracts the cell region.
    x, y, width, height = cell_rect
    cell = image[y:y+height, x:x+width]

    # 2. Uses an auxiliar version of the cell to get the elements contours.
    aux_cell = cv2.GaussianBlur(cell, (5, 5), 0)

    # WARNING: counter-intuitive. Erodes the black elements of the image, by
    # using a kernel that 'dilates' the white color. This tries to remove some
    # noise.
    kernel = np.ones((2, 2), np.uint8)
    aux_cell = cv2.dilate(aux_cell, kernel)

    # WARNING: counter-intuitive. Dilates the black elements (presumably
    # digits) of the binarized image, by using a kernel that 'erodes' the white
    # color. This attemps to be more easy to extract the digits.
    kernel = np.ones((3, 3), np.uint8)
    aux_cell = cv2.erode(aux_cell, kernel)

    # Finds contours in the auxiliar cell, inverting its colors (the 'find
    # contours' method works better with a black background rather than a white
    # one).
    inverted_color_cell = cv2.bitwise_not(aux_cell)
    contours, _ = cv2.findContours(inverted_color_cell.copy(),
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # 3. Creates a rectangle for each contour, keeping those that 'probably'
    # contains a digit. To avoid incorrect objects, filters out rectangles
    # whose area is less than 5% and greater than 60% of the total cell area.
    # The upper bound is because, even though the cell borders are cleaned up,
    # if there are 'remnants' of them, they can be detected as contours.
    rects = []
    cell_area = cell.shape[0] * cell.shape[1]

    digit_upper_threshold = 0.60
    digit_lower_threshold = 0.05
    for contour in contours:
        bounding_box = cv2.boundingRect(contour)
        x, y, w, h = bounding_box
        ratio = (w*h)/cell_area
        if (ratio >= digit_lower_threshold and ratio <= digit_upper_threshold):
            rects.append(bounding_box)

    if len(rects) == 0:
        return None

    # 4. Prepares the digit for its extraction.

    # Dilates the content of the cell, by using a kernel that 'erodes' the
    # white color. Then, creates an equalized version of the image, adjusting
    # its contrast (alpha) and brightness (beta).
    equalized_cell = cv2.convertScaleAbs(cell, alpha=1, beta=-75)
    aux_cell = equalized_cell

    # Generates a mask that extracts the cell's content. Note that, at this
    # step, the background is black. In this case, binarizing the image could
    # also be an option, but the idea is to preserve the original pixel
    # intensity of the content.
    mask = cv2.adaptiveThreshold(
        aux_cell, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 1)
    inverted_mask = cv2.bitwise_not(mask)
    cell = cv2.bitwise_or(equalized_cell, equalized_cell, mask=inverted_mask)

    # Replaces the black background with a white one.
    white_background = np.zeros_like(cell, dtype=np.uint8)
    white_background.fill(255)
    cell += cv2.bitwise_and(white_background, white_background, mask=mask)

    # Dilates the black elements, by using a kernel that 'erodes' the white
    # color.
    kernel = np.ones((3, 3), np.uint8)
    dilated_cell = cv2.erode(cell, kernel)

    # 5. Extracts the digit.

    # Keeps the 'largest' rectangle among the detected ones.
    rects.sort(key=lambda rect: rect[2] * rect[3], reverse=True)
    max_rectangle = rects[0]

    x, y, w, h = max_rectangle
    content = dilated_cell[y:y+h, x:x+w]
    return content


def extract_cell_contents(image, cells_rects):
    """
    Returns a tuple of (Image, Image). If there is not content, then the Image
    object will be None.
    """
    cell_contents = []
    for cell_rect in cells_rects:
        cell_content = extract_content_from_cell(image, cell_rect)
        cell_contents.append(cell_content)

    return cell_contents


# Main function.
def extract_digits_images(image):
    # 1. Performs image preprocessing.
    resized_binarized_image, resized_gray_image = prepare_image(image.copy())
    cv2.imwrite("/tmp/01-preprocessed_image.png", resized_gray_image)

    # 2. Clear irrelevant regions (such as logo and signatures regions).
    cleared_binarized_image = clear_irrelevant_image_regions(resized_binarized_image.copy())
    cleared_gray_image = clear_irrelevant_image_regions(resized_gray_image.copy())
    cv2.imwrite("/tmp/02-cleared_image.png", cleared_gray_image)

    # 3. Finds relevant rects.
    image_rects, contour_rects = find_rects_in_image(resized_binarized_image.copy())
    grades_rect = get_tentative_grades_rect(image_rects)

    # 4. Changes, if required and based in the grades rect, the image
    # orientation.
    reoriented_binarized_image, rotated = change_image_orientation_if_required(
        cleared_binarized_image.copy(), grades_rect)
    reoriented_gray_image, rotated = change_image_orientation_if_required(
        cleared_gray_image.copy(), grades_rect)

    # If the image has been rotated, recomputes the rects.
    if rotated:
        image_rects, contour_rects = find_rects_in_image(
            reoriented_binarized_image.copy())
        grades_rect = get_tentative_grades_rect(image_rects)
        cv2.imwrite("/tmp/03-reoriented_image.png", reoriented_gray_image)

    # 5. Aligns the image, if required.
    aligned_gray_image, aligned = align_image_if_required(
        reoriented_gray_image.copy(), contour_rects[0])
    aligned_binarized_image, aligned = align_image_if_required(
        reoriented_binarized_image.copy(), contour_rects[0])

    # If the image has been aligned, recomputes the rects.
    if aligned:
        image_rects, contour_rects = find_rects_in_image(aligned_binarized_image.copy())
        grades_rect = get_tentative_grades_rect(image_rects)
        cv2.imwrite("/tmp/04-aligned_image.png", aligned_gray_image)

    # 6. Extracts the grades image, based on the grades rect. Also, computes
    # the new grades rect, based on this new image.
    grades_image, new_grades_rect = extract_grades_image(
        aligned_gray_image.copy(), grades_rect)

    grades_image_binarized = cv2.threshold(grades_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((3, 3), np.uint8)
    grades_image_binarized = cv2.erode(grades_image_binarized, kernel)

    grades_rect = new_grades_rect

    # Computes cells regions containing the digits.
    cells_rects = compute_digit_cells_rects(grades_rect)

    cell_rects_image = image_utils.draw_rects(grades_image, cells_rects)
    cv2.imwrite("/tmp/05-grades_image.png", cell_rects_image)

    # 7. Transforms the 'non-pure white' background into white, while
    # preserving the original pixel intensity of the content. Note: 'non-pure
    # white' refers to the gray or yellowing of the image produced by the
    # scanning process.

    # Uses the binarized grades image to keep only the 'significant' elements
    # of the image. Note that, at this step, the background is black.
    grades_image_binarized_not = cv2.bitwise_not(grades_image_binarized)
    processed_grades_image = cv2.bitwise_or(grades_image, grades_image, mask=grades_image_binarized_not)

    # Replaces the black background of the grades image with a white one.
    white_background = np.zeros_like(grades_image, dtype=np.uint8)
    white_background.fill(255)
    processed_grades_image += cv2.bitwise_and(white_background, white_background, mask=grades_image_binarized)
    cv2.imwrite("/tmp/06-processed_grades_image.png", processed_grades_image)

    # 8. Extracts the digits from the cells.
    cell_contents = extract_cell_contents(processed_grades_image.copy(), cells_rects)

    # Saves the extracted digits as images, for debugging purposes.
    digits_output_dir = "/tmp/digits"
    os.makedirs(digits_output_dir, exist_ok=True)
    digit_pairs = zip(cell_contents[::2], cell_contents[1::2])
    for curr_pair, (cell1_content, cell2_content) in enumerate(digit_pairs):
        if cell1_content is not None:
            name = os.path.join(digits_output_dir, f"{curr_pair}-0.png")
            cv2.imwrite(name, cell1_content)

        if cell2_content is not None:
            name = os.path.join(digits_output_dir, f"{curr_pair}-1.png")
            cv2.imwrite(name, cell2_content)

    return cell_contents
