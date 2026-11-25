
import cv2
import numpy as np


def segment_digits(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image could not be loaded")

    original = image.copy()

    # 1. Grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # 2. Blur
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # 4. Remove small noise
    kernel = np.ones((2, 2), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    # 5. Find initial contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_boxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area < 25 or w < 5 or h < 10:
            continue

        valid_boxes.append((x, y, w, h))

    if not valid_boxes:
        raise ValueError("No handwriting detected")

    # 6. Automatic cropping
    x_min = min(b[0] for b in valid_boxes)
    y_min = min(b[1] for b in valid_boxes)

    x_max = max(b[0] + b[2] for b in valid_boxes)
    y_max = max(b[1] + b[3] for b in valid_boxes)

    padding = 15

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)

    x_max = min(binary.shape[1], x_max + padding)
    y_max = min(binary.shape[0], y_max + padding)

    cropped_binary = binary[
        y_min:y_max,
        x_min:x_max
    ]

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(20, binary.shape[1] // 15), 1)
    )

    horizontal_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    # Remove detected horizontal lines
    binary = cv2.subtract(
        binary,
        horizontal_lines
    )

    # 7. Find individual digits
    contours, _ = cv2.findContours(
        cropped_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area < 10 or w < 3 or h < 5:
            continue

        boxes.append((x, y, w, h))

    boxes.sort(key=lambda box: box[0])

    processed_digits = []

    output = original.copy()

    for x, y, w, h in boxes:

        digit = cropped_binary[y:y+h, x:x+w]

        side = max(w, h) + 8

        canvas = np.zeros(
            (side, side),
            dtype=np.uint8
        )

        x_offset = (side - w) // 2
        y_offset = (side - h) // 2

        canvas[
            y_offset:y_offset+h,
            x_offset:x_offset+w
        ] = digit

        resized = cv2.resize(
            canvas,
            (20, 20),
            interpolation=cv2.INTER_AREA
        )

        final_digit = np.zeros(
            (28, 28),
            dtype=np.uint8
        )

        final_digit[4:24, 4:24] = resized

        processed_digits.append(final_digit)

        # Draw boxes using original image coordinates
        cv2.rectangle(
            output,
            (x + x_min, y + y_min),
            (x + x_min + w, y + y_min + h),
            (0, 255, 0),
            2
        )

    return processed_digits, output
if __name__ == "__main__":

    print("Starting segmentation test...")

    digits, result = segment_digits(
        "handwriting.png"
    )

    print("Detected digits:", len(digits))

    cv2.imshow("Detected Digits", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Test completed!")
