import cv2
import numpy as np
import time

# Global variable for exponential color smoothing
smoothed_bgr = None
ALPHA = 0.15  # Smoothing factor (0.0 to 1.0)

def classify_skin_tone_and_palette(mean_bgr):
    """Classifies tone name and returns 3 complementary styling colors."""
    b, g, r = mean_bgr
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    if luminance > 175:
        tone_name = "Fair / Light"
        palette = [(93, 54, 27), (64, 77, 0), (96, 164, 244)]      # Navy, Emerald, Soft Peach
    elif luminance > 145:
        tone_name = "Warm Light"
        palette = [(140, 100, 60), (45, 110, 80), (100, 180, 240)] # Terracotta, Olive, Coral
    elif luminance > 115:
        tone_name = "Medium / Olive"
        palette = [(0, 85, 204), (128, 128, 0), (78, 14, 74)]       # Burnt Orange, Teal, Plum
    elif luminance > 85:
        tone_name = "Tan / Rich Honey"
        palette = [(171, 71, 0), (1, 173, 225), (49, 60, 224)]     # Cobalt, Mustard, Coral Red
    else:
        tone_name = "Deep / Dark"
        palette = [(230, 230, 250), (200, 191, 143), (180, 105, 255)] # Lavender, Pale Gold, Hot Pink

    return tone_name, int(luminance), palette

def draw_color_palette_card(img, x, y, w, h, mean_bgr, tone_name, palette):
    b, g, r = int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])
    hex_code = f"#{r:02X}{g:02X}{b:02X}"

    card_w, card_h = 240, 140
    card_x = x + w + 15
    card_y = y

    if card_x + card_w > img.shape[1]:
        card_x = max(10, x - card_w - 15)
    if card_y + card_h > img.shape[0]:
        card_y = max(10, img.shape[0] - card_h - 10)

    # Semi-transparent background card
    overlay = img.copy()
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (card_x, card_y), (card_x + card_w, card_y + card_h), (80, 80, 80), 1)

    # 1. Main Swatch Box
    swatch_x, swatch_y = card_x + 10, card_y + 12
    cv2.rectangle(img, (swatch_x, swatch_y), (swatch_x + 45, swatch_y + 45), (b, g, r), -1)
    cv2.rectangle(img, (swatch_x, swatch_y), (swatch_x + 45, swatch_y + 45), (255, 255, 255), 1)

    # 2. Details
    text_x = swatch_x + 55
    cv2.putText(img, tone_name, (text_x, card_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"HEX: {hex_code}", (text_x, card_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"R:{r} G:{g} B:{b}", (text_x, card_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    # 3. Recommended Complementary Palette
    cv2.putText(img, "Suggested Colors:", (card_x + 10, card_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    for idx, col in enumerate(palette):
        px = card_x + 10 + (idx * 40)
        py = card_y + 92
        cv2.rectangle(img, (px, py), (px + 32, py + 22), col, -1)
        cv2.rectangle(img, (px, py), (px + 32, py + 22), (255, 255, 255), 1)

def run_skin_detector():
    global smoothed_bgr
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    prev_time = time.time()
    print("System active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Dual color space thresholding
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([25, 255, 255]))

        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))

        skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_frame = frame.copy()

        # Find the largest skin area
        valid_contours = [c for c in contours if cv2.contourArea(c) > 3500]

        if valid_contours:
            largest_cnt = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_cnt)

            contour_mask = np.zeros(skin_mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [largest_cnt], -1, 255, -1)
            raw_bgr = np.array(cv2.mean(frame, mask=contour_mask)[:3])

            # Apply Temporal Smoothing (EMA)
            if smoothed_bgr is None:
                smoothed_bgr = raw_bgr
            else:
                smoothed_bgr = ALPHA * raw_bgr + (1 - ALPHA) * smoothed_bgr

            tone_name, _, palette = classify_skin_tone_and_palette(smoothed_bgr)

            # Draw visual feedback
            cv2.drawContours(output_frame, [largest_cnt], -1, (0, 255, 0), 2)
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            draw_color_palette_card(output_frame, x, y, w, h, smoothed_bgr, tone_name, palette)

        # Calculate and display FPS
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time
        cv2.putText(output_frame, f"FPS: {fps}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.imshow("Skin Tone Region & Analyzer", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_skin_detector()
