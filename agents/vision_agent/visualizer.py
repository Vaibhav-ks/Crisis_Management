
import cv2

def draw_zone_grid(image, zone_map, save_path, grid_size=10):
    """
    Draw grid with zone IDs on image and save it.
    """

    img = image.copy()
    h, w = img.shape[:2]

    cell_w = w // grid_size
    cell_h = h // grid_size

    for gy in range(grid_size):
        for gx in range(grid_size):

            x1 = gx * cell_w
            y1 = gy * cell_h
            x2 = (gx + 1) * cell_w
            y2 = (gy + 1) * cell_h

            zone_id = f"Z{gy}{gx}"

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # Put zone label
            cv2.putText(
                img,
                zone_id,
                (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )

    # Convert back to BGR for saving (since OpenCV uses BGR)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, img_bgr)

    print(f"[Visualizer] Grid image saved at: {save_path}")