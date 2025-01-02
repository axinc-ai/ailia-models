from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_heatmap(image, heatmap, bbox=None, inout_score=None):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(image.size, Image.Resampling.BILINEAR)
    heatmap = plt.cm.jet(np.array(heatmap) / 255.)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap).convert("RGBA")
    heatmap.putalpha(90)
    overlay_image = Image.alpha_composite(image.convert("RGBA"), heatmap)

    if bbox is not None:
        width, height = image.size
        xmin, ymin, xmax, ymax = bbox
        draw = ImageDraw.Draw(overlay_image)
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline="lime", width=int(min(width, height) * 0.01))

        if inout_score is not None:
            text = f"in-frame: {inout_score:.2f}"
            text_width = draw.textlength(text)
            text_height = int(height * 0.01)
            text_x = xmin * width
            text_y = ymax * height + text_height
            draw.text((text_x, text_y), text, fill="lime", font=ImageFont.load_default(size=int(min(width, height) * 0.05)))

    overlay_image = cv2.cvtColor(np.array(overlay_image), cv2.COLOR_RGBA2BGR)        
    return overlay_image


def visualize_all(image, heatmaps, bboxes, inout_scores, inout_thresh=0.5):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    colors = ['lime', 'tomato', 'cyan', 'fuchsia', 'yellow']
    overlay_image = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay_image)
    width, height = image.size

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        if bbox is None:
            raise ValueError("if input_bboxes is [[None]], use --heatmap option to visualize heatmap.")
        xmin, ymin, xmax, ymax = bbox
        color = colors[i % len(colors)]
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline=color, width=int(min(width, height) * 0.01))

        if inout_scores is not None:
            inout_score = inout_scores[i]
            text = f"in-frame: {inout_score:.2f}"
            text_width = draw.textlength(text)
            text_height = int(height * 0.01)
            text_x = xmin * width
            text_y = ymax * height + text_height
            draw.text((text_x, text_y), text, fill=color, font=ImageFont.load_default(size=int(min(width, height) * 0.05)))

        if inout_scores is not None and inout_score > inout_thresh:
            heatmap_np = heatmaps[i]
            max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
            gaze_target_x = max_index[1] / heatmap_np.shape[1] * width
            gaze_target_y = max_index[0] / heatmap_np.shape[0] * height
            bbox_center_x = ((xmin + xmax) / 2) * width
            bbox_center_y = ((ymin + ymax) / 2) * height

            draw.ellipse([(gaze_target_x-5, gaze_target_y-5), (gaze_target_x+5, gaze_target_y+5)], fill=color, width=int(0.005*min(width, height)))
            draw.line([(bbox_center_x, bbox_center_y), (gaze_target_x, gaze_target_y)], fill=color, width=int(0.005*min(width, height)))

    overlay_image = cv2.cvtColor(np.array(overlay_image), cv2.COLOR_RGBA2BGR)
    return overlay_image
