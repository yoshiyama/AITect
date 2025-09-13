import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

def draw_boxes(image_tensor, boxes, scores=None, score_thresh=0.5):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for i, box in enumerate(boxes):
        if scores is not None and scores[i] < score_thresh:
            continue
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if scores is not None:
            ax.text(x1, y1 - 5, f"{scores[i]:.2f}", color='yellow', fontsize=10)

    plt.axis('off')
    plt.tight_layout()
    plt.show()
