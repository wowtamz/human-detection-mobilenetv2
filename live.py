import torch
import torchvision.transforms as tfm

# pip install opencv-python
import cv2
from argparse import ArgumentParser

# TODO:
# from ... import get_anchor_grid
# from ... import batch_inference
# from ... import MmpNet

IMG_SIZE = 224
MIN_CONFIDENCE = 0.55


def clampi(x, minimum, maximum):
    return int(min(maximum, max(minimum, x)))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--camera", default=0, type=int, help="Camera index")
    args = parser.parse_args()

    # change the following values to the ones you used for training
    # or even better: 
    scales = [80, 140, 170, 220]
    aspect_ratios = [0.6, 1.0, 1.5, 2.2]
    scale_factor = 32.0

    transform = tfm.Compose(
        [
            tfm.ToTensor(),
            tfm.Resize(IMG_SIZE),
            tfm.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # prepare camera
    cap = cv2.VideoCapture(args.camera)

    # load model from checkpoint
    model = MmpNet(num_scales=len(scales), num_aspect_ratios=len(aspect_ratios))
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    # determine num_rows and num_cols from frame size
    ret, frame = cap.read()
    assert ret
    batch = transform(frame.copy()).unsqueeze(0)
    output = model(batch)
    num_rows = output.shape[-2]
    num_cols = output.shape[-1]
    print("num_rows:", num_rows, "num_cols:", num_cols)

    anchor_grid = get_anchor_grid(
        num_rows=num_rows,
        num_cols=num_cols,
        scale_factor=scale_factor,
        scales=scales,
        aspect_ratios=aspect_ratios,
    )

    with torch.no_grad():
        while True:
            # convert to torch tensor
            ret, frame = cap.read()
            assert ret
            batch = transform(frame.copy()).unsqueeze(0)
            boxes_scores = batch_inference(
                model, batch, torch.device("cpu"), anchor_grid
            )

            h, w, _ = frame.shape
            scale = h / float(IMG_SIZE)
            if frame is not None:
                for box, score in boxes_scores[0]:
                    if score < MIN_CONFIDENCE:
                        continue
                    # draw the boxes with a color ranging from blue to red,
                    # depending on the output confidence
                    color_score = 255 * (score - MIN_CONFIDENCE) / (1 - MIN_CONFIDENCE)
                    cv2.rectangle(
                        frame,
                        (clampi(box.x1 * scale, 0, w), clampi(box.y1 * scale, 0, h)),
                        (clampi(box.x2 * scale, 0, w), clampi(box.y2 * scale, 0, h)),
                        color=(255 - int(color_score), 0, int(color_score)),
                        thickness=2,
                    )
                cv2.imshow("frame", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
