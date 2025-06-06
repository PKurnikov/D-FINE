"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

# import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from pathlib import Path
import os

def find_files(catalog, extensions):
    """
    Самый быстрый вариант поиска файлов с расширениями в каталоге и подкаталогах.
    """
    return [str(p) for ext in extensions for p in Path(catalog).rglob(f'*{ext}')]

def resize_plain(image, size, interpolation=Image.BILINEAR):
    """
    Ресайз изображения до заданного размера без сохранения пропорций.

    :param image: PIL.Image объект
    :param size: tuple (width, height)
    :param interpolation: метод интерполяции (по умолчанию BILINEAR)
    :return: измененное изображение
    """
    return image.resize(size, interpolation)

def resize_with_aspect_ratio(image, size, interpolation=Image.BILINEAR):
    """Resizes an image while maintaining aspect ratio and pads it."""
    original_width, original_height = image.size
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    image = image.resize((new_width, new_height), interpolation)

    # Create a new image with the desired size and paste the resized image onto it
    new_image = Image.new("RGB", (size, size))
    new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
    return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2


def draw(images, labels, boxes, scores, ratios=None, paddings=None, thrh=0.4):
    result_images = []
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scr = scr[scr > thrh]

        # Задаем значения по умолчанию
        ratio = ratios[i] if ratios is not None else 1.0
        pad_w, pad_h = paddings[i] if paddings is not None else (0, 0)

        for lbl, bb in zip(lab, box):
            # Adjust bounding boxes according to the resizing and padding
            bb = [
                (bb[0] - pad_w) / ratio,
                (bb[1] - pad_h) / ratio,
                (bb[2] - pad_w) / ratio,
                (bb[3] - pad_h) / ratio,
            ]
            draw.rectangle(bb, outline="red")
            draw.text((bb[0], bb[1]), text=str(lbl), fill="blue")

        result_images.append(im)
    return result_images


def process_image(sess, im_pil):
    # Resize image while preserving aspect ratio
    # resized_im_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(im_pil, [576, 1024])
    # orig_size = torch.tensor([[resized_im_pil.size[1], resized_im_pil.size[0]]])

    resized_im_pil = resize_plain(im_pil, (1024, 576))

    transforms = T.Compose(
        [
            T.ToTensor(),
        ]
    )
    im_data = transforms(resized_im_pil).unsqueeze(0)

    output = sess.run(
        output_names=None,
        input_feed={"images": im_data.numpy()},
    )

    segmentations, detections = output

    boxes  = detections[:, :, :4] * 4.0
    scores = detections[:, :, 4:5][:,:,0]
    labels = detections[:, :, 5:6][:,:,0]
    # labels, boxes, scores = output

    result_images = draw([resized_im_pil], labels, boxes, scores)

    return result_images


def process_video(sess, video_path):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("onnx_result.mp4", fourcc, fps, (orig_w, orig_h))

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Resize frame while preserving aspect ratio
        resized_frame_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(frame_pil, 640)
        orig_size = torch.tensor([[resized_frame_pil.size[1], resized_frame_pil.size[0]]])

        transforms = T.Compose(
            [
                T.ToTensor(),
            ]
        )
        im_data = transforms(resized_frame_pil).unsqueeze(0)

        output = sess.run(
            output_names=None,
            input_feed={"images": im_data.numpy(), "orig_target_sizes": orig_size.numpy()},
        )

        labels, boxes, scores = output

        # Draw detections on the original frame
        result_images = draw([frame_pil], labels, boxes, scores, [ratio], [(pad_w, pad_h)])
        frame_with_detections = result_images[0]

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_with_detections), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'result.mp4'.")


def main(args):
    """Main function."""
    # Load the ONNX model
    sess = ort.InferenceSession(args.onnx)
    print(f"Using device: {ort.get_device()}")

    input_path = args.input
    output_path = args.output

    files = find_files(input_path, ['.jpg', '.png'])

    for f in files:
        try:
            # Try to open the input as an image
            im_pil = Image.open(f).convert("RGB")
            # # Преобразовать в NumPy и в BGR
            # im_np = np.array(im_pil)
            # im_bgr = im_np[:, :, ::-1]  # это переворачивает RGB → BGR
            # im_pil = Image.fromarray(im_bgr)

            result_images = process_image(sess, im_pil)
            dst = os.path.join(output_path, os.path.basename(f))
            result_images[0].save(dst)
            print("Image processing complete. Result saved as 'result.jpg'.")
        except IOError:
            print(f)
            # Not an image, process as video
            # process_video(sess, input_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True, help="Path to the ONNX model file.")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input dir image or video file."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output dir image or video file."
    )
    args = parser.parse_args()
    main(args)