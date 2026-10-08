from PIL import Image, ImageDraw

image_path = (
    "data/raw/lss_mri_dataset/Segmentation/"
    "Middle_Slice/0001_IM000007.png"
)
output_path = "outputs/sample_lumbar_bbox.png"

image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)

box = (188, 125, 378, 494)

draw.rectangle(box, outline=(255, 0, 0), width=4)
draw.text((box[0], max(0, box[1] - 15)), "Whole lumbar", fill=(255, 0, 0))

image.save(output_path)
print(f"Saved: {output_path}")