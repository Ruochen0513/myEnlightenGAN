from PIL import Image

def resize_image(input_path, output_path, size=(600, 400)):
    with Image.open(input_path) as img:
        resized_img = img.resize(size)
        resized_img.save(output_path)

if __name__ == "__main__":
    input_image_path = '/home/amadeus/桌面/frankfurt_000000_002197_leftImg8bit.png'
    output_image_path = '/home/amadeus/桌面/frankfurt_000000_002197_leftImg8bit.png'
    resize_image(input_image_path, output_image_path)