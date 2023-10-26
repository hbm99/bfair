import PIL
import clip
import torchvision.transforms as T

model, preprocess = clip.load("ViT-B/32", device="cpu")

image = preprocess(PIL.Image.open("datasets/noisymultifairface/9.jpg"))

transform = T.ToPILImage()

img = transform(image)

img.save("datasets/noisymultifairface/inverse_process_9.png")
