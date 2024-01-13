from PIL import Image

# Open the image with the alpha channel
image_with_alpha = Image.open('../data/training_set/distances/33_38.png')

# Convert the image to 'L' mode which stands for grayscale
# This will discard the alpha channel if it exists
greyscale_image = image_with_alpha.convert('L')

# Save the resulting image without the alpha channel
greyscale_image.save('../data/training_set/distances/33_38.png')
