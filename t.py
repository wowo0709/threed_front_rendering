import matplotlib.pyplot as plt
from PIL import Image

a = Image.open('/root/desktop/3D-FRONT/3D-FRONT-processed/bedrooms_without_lamps_full_images/images_256_depth/00110bde-f580-40be-b8bb-88715b338a2a_Bedroom-43072/0000.png')

plt.imshow(a, cmap='viridis')
plt.show()