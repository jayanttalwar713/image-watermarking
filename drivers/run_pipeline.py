from regions.segment import segment_image
from descriptors.compute_descriptors import compute_descriptors

image = load_image("test.jpg")

region_map, num_regions = segment_image(image)

descriptors = compute_descriptors(image, region_map, num_regions)

print(descriptors)