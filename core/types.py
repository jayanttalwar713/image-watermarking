class SegmentationResult:
    def __init__(self, region_map, num_regions):
        self.region_map = region_map
        self.num_regions = num_regions