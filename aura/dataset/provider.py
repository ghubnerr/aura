class DatasetProvider:
    """
    Class to provide 
    """
    def __init__(self):
        pass

    def get_next_image(self, **kwargs):
        """
        Returns a random, unprocessed image from our dataset
        """
        yield None

    def get_next_image_batch(self, **kwargs):
        """
        Returns groups of images (positives) to train SimCLR embeddings
        """
        yield None

    # TODO: Complete methods and add more methods