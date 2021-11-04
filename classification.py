#!/usr/bin/python3

"""This file is used to test the hypotheses to classify a given set of
data extracted from an image.

It will classify the data to see if its corresponding image exibits beta denudation,
has an even distribution of alpha and beta, or that we see the
opposite trend.

We can use a naive bayes classifier as a first stab, to classify the
boundary between a beta denudated reduction from the surface, and the
actual bulk. From this we can draw a line across the image, which
informs us where the bulk volume fraction is found with a given
confidence.

A prior distribution, which encodes how likely we think there could be
beta denudation in a sample could be encoded from a global gradient.

We will use Bayes rule to compute this.

"""

# Make an abstract base class which supplies a processing function upon an image
class DataClassification(ABC):
    
    @abstractmethod
    def classify(self, data):
        pass
            
    @abstractmethod
    def plot(self, data):
        pass


class BayesClassifier


class ClassificationContainer:
    def __init__(self, classification_method: Type[DataClassification], data_directory: str, image_directory: str):
        self.data_directory = data_directory
        self.image_directory = image_directory        
        self.data = os.listdir(self.data_directory)

        self.method = classification_method()
        self.method_name = self.method.name

        now = datetime.now()
        self.dt = now.strftime("%Y-%m-%d--%H-%M-%S")

    def get_filename_from_id(self, name):
        
        analysis_name = os.path.splitext(name.split('_')[0])[0]
        base_name = os.path.splitext(name.split('_')[1])[0]
        extension = os.path.splitext(name)[1]
        prefixed = [filename for filename in os.listdir(self.image_directory) if filename.startswith(base_name)]

        # Assuming just one identifier from image name, can't deal with multiple images'
        print("Unprocessed image: ", prefixed[0])
        return prefixed[0]

    def classify_images(self, plot=True):

        for data in self.data_directory:
            unprocessed_name = self.get_filename_from_id(image)
            original_image =   color.rgb2gray(imageio.imread(f"{self.image_directory}/{unprocessed_name}"))
            self.method.classify( original_image, unprocessed_image )
            if plot:
                self.method.plot(original_image)
            
            self.save(image, self.method.data)
            

            
        for image in self.images:
            original_image =   color.rgb2gray(imageio.imread(f"{self.image_directory}/{image}"))
            unprocessed_name = self.get_filename_from_id(image)
            unprocessed_image =   color.rgb2gray(imageio.imread(f"{self.original_image_directory}/{unprocessed_name}"))
            self.method.analyse( original_image, unprocessed_image )
            if plot:
                self.method.plot(original_image)
            
            self.save(image, self.method.data)


    def create_output_directory(self):
        dir_name = f"{self.method_name}_{self.dt}_{self.image_directory}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        return dir_name
    

    def save(self, image_name, data):
        self.dir_name = self.create_output_directory()
        name = os.path.splitext(image_name)[0]
        path = os.path.join(self.dir_name, f"{self.method_name}_{name}.dat")
        
        
        print(f"> Saving data to {path}")
        with open(path, 'w') as f:

            if isinstance(self.method.data, np.ndarray):
            
                f.write(self.method.comment + "\n")

                for row in self.method.data:
                    data_line = (' '.join(["{: 12.6f}" * len(row)])).format( *tuple(row) ) + "\n"
                    f.write(data_line)
            elif isinstance(self.method.data, str):
                f.write(self.method.data)
        
            

        


if __name__ == '__main__':
    # Test out the functionality

    plot=False
    image_directory = "images_RemoveBakelite_WhiteBackgroundRemoval_OtsuThreshold"
    
    print(f"Analysing images from {image_directory}..")    
    # Now analyse
    print(f"> Alpha-beta fraction")    
    analysis = AlphaBetaFraction
    ac = AnalysisContainer(analysis, image_directory)
    ac.analyse_images(plot=plot)

    print(f"> Chris Alpha-beta fraction")    
    analysis = ChrisAlphaBetaFraction
    ac = AnalysisContainer(analysis, image_directory)
    ac.analyse_images(plot=plot)

    
    
    
        



