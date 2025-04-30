import numpy as np
from typing import List, Tuple

class ExpertLSTMDataGenerator:
    def __init__(self, 
                 seed:              int     = 0,      
                 variation:         float   = .5, 
                 streakiness:       float   = .5,
                 translation_speed: float   = .1,
                 size_variation:    float   = .1,
                 num_experts:       int     = 3, 
                 num_samples:       int     = 1000,
                 num_classes:       int     = 2
                 ) -> None:
        #set vars
        self.seed = seed
        self.variation = variation
        self.streakiness = streakiness
        self.translation_speed = translation_speed
        self.size_variation = size_variation
        self.num_experts = num_experts
        self.num_samples = num_samples
        self.num_classes = num_classes
        # data format
        self.data_format = ["class_id", "center_x", "center_y", "width", "height"]

        #generate seed if needed and set np.random seed
        if self.seed == 0:
            self.seed = np.random.randint(0, 10000)

        np.random.seed(self.seed)
            
    def generate_ground_truth(self) -> List[List[float]]:
        data = [None] * self.num_samples
        #generate ground truth data
        
        #generate first frame
        new_frame = [None] * len(self.data_format)
        new_frame[0] = np.random.randint(1, self.num_classes)
        new_frame[1] = np.random.uniform(0, 1)
        new_frame[2] = np.random.uniform(0, 1)
        new_frame[3] = np.random.uniform(0, 1)
        new_frame[4] = np.random.uniform(0, 1)
        data[0] = new_frame

        for i in range(1, self.num_samples):
            new_frame = [None] * len(self.data_format)
            prev_frame = data[i-1]
            new_frame[0] = np.random.randint(1, self.num_classes)
            new_frame[1] = prev_frame[1] + np.random.uniform(-self.translation_speed, self.translation_speed)
            new_frame[2] = prev_frame[2] + np.random.uniform(-self.translation_speed, self.translation_speed)
            new_frame[3] = prev_frame[3] + np.random.uniform(-self.size_variation, self.size_variation)
            new_frame[4] = prev_frame[4] + np.random.uniform(-self.size_variation, self.size_variation)

            #make sure new frame is in bounds
            for tuple_index in range(1, 5):
                new_frame[tuple_index] = max(0, min(1, new_frame[tuple_index]))
            data[i] = new_frame        

        return data
    

if __name__ == "__main__":
    generator = ExpertLSTMDataGenerator()
    data = generator.generate_ground_truth()
    for i in range(10):
        print(data[i])



        






     


