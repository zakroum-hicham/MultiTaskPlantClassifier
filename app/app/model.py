import torch
from torch import nn
from torchvision import models


class MultiTaskLeafDiseaseClassifier(nn.Module):
    def __init__(self, num_leaf_classes=3, num_potato_disease_classes=3, 
                 num_tomato_disease_classes=10, num_pepper_disease_classes=2):
        super(MultiTaskLeafDiseaseClassifier, self).__init__()

        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc = nn.Identity()  # Remove the last layer

        # Leaf Type Classifier (Potato, Tomato, Pepper)
        self.leaf_classifier = nn.Linear(2048, num_leaf_classes)
        
        # Disease Classifiers
        self.disease_classifier = nn.Linear(2048, num_potato_disease_classes+num_pepper_disease_classes+num_tomato_disease_classes)

    def forward(self, x):

        features = self.base_model(x)
        
        # Leaf Classification
        leaf_type_output = self.leaf_classifier(features)

        # Disease Classification
        disease_output = self.disease_classifier(features)

        return leaf_type_output, disease_output


model = MultiTaskLeafDiseaseClassifier()



