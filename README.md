# Location Pinner
My model is modified from AlexNet, a heatmap is generated to decide the whether  
object in the heatmap and attention loss is generated with softmax function   
model train with attention loss and classification loss to pinpoint the location of object.  
I got around 83% of accuracy with my tool dataset with this model    
To train the model: 
```bash
python train_location_pinner.py ~/sample_dataset
``` 
To test the model, it will output the predicted location and image with center overlays: 
```bash
python pin_location.py sample/example.jpg
``` 
output overlay result:  
![alt text](example_overlay.jpg)  
green dot is predicted center point of the object and the red dot is labeled center point of the object  



