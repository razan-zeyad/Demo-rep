# Traffic Classification in Hazy Conditions

This project addresses the pressing issue of traffic congestion in urban areas, which 
exacerbates by hazy conditions, leading to increased travel times, fuel consumption, and air 
pollution. The proposed solution employs deep learning techniques to classify street 
congestion severity in simulated hazy conditions, aiming to provide valuable insights for 
transportation authorities, drivers, and urban planners.

The methodology encompasses several key steps, including data collection and preprocessing, model training and evaluation, fine-tuning for adverse weather conditions, and 
measuring model performance. By employing transfer learning and fine-tuning techniques on 
a pre-trained EfficientNet-V2L model, we customize the deep learning architecture 
specifically for congestion classification.

The training data was collected from Queen Rania Street in Amman and consisted of three 
recorded videos, each of which was split into two parts: one for training the model and 
covering the first 6 minutes, and the other for testing the model which covers the last 2 
minutes. The images were then processed and edited for two purposes: enhancing the model's 
ability to capture more features from clear images, and generating synthesized hazy images.

There was a variation in the amount of training data used in each fine tuning step. Initially, 
the model underwent training on 300 clear images. Subsequently, when assessing the model's 
performance under hazy weather conditions, the training dataset size expanded to include a 
total of 600 images: 300 clear and 300 hazy. For both experiments, the test dataset comprised 
of 157 images.

The training of the EfficientNet-V2L model lasted for 80 epochs, utilizing a batch size of 32, 
and commenced with a learning rate of 0.001. Employing the multiplicative learning rate
technique decreased it by a factor of 0.2 at each of the last 40 epochs. During training, the 
model utilized categorical cross-entropy as the loss function and was optimized using the 
Adam optimizer.

The model achieved 88% test accuracy when trained and evaluated on clear image datasets, 
while it attained 78.39% test accuracy when trained on both clear and hazy images and 
evaluated solely on hazy ones.

The project's impact is centered on leveraging a pre-trained model and applying transfer 
learning to adapt its capabilities to classify street density under challenging weather 
conditions. This advancement enables better understanding and management of traffic 
dynamics, leading to improved safety, efficiency, and overall urban mobility.git add README.md


