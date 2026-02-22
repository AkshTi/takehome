# TOPIC A

# Experiment 1

The first experiment aims to determine whether the temperature that is applied (which will affect the distribution of the logits produced) has an impact on the subliminal learning. In theory, what this will do is that an increasing temperature T will give more weightage (make the distribution more uniform) and uncover more of the dark probabilities that take place within a particular training run. We run in seeds of 3 and across temperatures values [2, 4, 6, 8, 10] and measure the accuracy as well as the gradient difference between teh weights per epoch per layer. Everything else in the experiment is kept constant. 

# Experiment 2

The second experiment aims to determine whether changign the activation function from a ReLU to a Tanh. In this case, the ReLU gates how the gardients flow and often times can also kill the negative acivations, affecting how much grdient reaches the weights during backpropagation. Meanwhile, switching to tanh gives us a different approach as we are bale to determine how the smoothness will affect how much flows into the gradient. 

# Experiment 3
The third experiment here demonstrates th effect that increasing the number of auxiliary logits has on the subliminal learning of the student model. In this case, increasing the auxiliary logits that the student model has to train on effecitvely gives it more "training data" in the final layer. We vary the number of auxiliary logits from [5, 10, 20, 30, 40, 50] in seeds of 3. Again we measure the gradient normalization per epoch per layer. 

# Experiment 4
The fourth experiment here essentially recreates the base experiment in the paper but trains the model on a completely differnet MNIST dataset (out of distribution inputs). The teacher model will be trained on MNIST as previously and instead the student model will be trained on fashionMNIST as labels and taught the distributions of the teacher model (as done previously) holding the number of auxiliary logits as zero. 