using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMLP.MLP
{
    public class NetworkTrainer
    {
        public void Train(Network network, List<TrainingData> trainingData)
        {
            int trainingDataLength = trainingData.Count;

            // iterate over each instance of the training data. 
            for (int t = 0; t < trainingDataLength; t++)
            {
                // get an instance of training data. 
                TrainingData data = trainingData[t];

                // set the input to the net. 
                network.SetInputLayer(data.X);
                // feed forward. 
                network.Feedforward();
                // back propagation
                network.Backpropagation(data.Y);
            }
        }
    }
}
