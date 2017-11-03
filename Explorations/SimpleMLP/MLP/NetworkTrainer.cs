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

            for (int iterationCount = 0; iterationCount < 100; iterationCount++)
            {
                double debugAverageOutputError = 0.0;
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
                    debugAverageOutputError += network.Backpropagation(data.Y);
                }
                //Console.WriteLine($"Average network error: {averageOutputError / (trainingDataLength * 1.0)}");

                // update the network. 
                network.UpdateNetwork(1.0);
            }
        }
    }
}
