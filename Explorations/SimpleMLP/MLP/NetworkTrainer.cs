using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace SimpleMLP.MLP
{
    public class NetworkTrainer
    {
        public void Train(Network network, List<TrainingData> trainingData, double stepSize, int numberOfIterations, int batchSize)
        {
            int trainingDataLength = trainingData.Count;

            Batch[] batches = Batch.CreateBatches(trainingData, numberOfIterations, batchSize, new Random());

            for (int b = 0; b < batches.Length; b++)
            {
                Batch batch = batches[b];
                double debugAverageOutputError = 0.0;
                // iterate over each instance of the training data. 
                for (int n = 0; n < batchSize; n++)
                {
                    // get an instance of training data. 
                    TrainingData data = batch.Data[n];

                    // set the input to the net. 
                    network.SetInputLayer(data.X);
                    // feed forward. 
                    network.Feedforward();
                    // back propagation
                    debugAverageOutputError += network.Backpropagation(data.Y);
                };

                if (b % 5 == 0)
                {
                    //fout.WriteLine(debugAverageOutputError / (trainingDataLength * 1.0)); 
                    Console.WriteLine($"Iteration {b}, network error: {debugAverageOutputError / (trainingDataLength * 1.0)}");
                }

                // update the network. 
                network.UpdateNetwork(stepSize);
            }
        }
    }
}
