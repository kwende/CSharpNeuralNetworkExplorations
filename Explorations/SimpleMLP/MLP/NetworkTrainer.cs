using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMLP.MLP
{
    public class NetworkTrainer
    {
        public void Train(Network network, List<TrainingData> trainingData, double stepSize, int numberOfIterations)
        {
            int trainingDataLength = trainingData.Count;
            //using (StreamWriter fout = File.CreateText("c:/users/ben/desktop/turd.csv"))
            {
                for (int iterationCount = 0; iterationCount < numberOfIterations; iterationCount++)
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
                    if(iterationCount % 5 == 0)
                    {
                        //fout.WriteLine(debugAverageOutputError / (trainingDataLength * 1.0)); 
                        Console.WriteLine($"Iteration {iterationCount}, network error: {debugAverageOutputError / (trainingDataLength * 1.0)}");
                    }

                    // update the network. 
                    network.UpdateNetwork(stepSize);

                    //Console.WriteLine($"\t0: {network.Execute(new double[1] { 0 })[0]}");
                    //Console.WriteLine($"\t1: {network.Execute(new double[1] { 1 })[0]}");
                }
            }
        }
    }
}
