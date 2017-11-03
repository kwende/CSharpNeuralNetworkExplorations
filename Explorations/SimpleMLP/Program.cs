using SimpleMLP.MLP;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMLP
{
    class Program
    {
        static void Main(string[] args)
        {
            // What I cannot create, I do not understand. 
            // ~Richard P. Feynman

            Network network = Network.BuildNetwork(6, 6, 5, 15);

            List<TrainingData> trainingData = new List<TrainingData>();
            for (int c = 0; c < 5; c++)
            {
                TrainingData data = new TrainingData
                {
                    X = new double[6],
                    Y = new double[6]
                };

                // input a number in one-hot encoding
                data.X[c] = 1;
                // get back the next number in one-hot encoding. 
                data.Y[c + 1] = 1;

                trainingData.Add(data);
            }

            NetworkTrainer networkTrainer = new NetworkTrainer();
            networkTrainer.Train(network, trainingData);

            TrainingData testData = trainingData[1];
            double[] input = testData.X;
            double[] expectedOutput = testData.Y;

            double[] actualOutput = network.Execute(trainingData[1].X);

            Console.WriteLine($"Got ({string.Join(",", actualOutput.Select(n => n.ToString()))}), Expected ({string.Join(",", expectedOutput.Select(n => n.ToString()))})");

            return;
        }
    }
}
