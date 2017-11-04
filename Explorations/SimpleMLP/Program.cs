using SimpleMLP.Documentation;
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
        static List<TrainingData> BuildTrainingData()
        {
            List<TrainingData> trainingData = new List<TrainingData>();

            trainingData.Add(new TrainingData
            {
                X = new double[2] { .05, .1 },
                Y = new double[2] { .01, .99 }
            });

            return trainingData;
        }

        static void Main(string[] args)
        {
            // What I cannot create, I do not understand. 
            // ~Richard P. Feynman

            Network network = Network.BuildNetwork(2, 2, 2);
            network.SerializeTo("C:/users/ben/desktop/test.xml");

            NetworkInDGML dgmlRepresentation = NetworkInDGML.Create(network);
            dgmlRepresentation.Serialize("networkTopology.dgml");

            List<TrainingData> trainingData = BuildTrainingData();

            NetworkTrainer networkTrainer = new NetworkTrainer();
            networkTrainer.Train(network, trainingData, .1, 100);

            foreach (TrainingData data in trainingData)
            {
                double[] actualOutput = network.Execute(data.X);

                Console.WriteLine($"Output: {actualOutput[0]}, Expected Output: {data.Y[0]}");
            }

            return;
        }
    }
}
