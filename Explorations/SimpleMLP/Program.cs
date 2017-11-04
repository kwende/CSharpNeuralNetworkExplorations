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
        static List<TrainingData> BuildXORTrainingData()
        {
            List<TrainingData> trainingData = new List<TrainingData>();

            trainingData.Add(new TrainingData
            {
                X = new double[2] { 0, 0 },
                Y = new double[1] { 0 }
            });

            trainingData.Add(new TrainingData
            {
                X = new double[2] { 1, 0 },
                Y = new double[1] { 1 }
            });

            trainingData.Add(new TrainingData
            {
                X = new double[2] { 0, 1 },
                Y = new double[1] { 1 }
            });

            trainingData.Add(new TrainingData
            {
                X = new double[2] { 1, 1 },
                Y = new double[1] { 0 }
            });

            return trainingData;
        }

        static void Main(string[] args)
        {
            // What I cannot create, I do not understand. 
            // ~Richard P. Feynman

            Network network = Network.BuildNetwork(2, 1, 2, 3);

            NetworkInDGML dgmlRepresentation = NetworkInDGML.Create(network);
            dgmlRepresentation.Serialize("networkTopology.dgml");

            List<TrainingData> trainingData = BuildXORTrainingData();

            NetworkTrainer networkTrainer = new NetworkTrainer();
            networkTrainer.Train(network, trainingData, .01, 1000);

            foreach (TrainingData data in trainingData)
            {
                double[] actualOutput = network.Execute(data.X);

                Console.WriteLine($"Output: {actualOutput[0]}, Expected Output: {data.Y[0]}");
            }

            return;
        }
    }
}
