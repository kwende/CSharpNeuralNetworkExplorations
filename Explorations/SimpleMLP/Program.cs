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

            Network network = Network.BuildNetwork(1, 1, 5, 15);

            List<TrainingData> trainingData = new List<TrainingData>();
            for (int c = 0; c < 10; c++)
            {
                TrainingData data = new TrainingData
                {
                    X = new double[1] { c },
                    Y = new double[1] { c + 1 },
                };
                trainingData.Add(data);
            }

            NetworkTrainer networkTrainer = new NetworkTrainer();
            networkTrainer.Train(network, trainingData);

            return;
        }
    }
}
