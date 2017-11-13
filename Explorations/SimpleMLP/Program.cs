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
            List<TrainingData> ret = new List<TrainingData>();

            List<MNIST.DigitImage> images = MNIST.Reader.Read(
                "train-labels.idx1-ubyte",
                "train-images.idx3-ubyte");

            foreach (MNIST.DigitImage image in images)
            {
                int height = image.pixels.Length;
                int width = image.pixels[0].Length;

                double[] imagePixels = new double[width * height];

                for (int y = 0, i = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++, i++)
                    {
                        imagePixels[i] = image.pixels[x][y];
                    }
                }

                double[] label = new double[10];
                label[image.label] = 1;

                ret.Add(new TrainingData
                {
                    X = imagePixels,
                    Y = label,
                });
            }

            return ret;
        }

        static void Main(string[] args)
        {
            // What I cannot create, I do not understand. 
            // ~Richard P. Feynman

            Network network = Network.BuildNetwork(784, 10, 15);

            //NetworkInDGML dgmlRepresentation = NetworkInDGML.Create(network);
            //dgmlRepresentation.Serialize("networkTopology.dgml");

            List<TrainingData> trainingData = BuildTrainingData();

            NetworkTrainer networkTrainer = new NetworkTrainer();
            networkTrainer.Train(network, trainingData, .5, 1000000);

            foreach (TrainingData data in trainingData)
            {
                double[] actualOutput = network.Execute(data.X);

                Console.WriteLine($"Output: {actualOutput[0]}, Expected Output: {data.Y[0]}");
            }

            return;
        }
    }
}
