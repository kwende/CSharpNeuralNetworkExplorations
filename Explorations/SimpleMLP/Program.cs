using SimpleMLP.Documentation;
using SimpleMLP.MLP;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
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
            networkTrainer.Train(network, trainingData, .5, 1);

            using (FileStream fout = File.Create("serialized.dat"))
            {
                BinaryFormatter bf = new BinaryFormatter();
                bf.Serialize(fout, network);
            }

            return;
        }
    }
}
