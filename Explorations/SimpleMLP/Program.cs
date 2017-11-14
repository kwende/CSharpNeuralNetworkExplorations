﻿using SimpleMLP.Documentation;
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
        static List<TrainingData> BuildTrainingDataFromMNIST(string labelsFile, string imagesFile)
        {
            List<TrainingData> ret = new List<TrainingData>();

            List<MNIST.DigitImage> images = MNIST.Reader.Read(labelsFile, imagesFile);

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

        static List<TrainingData> BuildTrainingDataForXOR()
        {
            List<TrainingData> ret = new List<TrainingData>();

            ret.Add(new TrainingData
            {
                X = new double[2] { 0, 1 },
                Y = new double[1] { 1 },
            });

            ret.Add(new TrainingData
            {
                X = new double[2] { 1, 0 },
                Y = new double[1] { 1 },
            });

            ret.Add(new TrainingData
            {
                X = new double[2] { 1, 1 },
                Y = new double[1] { 0 },
            });

            ret.Add(new TrainingData
            {
                X = new double[2] { 0, 0 },
                Y = new double[1] { 0 },
            });

            return ret;
        }

        static void Main(string[] args)
        {
            // What I cannot create, I do not understand. 
            // ~Richard P. Feynman

            Network network = Network.BuildNetwork(2, 1, 5, 15);

            //NetworkInDGML dgmlRepresentation = NetworkInDGML.Create(network);
            //dgmlRepresentation.Serialize("networkTopology.dgml");

            //List<TrainingData> trainingData = BuildTrainingDataFromMNIST("train-labels.idx1-ubyte", "train-images.idx3-ubyte");
            List<TrainingData> trainingData = BuildTrainingDataForXOR();

            NetworkTrainer networkTrainer = new NetworkTrainer();
            networkTrainer.Train(network, trainingData, .5, 100, 1);

            Console.WriteLine(string.Join(",", network.Execute(new double[2] { 0, 1 })));
            Console.WriteLine(string.Join(",", network.Execute(new double[2] { 1, 0 })));
            Console.WriteLine(string.Join(",", network.Execute(new double[2] { 0, 0 })));
            Console.WriteLine(string.Join(",", network.Execute(new double[2] { 1, 1 })));

            //using (FileStream fout = File.Create("serialized_meanSquaredError.dat"))
            //{
            //    BinaryFormatter bf = new BinaryFormatter();
            //    bf.Serialize(fout, network);
            //}

            // List<TrainingData> testData = BuildTrainingDataFromMNIST("t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte");
            //List<TrainingData> testData = BuildTrainingDataForXOR();

            //Console.WriteLine($"Accurancy: {(networkTrainer.Test(network, testData) * 100.0).ToString("000.00")}%");

            return;
        }
    }
}
