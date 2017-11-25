﻿using Common.DataStructures;
using MLP;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

// for Linux: https://stackoverflow.com/questions/17628660/how-can-i-use-xbuild-to-build-release-binary

namespace SimpleMLP
{
    class Program
    {
        static void WriteTrainingDataToDisk(TrainingData data, string outputFile)
        {
            using (Bitmap bmp = new Bitmap(data.XWidth, data.XHeight))
            {
                for (int y = 0, i = 0; y < data.XHeight; y++)
                {
                    for (int x = 0; x < data.XWidth; x++, i++)
                    {
                        byte v = (byte)(data.X[i] * 255);
                        bmp.SetPixel(x, y, Color.FromArgb(v, v, v));
                    }
                }

                bmp.Save(outputFile);
            }
        }

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
                        imagePixels[i] = image.pixels[x][y] / 255.0;
                    }
                }

                double[] label = new double[10];
                label[image.label] = 1;

                ret.Add(new TrainingData
                {
                    X = imagePixels,
                    Y = label,
                    XWidth = width,
                    XHeight = height,
                    Label = image.label,
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

        static void OnLearningProgress(LearningProgress progress)
        {
            //if (progress.Counter % 1000 == 0)
            //{
            //    Console.WriteLine($"Epoch {progress.Epoch}, Batch {progress.BatchNumber}, Error {progress.CurrentNetworkError}");
            //    //File.AppendAllText("learningRate.csv", $"{progress.CurrentNetworkError}\n");
            //}
        }

        static void Main(string[] args)
        {
            // What I cannot create, I do not understand. 
            // ~Richard P. Feynman

            List<TrainingData> trainingData = BuildTrainingDataFromMNIST(
                "train-labels.idx1-ubyte", "train-images.idx3-ubyte");
            List<TrainingData> testData = BuildTrainingDataFromMNIST(
                "t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte");

            trainingData = trainingData.Take(5000).ToList();
            Random rand = new Random(1234);

            Network network = Network.BuildNetwork(
                rand,
                new Math.CostFunctions.CrossEntropyCostFunction(),
                new Math.RegularizationFunctions.L2Normalization(.1),
                new DropoutLayerOptions(.5, 1),
                784, 10, 30);

            double totalAccuracy = 0.0;

            NetworkTrainer networkTrainer = new NetworkTrainer();
            networkTrainer.Train(network,
                trainingData,
                .25, 30, 5, OnLearningProgress);

            totalAccuracy += networkTrainer.Test(network, testData) * 100.0;

            Console.WriteLine($"Accurancy: {(totalAccuracy / 10.0).ToString("000.00")}%");

            return;
        }
    }
}
