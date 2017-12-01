using Common.DataStructures;
using Common.Interfaces;
using MLP;
using SimpleMLP.Trainers;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;

// for Linux: https://stackoverflow.com/questions/17628660/how-can-i-use-xbuild-to-build-release-binary

namespace SimpleMLP
{
    class Program
    {

        static int _lastEpoch = -1;

        static void OnLearningProgress(LearningProgress progress)
        {
            //if (_lastEpoch != progress.Epoch)
            //{
            //    _lastEpoch = progress.Epoch;

            //    Console.WriteLine($"Epoch {_lastEpoch}");
            //}
            //if (progress.Counter % 100 == 0)
            //{
            //    //Console.WriteLine($"Epoch {progress.Epoch}, Batch {progress.BatchNumber}, Error {progress.CurrentNetworkError}");
            //    File.AppendAllText("c:/users/ben/desktop/learningRate_smallDropOut.csv", $"{progress.CurrentNetworkError}\n");
            //}
        }

        static void OnValidationDataUpdate(double accuracy)
        {
            Console.WriteLine($"\tValidation accuracy {(accuracy * 100.0).ToString("000.00")}");
            //File.AppendAllText("C:/users/brush/desktop/validation.csv",
            //    $"{1 - accuracy}\n");
        }

        //static void FuckWithWeights(Network network)
        //{
        //    network.HiddenLayers[0].Neurons[0].Bias = -1.74976547;
        //    network.HiddenLayers[0].Neurons[0].UpstreamDendrites[0].Weight = 0.22117967;
        //    network.HiddenLayers[0].Neurons[0].UpstreamDendrites[1].Weight = -1.07004333;

        //    network.HiddenLayers[0].Neurons[1].Bias = 0.3426804;
        //    network.HiddenLayers[0].Neurons[1].UpstreamDendrites[0].Weight = -0.18949583;
        //    network.HiddenLayers[0].Neurons[1].UpstreamDendrites[1].Weight = 0.25500144;

        //    network.HiddenLayers[0].Neurons[2].Bias = 1.1530358;
        //    network.HiddenLayers[0].Neurons[2].UpstreamDendrites[0].Weight = -0.45802699;
        //    network.HiddenLayers[0].Neurons[2].UpstreamDendrites[1].Weight = 0.43516349;

        //    network.HiddenLayers[0].Neurons[3].Bias = -0.25243604;
        //    network.HiddenLayers[0].Neurons[3].UpstreamDendrites[0].Weight = -0.58359505;
        //    network.HiddenLayers[0].Neurons[3].UpstreamDendrites[1].Weight = 0.81684707;

        //    network.HiddenLayers[0].Neurons[4].Bias = 0.98132079;
        //    network.HiddenLayers[0].Neurons[4].UpstreamDendrites[0].Weight = 0.67272081;
        //    network.HiddenLayers[0].Neurons[4].UpstreamDendrites[1].Weight = -0.10441114;

        //    network.OutputLayer.Neurons[0].Bias = 0.51421884;
        //    network.OutputLayer.Neurons[0].UpstreamDendrites[0].Weight = -0.53128038;
        //    network.OutputLayer.Neurons[0].UpstreamDendrites[1].Weight = 1.02973269;
        //    network.OutputLayer.Neurons[0].UpstreamDendrites[2].Weight = -0.43813562;
        //    network.OutputLayer.Neurons[0].UpstreamDendrites[3].Weight = -1.11831825;
        //    network.OutputLayer.Neurons[0].UpstreamDendrites[4].Weight = 1.61898166;
        //}

        static void Main(string[] args)
        {
            // What I cannot create, I do not understand. 
            // ~Richard P. Feynman


            ITrainingDataBuilder trainingDataBuilder = new XORTrainingDataBuilder();
            trainingDataBuilder.BuildTrainingData();

            double totalAccuracy = 0.0;
            const int NumberOfIterations = 1;

            DateTime start = DateTime.Now;
            for (int c = 0; c < NumberOfIterations; c++)
            {
                Random rand = new Random();

                Network network = Network.BuildNetwork(
                    rand,
                    new Math.CostFunctions.CrossEntropyCostFunction(),
                    null, //new Math.RegularizationFunctions.L2Normalization(.1),
                    new DropoutLayerOptions(0),
                    2, 1, 5);

                NetworkTrainer networkTrainer = new NetworkTrainer();
                networkTrainer.Train(network,
                    trainingDataBuilder,
                    .5, 250, 1,
                    -1,
                    OnLearningProgress,
                    OnValidationDataUpdate);

                totalAccuracy += trainingDataBuilder.GradeResults(network, trainingDataBuilder.TestData) * 100.0;
            }

            Console.WriteLine($"Accurancy: {(totalAccuracy / (NumberOfIterations * 1.0)).ToString("000.00")}% in {(DateTime.Now - start).TotalSeconds} seconds.");

            return;
        }
    }
}