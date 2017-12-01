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
            if (_lastEpoch != progress.Epoch)
            {
                _lastEpoch = progress.Epoch;

                Console.WriteLine($"Epoch {_lastEpoch}");
            }
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

        static void FuckWithWeights(Network network)
        {
            double[] dendriteLayer1Neuron1Weights = new double[] { 0.84533894, 0.1411531, -0.82702529, -0.32890284, -0.30211765 };

            for (int c = 0; c < dendriteLayer1Neuron1Weights.Length; c++)
            {
                network.InputLayer.Neurons[0].DownstreamDendrites[c].Weight = dendriteLayer1Neuron1Weights[c];
            }

            double[] dendriteLayer1Neuron2Weights = new double[] { 0.22878778, 0.52066827, 0.36470008, 0.58120799, 0.36399436 };

            for (int c = 0; c < dendriteLayer1Neuron2Weights.Length; c++)
            {
                network.InputLayer.Neurons[1].DownstreamDendrites[c].Weight = dendriteLayer1Neuron2Weights[c];
            }

            foreach (Neuron neuron in network.HiddenLayers[0].Neurons)
            {
                neuron.Bias = 0;
            }

            double[] dendriteLayer2Neuron1Weights = new double[] { -0.07539773, -0.42560107, -0.93782991, 0.95553482, -0.41024405 };

            for (int c = 0; c < dendriteLayer2Neuron1Weights.Length; c++)
            {
                network.OutputLayer.Neurons[0].UpstreamDendrites[c].Weight = dendriteLayer2Neuron1Weights[c];
            }

            foreach (Neuron neuron in network.OutputLayer.Neurons)
            {
                neuron.Bias = 0;
            }
        }

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
                Random rand = new Random(100);

                Network network = Network.BuildNetwork(
                    rand,
                    new Math.CostFunctions.MeanSquaredErrorCostFunction(),
                    null, //new Math.RegularizationFunctions.L2Normalization(.1),
                    new DropoutLayerOptions(0),
                    2, 1, 5);

                FuckWithWeights(network);

                NetworkTrainer networkTrainer = new NetworkTrainer();
                networkTrainer.Train(network,
                    trainingDataBuilder,
                    .5, 100, 1,
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