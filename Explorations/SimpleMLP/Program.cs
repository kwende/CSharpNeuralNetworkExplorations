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

        static void Main(string[] args)
        {
            // What I cannot create, I do not understand. 
            // ~Richard P. Feynman


            ITrainingDataBuilder trainingDataBuilder = new MNISTTrainingDataBuilder();
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
                    784, 10, 30);

                NetworkTrainer networkTrainer = new NetworkTrainer();
                networkTrainer.Train(network,
                    trainingDataBuilder,
                    .5, 30, 10,
                    1,
                    OnLearningProgress,
                    OnValidationDataUpdate);

                totalAccuracy += trainingDataBuilder.GradeResults(network, trainingDataBuilder.TestData) * 100.0;
            }

            Console.WriteLine($"Accurancy: {(totalAccuracy / (NumberOfIterations * 1.0)).ToString("000.00")}% in {(DateTime.Now - start).TotalSeconds} seconds.");

            return;
        }
    }
}