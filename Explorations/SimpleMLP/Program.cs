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

            //trainingData.Add(new TrainingData
            //{
            //    X = new double[2] { 0, 0 },
            //    Y = new double[1] { 0 }
            //});

            //trainingData.Add(new TrainingData
            //{
            //    X = new double[2] { 1, 1 },
            //    Y = new double[1] { 0 }
            //});

            //trainingData.Add(new TrainingData
            //{
            //    X = new double[2] { 1, 0 },
            //    Y = new double[1] { 1 }
            //});

            //trainingData.Add(new TrainingData
            //{
            //    X = new double[2] { 0, 1 },
            //    Y = new double[1] { 1 }
            //});

            return trainingData;
        }

        static void Main(string[] args)
        {
            // What I cannot create, I do not understand. 
            // ~Richard P. Feynman

            Network network = Network.BuildNetwork(2, 2, 2);

            // hackery for testing. 
            network.HiddenLayers[0].Neurons[0].Dendrites[0].Weight = .15;
            network.HiddenLayers[0].Neurons[0].Dendrites[1].Weight = .2;
            network.HiddenLayers[0].Neurons[1].Dendrites[0].Weight = .25;
            network.HiddenLayers[0].Neurons[1].Dendrites[1].Weight = .30;

            network.HiddenLayers[0].Neurons[0].Bias = .35;
            network.HiddenLayers[0].Neurons[1].Bias = .35;

            network.OutputLayer.Neurons[0].Dendrites[0].Weight = .4;
            network.OutputLayer.Neurons[0].Dendrites[1].Weight = .45;
            network.OutputLayer.Neurons[1].Dendrites[0].Weight = .50;
            network.OutputLayer.Neurons[1].Dendrites[1].Weight = .55;

            network.OutputLayer.Neurons[0].Bias = .6;
            network.OutputLayer.Neurons[1].Bias = .6;

            NetworkInDGML dgmlRepresentation = NetworkInDGML.Create(network);
            dgmlRepresentation.Serialize("networkTopology.dgml");

            List<TrainingData> trainingData = BuildTrainingData();

            NetworkTrainer networkTrainer = new NetworkTrainer();
            networkTrainer.Train(network, trainingData, .5, 50000);

            foreach (TrainingData data in trainingData)
            {
                double[] actualOutput = network.Execute(data.X);

                Console.WriteLine($"Output: {actualOutput[0]}, Expected Output: {data.Y[0]}");
            }

            return;
        }
    }
}
