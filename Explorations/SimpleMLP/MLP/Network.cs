﻿using Common.Interfaces;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace SimpleMLP.MLP
{
    [Serializable]
    public class Network
    {
        public InputLayer InputLayer { get; set; }
        public List<HiddenLayer> HiddenLayers { get; set; }
        public OutputLayer OutputLayer { get; set; }

        private ICostFunction _costFunction;

        private Network(ICostFunction costFunction)
        {
            HiddenLayers = new List<HiddenLayer>();
            _costFunction = costFunction;
        }

        public static Network BuildNetwork(ICostFunction costFunction, int inputNeuronCount,
            int outputNeuronCount, params int[] hiddenLayerCounts)
        {
            Math.RandomNormal rand = new Math.RandomNormal(0, 1);

            Network network = new Network(costFunction);

            network.InputLayer = InputLayer.BuildInputLayer(rand, inputNeuronCount);

            Layer previousLayer = network.InputLayer;
            for (int c = 0; c < hiddenLayerCounts.Length; c++)
            {
                int currentLayerCount = hiddenLayerCounts[c];
                HiddenLayer hiddenLayer = HiddenLayer.BuildHiddenLayer(rand, previousLayer, currentLayerCount);
                network.HiddenLayers.Add(hiddenLayer);
                previousLayer = hiddenLayer;
            }

            network.OutputLayer = OutputLayer.BuildOutputLayer(rand, (HiddenLayer)previousLayer, outputNeuronCount);

            return network;
        }

        public void UpdateNetwork(double stepSize)
        {
            List<Layer> layersToUpdate = new List<Layer>();
            foreach (HiddenLayer hiddenLayer in HiddenLayers)
            {
                layersToUpdate.Add(hiddenLayer);
            }
            layersToUpdate.Add(OutputLayer);

            for (int c = layersToUpdate.Count - 1; c >= 0; c--)
            {
                foreach (Neuron neuron in layersToUpdate[c].Neurons)
                {
                    double delta = stepSize * neuron.BatchErrors.Average();
                    neuron.BatchErrors.Clear();

                    neuron.Bias = neuron.Bias - delta;

                    foreach (Dendrite dendrite in neuron.Dendrites)
                    {
                        double changeInErrorRelativeToWeight =
                            (delta * ((Neuron)dendrite.UpStreamNeuron).Activation);

                        dendrite.Weight = dendrite.Weight -
                            (changeInErrorRelativeToWeight);
                    }
                }
            }
        }

        public double Backpropagation(double[] expectedValues)
        {
            double totalNetworkError = 0.0;
            // Compute error for the output neurons to get the ball rolling. 
            // See https://github.com/kwende/CSharpNeuralNetworkExplorations/blob/master/Explorations/SimpleMLP/Documentation/OutputNeuronErrors.png
            for (int d = 0; d < expectedValues.Length; d++)
            {
                Neuron outputNeuronBeingExamined = OutputLayer.Neurons[d];
                double expectedOutput = expectedValues[d];
                double actualOutput = outputNeuronBeingExamined.Activation;
                double actualInput = outputNeuronBeingExamined.TotalInput;

                double error = _costFunction.Compute(expectedOutput, actualOutput);
                totalNetworkError += error;

                double changeInErrorRelativeToActivation =
                    (_costFunction.ComputeDerivativeWRTActivation(actualOutput, expectedOutput));

                double delta = changeInErrorRelativeToActivation *
                    Math.Sigmoid.ComputeDerivative(actualInput);

                outputNeuronBeingExamined.BatchErrors.Add(delta);
            }

            // Compute error for each neuron in each layer moving backwards (backprop). 
            Layer nextLayer = OutputLayer;
            for (int d = HiddenLayers.Count - 1; d >= 0; d--)
            {
                HiddenLayer hiddenLayer = HiddenLayers[d];
                for (int e = 0; e < hiddenLayer.Neurons.Count; e++)
                {
                    Neuron thisLayerNeuron = (Neuron)hiddenLayer.Neurons[e];
                    double input = thisLayerNeuron.TotalInput;

                    double errorSum = 0.0;
                    List<Dendrite> downStreamDendrites = nextLayer.Neurons.SelectMany(
                        n => n.Dendrites.Where(l => l.UpStreamNeuron == thisLayerNeuron)).ToList();

                    for (int f = 0; f < downStreamDendrites.Count; f++)
                    {
                        Dendrite currentDendrite = downStreamDendrites[f];
                        Neuron downStreamNeuron = currentDendrite.DownStreamNeuron;

                        double delta = downStreamNeuron.BatchErrors.Last();
                        double weight = currentDendrite.Weight;
                        double error = delta * weight;
                        errorSum += error;
                    }

                    thisLayerNeuron.BatchErrors.Add(errorSum * Math.Sigmoid.ComputeDerivative(input));
                }

                nextLayer = hiddenLayer;
            }

            return totalNetworkError;
        }

        public double[] Execute(double[] inputs)
        {
            Feedforward(inputs);
            return OutputLayer.Neurons.Select(n => ((Neuron)n).Activation).ToArray();
        }

        public double[] Feedforward(double[] x)
        {
            for (int d = 0; d < x.Length; d++)
            {
                (InputLayer.Neurons[d]).Activation = x[d];
            }

            foreach (HiddenLayer hiddenLayer in HiddenLayers)
            {
                foreach (Neuron currentLayerNeuron in hiddenLayer.Neurons)
                {
                    currentLayerNeuron.ComputeOutput();
                }
            }

            List<double> output = new List<double>();
            foreach (Neuron currentLayerNeuron in OutputLayer.Neurons)
            {
                output.Add(currentLayerNeuron.ComputeOutput());
            }

            return output.ToArray();
        }
    }
}
