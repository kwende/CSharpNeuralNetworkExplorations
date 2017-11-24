using Common.Interfaces;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace MLP
{
    [Serializable]
    public class Network
    {
        public InputLayer InputLayer { get; set; }
        public List<HiddenLayer> HiddenLayers { get; set; }
        public OutputLayer OutputLayer { get; set; }
        public Random NetworkRandom { get; private set; }

        private ICostFunction _costFunction;
        private IRegularizationFunction _regularizationFunction;
        private DropoutOptions _dropoutOptions;

        private Network(ICostFunction costFunction,
            IRegularizationFunction regularizationFunction,
            Random rand,
            DropoutOptions dropoutOptions)
        {
            HiddenLayers = new List<HiddenLayer>();

            _costFunction = costFunction;
            _regularizationFunction = regularizationFunction;
            _dropoutOptions = dropoutOptions;

            NetworkRandom = rand;
        }

        public static Network BuildNetwork(Random random,
            ICostFunction costFunction, IRegularizationFunction regularizationFunction,
            DropoutOptions dropoutOptions,
            int inputNeuronCount, int outputNeuronCount, params int[] hiddenLayerCounts)
        {
            Network network = new Network(costFunction, regularizationFunction,
                random, dropoutOptions);

            Math.RandomNormal rand = new Math.RandomNormal(0, 1, network.NetworkRandom);

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

        public void UpdateNetwork(double stepSize, int sizeOfTrainingData)
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
                    if (!neuron.DropOut)
                    {
                        double delta = neuron.BatchErrors.Average();
                        neuron.BatchErrors.Clear();

                        neuron.Bias = neuron.Bias - stepSize * delta;

                        foreach (Dendrite dendrite in neuron.UpstreamDendrites)
                        {
                            Neuron upstreamNeuron = (Neuron)dendrite.UpStreamNeuron;
                            if(!upstreamNeuron.DropOut)
                            {
                                double changeInErrorRelativeToWeight =
                                    (delta * upstreamNeuron.Activation);

                                double regularization = 0.0;
                                if (_regularizationFunction != null)
                                {
                                    regularization = _regularizationFunction.Compute(dendrite.Weight, sizeOfTrainingData);
                                }

                                dendrite.Weight = dendrite.Weight -
                                    stepSize * (changeInErrorRelativeToWeight + regularization);
                            }
                        }
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
                    if (!thisLayerNeuron.DropOut)
                    {
                        double input = thisLayerNeuron.TotalInput;

                        double errorSum = 0.0;
                        List<Dendrite> downStreamDendrites = thisLayerNeuron.DownstreamDendrites;

                        for (int f = 0; f < downStreamDendrites.Count; f++)
                        {
                            Dendrite currentDendrite = downStreamDendrites[f];
                            Neuron downStreamNeuron = currentDendrite.DownStreamNeuron;

                            if (!downStreamNeuron.DropOut)
                            {
                                double delta = downStreamNeuron.BatchErrors.Last();
                                double weight = currentDendrite.Weight;
                                double error = delta * weight;
                                errorSum += error;
                            }
                        }

                        thisLayerNeuron.BatchErrors.Add(errorSum * Math.Sigmoid.ComputeDerivative(input));
                    }
                }

                nextLayer = hiddenLayer;
            }

            return totalNetworkError;
        }

        public double[] Execute(double[] inputs)
        {
            for (int d = 0; d < inputs.Length; d++)
            {
                (InputLayer.Neurons[d]).Activation = inputs[d];
            }

            int layerIndex = 1;
            foreach (HiddenLayer hiddenLayer in HiddenLayers)
            {
                if (_dropoutOptions.DropoutLayerIndices.Contains(layerIndex))
                {
                    hiddenLayer.ComputeFullLayerOutputConsideringDropouts(
                        _dropoutOptions.ProbabilityOfNeuronDropout);
                }
                else
                {
                    hiddenLayer.ComputeFullLayerOutput();
                }
            }

            if (_dropoutOptions.DropoutLayerIndices.Contains(layerIndex))
            {
                OutputLayer.ComputeFullLayerOutputConsideringDropouts(
                    _dropoutOptions.ProbabilityOfNeuronDropout);
            }
            else
            {
                OutputLayer.ComputeFullLayerOutput();
            }

            return OutputLayer.Neurons.Select(n => ((Neuron)n).Activation).ToArray();
        }

        public double[] Feedforward(double[] x)
        {
            for (int d = 0; d < x.Length; d++)
            {
                (InputLayer.Neurons[d]).Activation = x[d];
            }

            int layerIndex = 1;
            foreach (HiddenLayer hiddenLayer in HiddenLayers)
            {
                if (_dropoutOptions.DropoutLayerIndices.Contains(layerIndex))
                {
                    hiddenLayer.ComputeLayerOutputWithDropouts(
                        _dropoutOptions.ProbabilityOfNeuronDropout, NetworkRandom);
                }
                else
                {
                    hiddenLayer.ComputeFullLayerOutput();
                }

                layerIndex++;
            }

            List<double> output = new List<double>();

            if (_dropoutOptions.DropoutLayerIndices.Contains(layerIndex))
            {
                OutputLayer.ComputeLayerOutputWithDropouts(_dropoutOptions.ProbabilityOfNeuronDropout,
                    NetworkRandom);
            }
            else
            {
                OutputLayer.ComputeFullLayerOutput();
            }

            return output.ToArray();
        }
    }
}
