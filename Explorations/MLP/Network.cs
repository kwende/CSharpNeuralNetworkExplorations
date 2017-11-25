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
        private DropoutLayerOptions _dropoutLayerOptions;

        private Network(ICostFunction costFunction,
            IRegularizationFunction regularizationFunction,
            DropoutLayerOptions dropoutLayerOptions,
            Random rand)
        {
            HiddenLayers = new List<HiddenLayer>();

            _costFunction = costFunction;
            _regularizationFunction = regularizationFunction;
            _dropoutLayerOptions = dropoutLayerOptions;

            NetworkRandom = rand;
        }

        public static Network BuildNetwork(Random random,
            ICostFunction costFunction, IRegularizationFunction regularizationFunction,
            DropoutLayerOptions dropoutLayerOptions,
            int inputNeuronCount, int outputNeuronCount, params int[] hiddenLayerCounts)
        {
            Network network = new Network(costFunction, regularizationFunction,
                dropoutLayerOptions, random);

            Math.RandomNormal rand = new Math.RandomNormal(0, 1, network.NetworkRandom);

            network.InputLayer = InputLayer.BuildInputLayer(rand, inputNeuronCount);

            Layer previousLayer = network.InputLayer;
            int dropoutLayerIndex = 1;
            bool isDropoutLayer = false;
            for (int c = 0; c < hiddenLayerCounts.Length; c++)
            {
                isDropoutLayer = dropoutLayerOptions.DropoutLayerIndices.Contains(dropoutLayerIndex);
                int currentLayerCount = hiddenLayerCounts[c];

                HiddenLayer hiddenLayer = HiddenLayer.BuildHiddenLayer(rand, previousLayer,
                    currentLayerCount, isDropoutLayer ? dropoutLayerOptions.ProbabilityOfDropout : 0);

                network.HiddenLayers.Add(hiddenLayer);
                previousLayer = hiddenLayer;

                dropoutLayerIndex++;
            }

            isDropoutLayer = dropoutLayerOptions.DropoutLayerIndices.Contains(dropoutLayerIndex);
            network.OutputLayer = OutputLayer.BuildOutputLayer(rand, (HiddenLayer)previousLayer,
                outputNeuronCount, isDropoutLayer ? dropoutLayerOptions.ProbabilityOfDropout : 0);

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
                //foreach (Neuron neuron in layersToUpdate[c].Neurons)
                for (int n = 0; n < layersToUpdate[c].Neurons.Count; n++)
                {
                    Neuron neuron = layersToUpdate[c].Neurons[n];
                    double dropoutBit = layersToUpdate[c].DropOutMask[n];

                    double delta = neuron.BatchErrors.Average();
                    neuron.BatchErrors.Clear();

                    neuron.Bias = neuron.Bias - (stepSize * delta) * dropoutBit;

                    foreach (Dendrite dendrite in neuron.UpstreamDendrites)
                    {
                        Neuron upstreamNeuron = (Neuron)dendrite.UpStreamNeuron;
                        double changeInErrorRelativeToWeight =
                            (delta * upstreamNeuron.Activation);

                        double regularization = 0.0;
                        if (_regularizationFunction != null)
                        {
                            regularization = _regularizationFunction.Compute(dendrite.Weight, sizeOfTrainingData);
                        }

                        dendrite.Weight = dendrite.Weight -
                            (stepSize * (changeInErrorRelativeToWeight + regularization)) * dropoutBit;
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
            for (int d = HiddenLayers.Count - 1; d >= 0; d--)
            {
                HiddenLayer hiddenLayer = HiddenLayers[d];
                for (int e = 0; e < hiddenLayer.Neurons.Count; e++)
                {
                    Neuron thisLayerNeuron = (Neuron)hiddenLayer.Neurons[e];
                    double input = thisLayerNeuron.TotalInput;

                    double errorSum = 0.0;
                    List<Dendrite> downStreamDendrites = thisLayerNeuron.DownstreamDendrites;

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
            }

            return totalNetworkError;
        }

        public double[] Execute(double[] inputs)
        {
            for (int d = 0; d < inputs.Length; d++)
            {
                (InputLayer.Neurons[d]).Activation = inputs[d];
            }

            foreach (HiddenLayer hiddenLayer in HiddenLayers)
            {
                hiddenLayer.ComputeLayerExecutionOutput();
            }

            OutputLayer.ComputeLayerExecutionOutput();

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
                hiddenLayer.ComputeLayerTrainingOutput();
            }

            List<double> output = new List<double>();
            output.AddRange(OutputLayer.ComputeLayerTrainingOutput());

            return output.ToArray();
        }
    }
}
