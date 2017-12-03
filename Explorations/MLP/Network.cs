﻿using Common;
using Common.Exceptions;
using Common.Interfaces;
using Math.WeightInitialization;
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
    public class Network : ITrainedNeuralNetwork
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
            WeightIntializerType weightIntializerType,
            DropoutLayerOptions dropoutLayerOptions,
            int inputNeuronCount, int outputNeuronCount, params int[] hiddenLayerCounts)
        {
            Network network = new Network(costFunction, regularizationFunction,
                dropoutLayerOptions, random);

            network.InputLayer = InputLayer.BuildInputLayer(null, inputNeuronCount, random);

            Layer previousLayer = network.InputLayer;
            int dropoutLayerIndex = 1;
            bool isDropoutLayer = false;
            IWeightBuilder weightBuilder = null; 
            for (int c = 0; c < hiddenLayerCounts.Length; c++)
            {
                isDropoutLayer = dropoutLayerOptions.DropoutLayerIndices.Contains(dropoutLayerIndex);
                int currentLayerCount = hiddenLayerCounts[c];

                switch(weightIntializerType)
                {
                    case WeightIntializerType.RandomGaussianWithNeuronCount:
                        weightBuilder = new RandomGaussianWithNeuronCount(previousLayer.Neurons.Count, 0, random);
                        break;
                    case WeightIntializerType.RandomNormal:
                        weightBuilder = new RandomGaussian(0, 1, random); 
                        break; 
                }

                HiddenLayer hiddenLayer = HiddenLayer.BuildHiddenLayer(weightBuilder, previousLayer,
                    currentLayerCount, isDropoutLayer ? dropoutLayerOptions.ProbabilityOfDropout : 0, random);

                network.HiddenLayers.Add(hiddenLayer);
                previousLayer = hiddenLayer;

                dropoutLayerIndex++;
            }

            isDropoutLayer = dropoutLayerOptions.DropoutLayerIndices.Contains(dropoutLayerIndex);

            switch (weightIntializerType)
            {
                case WeightIntializerType.RandomGaussianWithNeuronCount:
                    weightBuilder = new RandomGaussianWithNeuronCount(previousLayer.Neurons.Count, 0, random);
                    break;
                case WeightIntializerType.RandomNormal:
                    weightBuilder = new RandomGaussian(0, 1, random);
                    break;
            }

            network.OutputLayer = OutputLayer.BuildOutputLayer(weightBuilder, (HiddenLayer)previousLayer,
                outputNeuronCount, isDropoutLayer ? dropoutLayerOptions.ProbabilityOfDropout : 0, random);

            return network;
        }

        public void UpdateNetwork(double stepSize, int sizeOfTrainingData, int batchSize)
        {
            // Generate the list of layers to update, starting from the
            // beginnign to the end (the output layer). 
            List<Layer> layersToUpdate = new List<Layer>();
            foreach (HiddenLayer hiddenLayer in HiddenLayers)
            {
                layersToUpdate.Add(hiddenLayer);
            }
            layersToUpdate.Add(OutputLayer);

            // start from the last layer and work my way backward. 
            for (int c = layersToUpdate.Count - 1; c >= 0; c--)
            {
                // take each of the neurons in the layer and update the bias
                // take each of the dendrites attached to the neuron and update the weight. 
                for (int n = 0; n < layersToUpdate[c].Neurons.Count; n++)
                {
                    // the current neuron being examined. 
                    Neuron thisNeuron = layersToUpdate[c].Neurons[n];
                    bool isNeuronDropped = layersToUpdate[c].DropOutMask[n] == 0;

                    // get the sum of all the errors of this neuron across the batch. 
                    // divide by the batch size. This is the average error for the neuron. 
                    double averageNeuronError = thisNeuron.SumOfErrorsOfNeuron /= (batchSize * 1.0);
                    thisNeuron.ClearError();

                    // updating the weight is just subtracting the bias (the delta is the gradient 
                    // of the bias). dC/db = delta. Make sure to weight the delta by the step size. 
                    thisNeuron.Bias = thisNeuron.Bias - (stepSize * averageNeuronError);

                    for (int d = 0; d < thisNeuron.UpstreamDendrites.Count; d++)
                    {
                        Dendrite dendrite = thisNeuron.UpstreamDendrites[d];

                        double averageErrorWrtWeight = dendrite.SumOfErrorsWrtWeights / (batchSize * 1.0);
                        dendrite.ClearError();

                        double regularization = 0.0;
                        if (_regularizationFunction != null && !isNeuronDropped)
                        {
                            regularization = _regularizationFunction.Compute(dendrite.Weight, sizeOfTrainingData);
                        }

                        dendrite.Weight = dendrite.Weight -
                            (stepSize * (averageErrorWrtWeight + regularization));
                    }
                }
            }
        }

        public double Backpropagation(double[] expectedValues)
        {
            double totalNetworkCost = 0.0;
            // Compute error for the output neurons to get the ball rolling. 
            // See https://github.com/kwende/CSharpNeuralNetworkExplorations/blob/master/Explorations/SimpleMLP/Documentation/OutputNeuronErrors.png
            for (int d = 0; d < expectedValues.Length; d++)
            {
                Neuron outputNeuronBeingExamined = OutputLayer.Neurons[d];
                double expectedOutput = expectedValues[d];
                double actualOutput = outputNeuronBeingExamined.Activation;
                double actualInput = outputNeuronBeingExamined.TotalInput;

                double cost = _costFunction.Compute(expectedOutput, actualOutput);
                totalNetworkCost += cost;

                double errorRelativeToActivation =
                    (_costFunction.ComputeDerivativeWRTActivation(actualOutput, expectedOutput));

                double errorWrtToNeuron = errorRelativeToActivation * Math.Sigmoid.ComputeDerivative(actualInput);

                outputNeuronBeingExamined.AddError(errorWrtToNeuron);

                for (int e = 0; e < outputNeuronBeingExamined.UpstreamDendrites.Count; e++)
                {
                    Dendrite dendrite = outputNeuronBeingExamined.UpstreamDendrites[e];
                    Neuron upstreamNeuron = (Neuron)dendrite.UpStreamNeuron;
                    double errorRelativeToWeight = (errorWrtToNeuron * upstreamNeuron.Activation);

                    dendrite.AddError(errorRelativeToWeight);
                }
            }

            // Compute error for each neuron in each layer moving backwards (backprop). 
            for (int d = HiddenLayers.Count - 1; d >= 0; d--)
            {
                HiddenLayer hiddenLayer = HiddenLayers[d];
                for (int e = 0; e < hiddenLayer.Neurons.Count; e++)
                {
                    Neuron thisNeuron = (Neuron)hiddenLayer.Neurons[e];
                    double dropoutBit = hiddenLayer.DropOutMask[e];

                    double input = thisNeuron.TotalInput;

                    double errorSum = 0.0;
                    List<Dendrite> downStreamDendrites = thisNeuron.DownstreamDendrites;

                    for (int f = 0; f < downStreamDendrites.Count; f++)
                    {
                        Dendrite currentDendrite = downStreamDendrites[f];
                        Neuron downStreamNeuron = currentDendrite.DownStreamNeuron;

                        double delta = downStreamNeuron.CurrentNeuronError;
                        double weight = currentDendrite.Weight;
                        errorSum += delta * weight;
                    }

                    double errorWrtToThisNeuron = errorSum * Math.Sigmoid.ComputeDerivative(input) * dropoutBit;
                    thisNeuron.AddError(errorWrtToThisNeuron);

                    for (int f = 0; f < thisNeuron.UpstreamDendrites.Count; f++)
                    {
                        Dendrite dendrite = thisNeuron.UpstreamDendrites[f];
                        Neuron upstreamNeuron = (Neuron)dendrite.UpStreamNeuron;
                        double errorRelativeToWeight = (errorWrtToThisNeuron * upstreamNeuron.Activation);
                        dendrite.AddError(errorRelativeToWeight);
                    }
                }
            }

            return totalNetworkCost;
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

        public void UpdateDroputLayers()
        {
            foreach (HiddenLayer hiddenLayer in HiddenLayers)
            {
                if (hiddenLayer.IsDropoutLayer)
                {
                    hiddenLayer.UpdateDropoutMask();
                }
            }

            if (OutputLayer.IsDropoutLayer)
            {
                OutputLayer.UpdateDropoutMask();
            }
        }

        public double[] Feedforward(double[] x)
        {
            if (x.Length != InputLayer.Neurons.Count)
            {
                throw new DataAndInputLayerSizeMismatchException(
                    "Resize the input layer to match the size of the training data.");
            }

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
