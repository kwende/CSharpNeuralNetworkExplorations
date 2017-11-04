using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMLP.MLP
{
    public class Network
    {
        public InputLayer InputLayer { get; private set; }
        public List<HiddenLayer> HiddenLayers { get; private set; }
        public OutputLayer OutputLayer { get; private set; }

        private Network()
        {
            HiddenLayers = new List<HiddenLayer>();
        }

        public static Network BuildNetwork(int inputNeuronCount, int outputNeuronCount, params int[] hiddenLayerCounts)
        {
            Math.RandomNormal rand = new Math.RandomNormal(0, 1);

            Network network = new Network();

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
            layersToUpdate.Add(InputLayer);
            foreach (HiddenLayer hiddenLayer in HiddenLayers)
            {
                layersToUpdate.Add(hiddenLayer);
            }
            layersToUpdate.Add(OutputLayer);

            for (int c = layersToUpdate.Count - 1; c >= 0; c--)
            {
                foreach (Neuron neuron in layersToUpdate[c].Neurons)
                {
                    double delta = neuron.BatchErrors.Average();
                    neuron.BatchErrors.Clear();

                    neuron.Bias = neuron.Bias - (stepSize * delta);

                    //if (neuron.UniqueName == "8")
                    //{
                    //    File.AppendAllText("C:/users/brush/desktop/8_bias.csv", $"{delta},{neuron.Bias}\n");
                    //}

                    foreach (Dendrite dendrite in neuron.Dendrites)
                    {
                        dendrite.Weight = dendrite.Weight - stepSize * (delta * ((Neuron)dendrite.UpStreamNeuron).Activation);
                    }
                }
            }
        }

        public double Backpropagation(double[] expectedValues)
        {
            double averagedOutputDelta = 0.0;
            // Compute error for the output neurons to get the ball rolling. 
            // See https://github.com/kwende/CSharpNeuralNetworkExplorations/blob/master/Explorations/SimpleMLP/Documentation/OutputNeuronErrors.png
            for (int d = 0; d < expectedValues.Length; d++)
            {
                Neuron outputNeuronBeingExamined = OutputLayer.Neurons[d];
                double expectedOutput = expectedValues[d];
                double actualOutput = outputNeuronBeingExamined.Activation;

                double delta = (Math.CostFunction.ComputeDerivative(actualOutput, expectedOutput) *
                    Math.Sigmoid.ComputeDerivative(actualOutput));

                outputNeuronBeingExamined.BatchErrors.Add(delta);

                averagedOutputDelta += delta;
            }
            averagedOutputDelta /= (expectedValues.Length * 1.0);

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
                    List<Dendrite> downStreamDendrites = nextLayer.Neurons.SelectMany(n => n.Dendrites.Where(l => l.UpStreamNeuron == thisLayerNeuron)).ToList();
                    for (int f = 0; f < downStreamDendrites.Count; f++)
                    {
                        Dendrite currentDendrite = downStreamDendrites[f];
                        Neuron downStreamNeuron = currentDendrite.DownStreamNeuron;

                        double delta = downStreamNeuron.BatchErrors.Last();
                        double weight = currentDendrite.Weight;

                        errorSum += delta * weight;
                    }

                    thisLayerNeuron.BatchErrors.Add(errorSum * Math.Sigmoid.ComputeDerivative(input));
                }

                nextLayer = hiddenLayer;
            }

            // Input layer errors. 
            for (int e = 0; e < InputLayer.Neurons.Count; e++)
            {
                Neuron thisLayerNeuron = (Neuron)InputLayer.Neurons[e];
                double input = thisLayerNeuron.TotalInput;

                double errorSum = 0.0;
                List<Dendrite> downStreamDendrites = nextLayer.Neurons.SelectMany(n => n.Dendrites.Where(l => l.UpStreamNeuron == thisLayerNeuron)).ToList();
                for (int f = 0; f < downStreamDendrites.Count; f++)
                {
                    Dendrite currentDendrite = downStreamDendrites[f];
                    Neuron downStreamNeuron = currentDendrite.DownStreamNeuron;

                    double error = downStreamNeuron.BatchErrors.Last();
                    double weight = currentDendrite.Weight;

                    errorSum += error * weight;
                }

                double thisLayerNeuronError = Math.Sigmoid.ComputeDerivative(input) * errorSum;
                thisLayerNeuron.BatchErrors.Add(thisLayerNeuronError * Math.Sigmoid.ComputeDerivative(input));
            }

            return averagedOutputDelta;
        }

        public void SetInputLayer(double[] x)
        {
            // set the inputs.
            for (int d = 0; d < x.Length; d++)
            {
                (InputLayer.Neurons[d]).Activation = x[d];
            }
        }

        public double[] Execute(double[] inputs)
        {
            SetInputLayer(inputs);
            Feedforward();
            return OutputLayer.Neurons.Select(n => ((Neuron)n).Activation).ToArray();
        }

        public void Feedforward()
        {
            Layer previousLayer = (Layer)InputLayer;

            foreach (HiddenLayer hiddenLayer in HiddenLayers)
            {
                foreach (Neuron currentLayerNeuron in hiddenLayer.Neurons)
                {
                    foreach (Neuron previousLayerNeuron in previousLayer.Neurons)
                    {
                        currentLayerNeuron.Inputs.Add(previousLayerNeuron.Activation);
                    }

                    currentLayerNeuron.ComputeOutput();
                }

                previousLayer = hiddenLayer;
            }

            //foreach (WeightedNeuron currentLayerNeuron in OutputLayer.Neurons)
            for (int i = 0; i < OutputLayer.Neurons.Count; i++)
            {
                Neuron currentLayerNeuron = (Neuron)OutputLayer.Neurons[i];

                foreach (Neuron previousLayerNeuron in previousLayer.Neurons)
                {
                    currentLayerNeuron.Inputs.Add(previousLayerNeuron.Activation);
                }

                currentLayerNeuron.ComputeOutput();
            }
        }
    }
}
