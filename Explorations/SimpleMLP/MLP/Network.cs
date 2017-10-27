using System;
using System.Collections.Generic;
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

        public void Train(double[,] x, double[,] y)
        {
            if (x.GetLength(1) != InputLayer.Neurons.Count)
            {
                throw new ArgumentOutOfRangeException("inputs",
                    "Number of inputs supplied doesn't match the size of the input layer.");
            }

            for (int t = 0; t < x.GetLength(0); t++)
            {
                for (int d = 0; d < x.GetLength(1); d++)
                {
                    ((InputNeuron)InputLayer.Neurons[d]).Output = x[t, d];
                }

                double[] outputs = Feedforward();
                double error = 0;
                for (int d = 0; d < outputs.Length; d++)
                {
                    double output = outputs[d];
                    double expectedOutput = y[t, d];

                    Math.CostFunction.ComputeDerivative(output, expectedOutput)
                }
            }
        }

        private double[] Feedforward()
        {
            Layer previousLayer = (Layer)InputLayer;

            foreach (HiddenLayer hiddenLayer in HiddenLayers)
            {
                foreach (WeightedNeuron currentLayerNeuron in hiddenLayer.Neurons)
                {
                    currentLayerNeuron.Inputs.Clear();

                    foreach (Neuron previousLayerNeuron in previousLayer.Neurons)
                    {
                        currentLayerNeuron.Inputs.Add(previousLayerNeuron.Output);
                    }

                    currentLayerNeuron.ComputeOutput();
                }

                previousLayer = hiddenLayer;
            }

            foreach (WeightedNeuron currentLayerNeuron in OutputLayer.Neurons)
            {
                currentLayerNeuron.Inputs.Clear();

                foreach (Neuron previousLayerNeuron in previousLayer.Neurons)
                {
                    currentLayerNeuron.Inputs.Add(previousLayerNeuron.Output);
                }

                currentLayerNeuron.ComputeOutput();
            }

            return OutputLayer.Neurons.Select(n => n.Output).ToArray();
        }
    }
}
