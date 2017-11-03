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
                foreach (WeightedNeuron neuron in layersToUpdate[c].Neurons)
                {
                    neuron.BatchErrors.Average();
                }
            }
        }

        public void Backpropagation(double[] expectedValues)
        {
            // Compute error for the output neurons to get the ball rolling. 
            // See https://github.com/kwende/CSharpNeuralNetworkExplorations/blob/master/Explorations/SimpleMLP/Documentation/OutputNeuronErrors.png
            for (int d = 0; d < expectedValues.Length; d++)
            {
                Neuron outputNeuronBeingExamined = OutputLayer.Neurons[d];
                double expectedOutput = expectedValues[d];
                double actualOutput = outputNeuronBeingExamined.Output;

                double error = (Math.CostFunction.ComputeDerivative(actualOutput, expectedOutput) *
                    Math.Sigmoid.ComputeDerivative(actualOutput));

                outputNeuronBeingExamined.BatchErrors.Add(error);
            }

            // Compute error for each neuron in each layer moving backwards (backprop). 
            Layer nextLayer = OutputLayer;
            for (int d = HiddenLayers.Count - 1; d >= 0; d--)
            {
                HiddenLayer hiddenLayer = HiddenLayers[d];
                for (int e = 0; e < hiddenLayer.Neurons.Count; e++)
                {
                    WeightedNeuron thisLayerNeuron = (WeightedNeuron)hiddenLayer.Neurons[e];
                    double input = thisLayerNeuron.TotalInput;

                    double errorSum = 0.0;
                    for (int f = 0; f < nextLayer.Neurons.Count; f++)
                    {
                        WeightedNeuron nextLayerNeuron = (WeightedNeuron)nextLayer.Neurons[f];

                        double error = nextLayerNeuron.BatchErrors.Last();
                        double weight = nextLayerNeuron.Dendrites[e].Weight;

                        errorSum += error * weight;
                    }

                    double thisLayerNeuronError = Math.Sigmoid.ComputeDerivative(input) * errorSum;
                    thisLayerNeuron.BatchErrors.Add(thisLayerNeuronError);
                }
            }


        }

        //public void Train(double[,] x, double[,] y)
        //{
        //    if (x.GetLength(1) != InputLayer.Neurons.Count)
        //    {
        //        throw new ArgumentOutOfRangeException("inputs",
        //            "Number of inputs supplied doesn't match the size of the input layer.");
        //    }

        //    int numberOfTrainingExamples = x.GetLength(0);

        //    for (int t = 0; t < numberOfTrainingExamples; t++)
        //    {
        //        // set input. 
        //        SetInputLayer(t, x);

        //        // feed forward. get outputs. 
        //        double[] outputLayerValues = Feedforward();

        //        // compute the error of the outputs. 
        //        double[] outputLayerErrors =
        //            ComputeErrorForOutputNeurons(t, outputLayerValues, y);

        //        // assign the errors to the output neurons. 
        //        for (int n = 0; n < outputLayerErrors.Length; n++)
        //        {

        //        }
        //    }
        //}

        //private double[] ComputeErrorForOutputNeurons(int trainingIndex, double[] outputs, double[,] y)
        //{
        //    // 

        //    int numberOfTrainingExamples = y.GetLength(0);

        //    double[] errors = new double[outputs.Length];
        //    for (int d = 0; d < outputs.Length; d++)
        //    {
        //        double output = outputs[d];
        //        double expectedOutput = y[trainingIndex, d];

        //        errors[d] = (Math.CostFunction.ComputeDerivative(output, expectedOutput) *
        //            Math.Sigmoid.ComputeDerivative(output)) / (numberOfTrainingExamples * 1.0);
        //    }

        //    return errors;
        //}

        public void SetInputLayer(double[] x)
        {
            // set the inputs.
            for (int d = 0; d < x.Length; d++)
            {
                ((InputNeuron)InputLayer.Neurons[d]).Output = x[d];
            }
        }

        public void Feedforward()
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


        }
    }
}
