using SimpleMLP.MLP.Exceptionx;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMLP.MLP
{
    public class WeightedNeuron : Neuron
    {
        public double Bias { get; private set; }
        public List<Weight> Weights { get; private set; }
        public List<double> Inputs { get; private set; }

        private WeightedNeuron()
        {
            Weights = new List<Weight>();
            Inputs = new List<double>();
        }

        public void ComputeOutput()
        {
            if (Inputs.Count != Weights.Count)
            {
                throw new InputAndWeightCountMismatchException();
            }

            double k = 0.0;
            for (int c = 0; c < Inputs.Count; c++)
            {
                k += Inputs[c] * Weights[c].Value;
            }

            k += Bias;

            Output = k;
        }

        public static WeightedNeuron BuildNeuron(Random rand, Layer previousLayer)
        {
            WeightedNeuron toReturn = new WeightedNeuron();

            toReturn.Bias = rand.NextDouble();

            for (int c = 0; c < previousLayer.Neurons.Count; c++)
            {
                Neuron previousNeuron = previousLayer.Neurons[c];
                toReturn.Weights.Add(Weight.BuildWeight(previousNeuron, toReturn, rand.NextDouble()));
            }

            return toReturn;
        }
    }
}
