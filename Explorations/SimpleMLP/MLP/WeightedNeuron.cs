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
        public List<Dendrite> Dendrites { get; private set; }
        public List<double> Inputs { get; private set; }
        public double TotalInput { get; set; }

        private WeightedNeuron()
        {
            Dendrites = new List<Dendrite>();
            Inputs = new List<double>();
        }

        public void ComputeOutput()
        {
            if (Inputs.Count != Dendrites.Count)
            {
                throw new InputAndWeightCountMismatchException();
            }

            double k = 0.0;
            for (int c = 0; c < Inputs.Count; c++)
            {
                k += Inputs[c] * Dendrites[c].Weight;
            }

            TotalInput = k + Bias;

            Output = Math.Sigmoid.Compute(TotalInput);
        }

        public static WeightedNeuron BuildNeuron(Math.RandomNormal rand, Layer previousLayer)
        {
            WeightedNeuron toReturn = new WeightedNeuron();

            toReturn.Bias = rand.Next();

            for (int c = 0; c < previousLayer.Neurons.Count; c++)
            {
                Neuron previousNeuron = previousLayer.Neurons[c];
                toReturn.Dendrites.Add(Dendrite.BuildWeight(previousNeuron, toReturn, rand.Next()));
            }

            return toReturn;
        }
    }
}
