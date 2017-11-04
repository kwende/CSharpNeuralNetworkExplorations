using SimpleMLP.MLP.Exceptionx;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace SimpleMLP.MLP
{
    public class Neuron
    {
        private static int _uniqueId = 0;

        private static int GetNextUniqueId()
        {
            return Interlocked.Increment(ref _uniqueId);
        }

        public double Bias { get; set; }
        public List<Dendrite> Dendrites { get; set; }
        public double TotalInput { get; set; }
        public double Activation { get; set; }
        public List<double> BatchErrors { get; set; }
        public string UniqueName { get; private set; }

        private Neuron()
        {
            Dendrites = new List<Dendrite>();
            BatchErrors = new List<double>();
            UniqueName = GetNextUniqueId().ToString();
        }

        public double ComputeOutput()
        {
            if (Dendrites.Count > 0)
            {
                double k = 0.0;
                for (int c = 0; c < Dendrites.Count; c++)
                {
                    k += Dendrites[c].UpStreamNeuron.Activation * Dendrites[c].Weight;
                }

                TotalInput = k + Bias;

                Activation = Math.Sigmoid.Compute(TotalInput);
            }

            return Activation;
        }

        public static Neuron BuildNeuron(Math.RandomNormal rand, Layer previousLayer)
        {
            Neuron toReturn = new Neuron();

            toReturn.Bias = rand.Next();

            if (previousLayer != null)
            {
                for (int c = 0; c < previousLayer.Neurons.Count; c++)
                {
                    Neuron previousNeuron = previousLayer.Neurons[c];
                    toReturn.Dendrites.Add(Dendrite.BuildWeight(previousNeuron, toReturn, rand.Next()));
                }
            }

            return toReturn;
        }

        public override string ToString()
        {
            return $"Neuron {UniqueName}";
        }
    }
}
