using SimpleMLP.MLP.Exceptionx;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLP
{
    [Serializable]
    public class Neuron
    {
        private static int _uniqueId = 0;

        private static int GetNextUniqueId()
        {
            return Interlocked.Increment(ref _uniqueId);
        }

        public double Bias { get; set; }
        public List<Dendrite> UpstreamDendrites { get; set; }
        public List<Dendrite> DownstreamDendrites { get; set; }
        public double TotalInput { get; set; }
        public double Activation { get; set; }
        public List<double> BatchErrorsWrtBias { get; set; }
        public string UniqueName { get; private set; }

        private Neuron()
        {
            UpstreamDendrites = new List<Dendrite>();
            DownstreamDendrites = new List<Dendrite>();
            BatchErrorsWrtBias = new List<double>(1000);
            UniqueName = GetNextUniqueId().ToString();
        }

        public double ComputeExecutionOutput(double probabilityOfDropout)
        {
            if (UpstreamDendrites.Count > 0)
            {
                double k = 0.0;
                for (int c = 0; c < UpstreamDendrites.Count; c++)
                {
                    Neuron upstreamNeuron = UpstreamDendrites[c].UpStreamNeuron;
                    k += upstreamNeuron.Activation * UpstreamDendrites[c].Weight;
                }

                TotalInput = (k + Bias);

                Activation = Math.Sigmoid.Compute(TotalInput);
            }

            return Activation;
        }

        public double ComputeTrainingOutput(double dropOutBit = 1)
        {
            if (UpstreamDendrites.Count > 0)
            {
                double k = 0.0;
                for (int c = 0; c < UpstreamDendrites.Count; c++)
                {
                    Neuron upstreamNeuron = UpstreamDendrites[c].UpStreamNeuron;
                    k += upstreamNeuron.Activation * UpstreamDendrites[c].Weight;
                }

                TotalInput = (k + Bias);


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

                    Dendrite dendrite = Dendrite.BuildWeight(previousNeuron, toReturn, rand.Next());

                    toReturn.UpstreamDendrites.Add(dendrite);
                    previousNeuron.DownstreamDendrites.Add(dendrite);
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
