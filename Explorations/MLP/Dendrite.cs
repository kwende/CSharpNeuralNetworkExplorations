using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    [Serializable]
    public class Dendrite
    {
        public Neuron UpStreamNeuron { get; set; }
        public Neuron DownStreamNeuron { get; set; }
        public double Weight { get; set; }
        public List<double> BatchErrorsWrtWeights { get; set; }

        private Dendrite()
        {
            BatchErrorsWrtWeights = new List<double>(10000);
        }
        public static Dendrite BuildWeight(Neuron upstreamNeuron, Neuron downStreamNeuron, double weight)
        {
            Dendrite toReturn = new Dendrite();

            toReturn.DownStreamNeuron = downStreamNeuron;
            toReturn.UpStreamNeuron = upstreamNeuron;
            toReturn.Weight = weight;

            return toReturn;
        }
        public override string ToString()
        {
            return $"{UpStreamNeuron}-->{DownStreamNeuron}";
        }
    }
}
