using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMLP.MLP
{
    public class Dendrite
    {
        public Neuron UpStreamNeuron { get; private set; }
        public Neuron DownStreamNeuron { get; private set; }
        public double Weight { get; private set; }
        private Dendrite()
        {
        }
        public static Dendrite BuildWeight(Neuron upstreamNeuron, Neuron downStreamNeuron, double weight)
        {
            Dendrite toReturn = new Dendrite();

            toReturn.DownStreamNeuron = downStreamNeuron;
            toReturn.UpStreamNeuron = upstreamNeuron;
            toReturn.Weight = weight;

            return toReturn;
        }
    }
}
