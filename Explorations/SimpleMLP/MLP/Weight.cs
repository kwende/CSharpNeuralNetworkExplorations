using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMLP.MLP
{
    public class Weight
    {
        public Neuron UpStreamNeuron { get; private set; }
        public Neuron DownStreamNeuron { get; private set; }
        public double Value { get; private set; }
        private Weight()
        {
        }
        public static Weight BuildWeight(Neuron upstreamNeuron, Neuron downStreamNeuron, double value)
        {
            Weight toReturn = new Weight();

            toReturn.DownStreamNeuron = downStreamNeuron;
            toReturn.UpStreamNeuron = upstreamNeuron;
            toReturn.Value = value;

            return toReturn;
        }
    }
}
