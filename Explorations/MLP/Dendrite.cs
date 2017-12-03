using Common.Interfaces;
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
        public double SumOfErrorsWrtWeights { get; set; }
        public void AddError(double error)
        {
            SumOfErrorsWrtWeights += error;
        }

        public void ClearError()
        {
            SumOfErrorsWrtWeights = 0.0;
        }

        private Dendrite()
        {
            SumOfErrorsWrtWeights = 0;
        }
        public static Dendrite BuildDendrite(Neuron upstreamNeuron, Neuron downStreamNeuron, IWeightBuilder weightBuilder)
        {
            Dendrite toReturn = new Dendrite();

            toReturn.DownStreamNeuron = downStreamNeuron;
            toReturn.UpStreamNeuron = upstreamNeuron;
            toReturn.Weight = weightBuilder.BuildWeight();

            return toReturn;
        }
        public override string ToString()
        {
            return $"{UpStreamNeuron}-->{DownStreamNeuron}";
        }
    }
}
