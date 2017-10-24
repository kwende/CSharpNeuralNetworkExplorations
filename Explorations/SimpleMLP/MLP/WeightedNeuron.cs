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

        private WeightedNeuron()
        {
            Weights = new List<Weight>();
        }

        public static WeightedNeuron BuildNeuron(Layer previousLayer)
        {
            Random rand = new Random();
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
