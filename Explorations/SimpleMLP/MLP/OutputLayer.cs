using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMLP.MLP
{
    public class OutputLayer : Layer
    {
        private OutputLayer() : base()
        {
        }
        public static OutputLayer BuildOutputLayer(Math.RandomNormal rand, HiddenLayer previousLayer, int numberOfNeurons)
        {
            OutputLayer toReturn = new OutputLayer();
            for (int c = 0; c < numberOfNeurons; c++)
            {
                toReturn.Neurons.Add(Neuron.BuildNeuron(rand, previousLayer));
            }
            return toReturn;
        }
    }
}
