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
        public static OutputLayer BuildOutputLayer(HiddenLayer previousLayer, int numberOfNeurons)
        {
            OutputLayer toReturn = new OutputLayer();
            for (int c = 0; c < numberOfNeurons; c++)
            {
                toReturn.Neurons.Add(WeightedNeuron.BuildNeuron(previousLayer));
            }
            return toReturn;
        }
    }
}
