using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    [Serializable]
    public class HiddenLayer : Layer
    {
        private HiddenLayer() : base()
        {
        }
        public static HiddenLayer BuildHiddenLayer(Math.RandomNormal rand, Layer previousLayer, int numberOfNeurons)
        {
            HiddenLayer toReturn = new HiddenLayer();

            for (int c = 0; c < numberOfNeurons; c++)
            {
                toReturn.Neurons.Add(Neuron.BuildNeuron(rand, previousLayer));
            }

            return toReturn;
        }
    }
}
